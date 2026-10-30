"""
Stage 2.5 of the chopify pipeline - decide WHERE a clip gets B-roll cutaways.

Reads work/transcript.json + work/segments.json and writes work/broll.json:

    [{"start": 134.2, "end": 187.6,
      "cuts": [{"t": 141.3, "dur": 2.4, "query": "stock market crash"}, ...]}]

All times are ABSOLUTE source-video seconds (same clock as segments.json), so
render_clips.py can re-time them after --tighten removes filler/silence.

Two planners, both free:
  1. heuristic (default) - picks concrete nouns / proper nouns from the transcript
     and anchors a cutaway just before the word is spoken.
  2. --llm MODEL - asks a LOCAL Ollama server for the cutaway list; falls back to
     the heuristic if Ollama is unreachable or returns junk.

Both go through enforce(), which guarantees an edit that feels like an edit and
not a slideshow: the hook stays on the speaker's face, cutaways never overlap,
there is a minimum gap between them, and B-roll never covers more than
--max-pct of the clip.

    python broll_plan.py work
    python broll_plan.py work --llm qwen2.5:7b --max-pct 0.35
"""
import re
import json
import argparse
import urllib.request
from pathlib import Path

from score_clips import OLLAMA_HOST, load_transcript, sentenceize

INTRO = 1.5          # s at the start of a clip that always stays on the speaker (the hook)
OUTRO = 1.0          # s at the end that always stays on the speaker
MIN_DUR = 1.6        # shortest cutaway (s)
MAX_DUR = 3.2        # longest cutaway (s)
DEFAULT_DUR = 2.6
MIN_GAP = 2.5        # min speaker time between two cutaways (s)
LEAD = 0.25          # cutaway appears this long BEFORE the keyword is spoken (s)
DEFAULT_MAX_PCT = 0.40
DEFAULT_MAX_CUTS = 8

STOPWORDS = set("""
a about above after again all also am an and any are around as at be because been before being
below between both but by can could did do does doing done down during each either else even
ever every few for from further get gets getting go goes going gonna gotta got had has have having
he her here heres hers herself him himself his how i if in into is it its itself just know let lets
like make makes making many may me might more most much must my myself no nor not now of off oh
ok okay on once one only or other our ours out over own people really right same say says she should
so some something someone anything everything nothing such than that thats the their theirs them
then there theres these they theyre thing things think this those though through to too under until
up upon us very want wants was way we well were what whats when where which while who whom whose why
will with within without would yeah yes you your yours youre yourself actually basically literally
maybe probably pretty quite kind sort stuff gonna wanna need needs needed time times first second
third next last new good great best bad little big long today tomorrow yesterday always never ever
still back come comes came look looks looking take takes took give gives gave tell tells told talk
talking talked ask asked asking use used using work works working start started starting
instead everybody everyone anybody nobody somebody anyway whole almost another around already
although among behind beside beyond enough however meanwhile nearly neither otherwise perhaps
rather since therefore together toward towards unless whatever whenever whether whoever anymore
anywhere everywhere somewhere nowhere often sometimes usually watched because before after
""".split())

FILLERS = {"um", "uh", "uhh", "umm", "erm", "er", "hmm", "mmm", "mm", "ah", "eh"}

LLM_SYSTEM = (
    "You are a short-form video editor adding B-roll cutaways to a clip. You receive "
    "a timestamped transcript of ONE clip. Pick the moments where a stock video or "
    "photo would make the edit feel richer (a concrete thing, place, person, object "
    "or action being talked about). Rules:\n"
    "- At most {max_cuts} cutaways, each {min_dur}-{max_dur} seconds long.\n"
    "- Never in the first {intro} seconds or last {outro} seconds of the clip.\n"
    "- Cutaways must not overlap and need at least {min_gap} seconds between them.\n"
    "- 'query' is a 2-4 word stock-footage search phrase for something VISUAL and "
    "concrete (no abstract concepts, no names of private people).\n"
    "- 't' is the absolute second the cutaway starts (use the transcript timestamps).\n"
    'Respond with JSON only: {{"cuts": [{{"t": <sec>, "dur": <sec>, "query": "<phrase>"}}]}}'
)


# ------------------------------------------------------------------ helpers

def _norm(w):
    return re.sub(r"[^\w']", "", w.lower())


def _clean_word(w):
    return re.sub(r"[^\w'-]", "", w)


def make_query(sent_words, max_terms=2):
    """Pick up to `max_terms` concrete keywords from one sentence's words.

    Returns (query, anchor_time, score) or None. Proper nouns (capitalised
    mid-sentence) and longer words score higher; abstract-sounding words score
    lower; stopwords, fillers, numbers and very short words are ignored.
    anchor_time is when the first chosen keyword is spoken; score is the sum of the
    chosen keywords' scores (used to rank cutaway candidates best-first).
    """
    scored = []
    for i, w in enumerate(sent_words):
        raw = _clean_word(w["word"].strip())
        tok = _norm(raw)
        if len(tok) < 4 or tok in STOPWORDS or tok in FILLERS or tok.isdigit():
            continue
        score = 1.0
        if i > 0 and raw[:1].isupper():
            score += 2.0                      # proper noun
        if len(tok) >= 7:
            score += 1.0
        if tok.endswith(("ness", "tion", "ment", "ity", "ship", "ism")):
            score -= 1.5                      # abstract -> poor stock footage
        if tok.endswith("ly"):
            score -= 1.0
        scored.append((score, i, tok, w["start"]))
    if not scored:
        return None
    best = sorted(scored, key=lambda x: (-x[0], x[1]))[:max_terms]
    if best[0][0] < 1.0:
        return None
    total = sum(b[0] for b in best)
    best.sort(key=lambda x: x[1])             # keep spoken order
    return " ".join(b[2] for b in best), best[0][3], total


def enforce(cuts, seg_start, seg_end, max_pct=DEFAULT_MAX_PCT, max_cuts=DEFAULT_MAX_CUTS,
            min_gap=MIN_GAP, intro=INTRO, outro=OUTRO, min_dur=MIN_DUR, max_dur=MAX_DUR):
    """Clamp / filter a list of {t, dur, query} so the edit has sane pacing.

    Keeps the earliest-listed cuts when a rule forces a drop, so callers should
    pass cuts ordered best-first; the result is returned sorted by time.
    """
    lo, hi = seg_start + intro, seg_end - outro
    budget = max(0.0, (seg_end - seg_start) * max_pct)
    kept, used = [], 0.0
    for c in cuts:
        try:
            t, d = float(c["t"]), float(c.get("dur", DEFAULT_DUR))
            q = str(c.get("query", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not q:
            continue
        d = max(min_dur, min(max_dur, d))
        t = max(t, lo)
        if t + d > hi:
            d = hi - t
        if d < min_dur:
            continue
        if any(t < k["t"] + k["dur"] + min_gap and t + d + min_gap > k["t"] for k in kept):
            continue
        if used + d > budget + 1e-9:
            continue
        kept.append({"t": round(t, 2), "dur": round(d, 2), "query": q})
        used += d
        if len(kept) >= max_cuts:
            break
    kept.sort(key=lambda k: k["t"])
    return kept


# --------------------------------------------------------------- heuristic

def plan_heuristic(sentences, seg_start, seg_end, **limits):
    """Heuristic planner: one candidate per sentence, best keywords first."""
    cands = []
    for s in sentences:
        if s["end"] <= seg_start or s["start"] >= seg_end:
            continue
        r = make_query(s["words"])
        if not r:
            continue
        query, anchor, score = r
        cands.append({"t": anchor - LEAD, "dur": DEFAULT_DUR, "query": query,
                      "_score": score})
    cands.sort(key=lambda c: -c["_score"])
    return enforce(cands, seg_start, seg_end, **limits)


# --------------------------------------------------------------------- llm

def _ollama_cuts(lines, seg_start, seg_end, model, host, limits, timeout=600):
    """Ask local Ollama for cuts for one clip. Returns a list, or None on failure."""
    sys_msg = LLM_SYSTEM.format(
        max_cuts=limits.get("max_cuts", DEFAULT_MAX_CUTS), min_dur=MIN_DUR, max_dur=MAX_DUR,
        intro=INTRO, outro=OUTRO, min_gap=limits.get("min_gap", MIN_GAP))
    payload = {
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": 0.2},
        "messages": [{"role": "system", "content": sys_msg},
                     {"role": "user", "content":
                      f"Clip runs {seg_start:.1f}s-{seg_end:.1f}s.\n\nTranscript:\n{lines}"}],
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        content = resp.get("message", {}).get("content", "")
        data = json.loads(content)
        cuts = data.get("cuts") if isinstance(data, dict) else data
        return cuts if isinstance(cuts, list) else None
    except Exception as e:                                   # noqa: BLE001
        print(f"  Ollama unavailable ({e}) -> heuristic B-roll plan", flush=True)
        return None


def plan_llm(sentences, seg_start, seg_end, model, host=OLLAMA_HOST, **limits):
    """LLM planner with heuristic fallback. Returns the enforced cut list."""
    inside = [s for s in sentences if s["end"] > seg_start and s["start"] < seg_end]
    lines = "\n".join(f"{s['start']:.1f}-{s['end']:.1f}: {s['text']}" for s in inside)
    raw = _ollama_cuts(lines, seg_start, seg_end, model, host, limits)
    if raw:
        cuts = enforce(raw, seg_start, seg_end, **limits)
        if cuts:
            return cuts
    return plan_heuristic(sentences, seg_start, seg_end, **limits)


# --------------------------------------------------------------------- run

def run(workdir, llm=None, host=OLLAMA_HOST, max_pct=DEFAULT_MAX_PCT,
        max_cuts=DEFAULT_MAX_CUTS, min_gap=MIN_GAP):
    """Plan B-roll for every segment in workdir and write workdir/broll.json."""
    workdir = Path(workdir)
    data = load_transcript(workdir)
    seg_path = workdir / "segments.json"
    if not seg_path.exists():
        raise SystemExit(f"No segments at {seg_path} - run score_clips.py first")
    segments = json.loads(seg_path.read_text(encoding="utf-8-sig"))
    sentences = sentenceize(data["words"])
    limits = {"max_pct": max_pct, "max_cuts": max_cuts, "min_gap": min_gap}

    plan = []
    for i, seg in enumerate(segments, 1):
        s0, s1 = float(seg["start"]), float(seg["end"])
        if llm:
            cuts = plan_llm(sentences, s0, s1, llm, host, **limits)
        else:
            cuts = plan_heuristic(sentences, s0, s1, **limits)
        plan.append({"start": round(s0, 2), "end": round(s1, 2), "cuts": cuts})
        total = sum(c["dur"] for c in cuts)
        print(f"  clip {i}: {len(cuts)} cutaway(s), {total:.1f}s of "
              f"{s1 - s0:.1f}s ({total / max(s1 - s0, 0.1):.0%})", flush=True)
        for c in cuts:
            print(f"      {c['t']:8.2f}  +{c['dur']:.1f}s  {c['query']}", flush=True)

    out = workdir / "broll.json"
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return plan


def main():
    ap = argparse.ArgumentParser(description="Plan B-roll cutaways for chopify clips.")
    ap.add_argument("workdir", nargs="?", default="work")
    ap.add_argument("--llm", nargs="?", const="qwen2.5:7b", default=None, metavar="MODEL",
                    help="plan with a local Ollama model (falls back to heuristic)")
    ap.add_argument("--host", default=OLLAMA_HOST)
    ap.add_argument("--max-pct", type=float, default=DEFAULT_MAX_PCT,
                    help="max fraction of each clip covered by B-roll (default 0.40)")
    ap.add_argument("--max-cuts", type=int, default=DEFAULT_MAX_CUTS)
    ap.add_argument("--min-gap", type=float, default=MIN_GAP,
                    help="min seconds of speaker between cutaways (default 2.5)")
    args = ap.parse_args()
    run(args.workdir, args.llm, args.host, args.max_pct, args.max_cuts, args.min_gap)
    print("B-ROLL PLAN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
