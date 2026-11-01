"""
Stage 2.5 of the chopify pipeline - decide WHERE a clip gets B-roll cutaways.

Reads work/transcript.json + work/segments.json and writes work/broll.json:

    [{"start": 134.2, "end": 187.6,
      "cuts": [{"t": 141.3, "dur": 2.2, "query": "stock market"}, ...]}]

All times are ABSOLUTE source-video seconds (same clock as segments.json), so
render_clips.py can re-time them after --tighten removes filler/silence.

Two planners, both free:
  1. heuristic (default) - cuts away ONLY when the speaker says a concrete,
     filmable thing (money, laptop, rocket, city...) from the curated VISUAL list
     below. Names, verbs, contractions and abstract words ("believe", "privacy",
     "Syed") are never searched for - stock sites return random footage for them.
     A sentence with nothing filmable simply gets no cutaway.
  2. --llm MODEL - asks a LOCAL Ollama server for the cutaway list; falls back to
     the heuristic if Ollama is unreachable or returns junk.

Both go through enforce(), which guarantees an edit that feels like an edit and
not a slideshow: the hook (first INTRO seconds) stays on the speaker's face,
cutaways never overlap, there is a minimum gap between them, and B-roll never
covers more than --max-pct of the clip.

    python broll_plan.py work
    python broll_plan.py work --llm qwen2.5:7b --max-pct 0.25
"""
import re
import json
import argparse
import urllib.request
from pathlib import Path

from score_clips import OLLAMA_HOST, load_transcript, sentenceize

INTRO = 3.0          # s at the start of a clip that always stays on the speaker (the hook)
OUTRO = 1.0          # s at the end that always stays on the speaker
MIN_DUR = 1.6        # shortest cutaway (s)
MAX_DUR = 3.0        # longest cutaway (s)
DEFAULT_DUR = 2.2
MIN_GAP = 3.0        # min speaker time between two cutaways (s)
LEAD = 0.25          # cutaway appears this long BEFORE the keyword is spoken (s)
DEFAULT_MAX_PCT = 0.30
DEFAULT_MAX_CUTS = 6

# Things a stock-footage library can actually show. Singular forms; plurals and
# possessives ("coins", "bitcoin's") are matched by _lookup(). Add your own words.
VISUAL = set("""
money cash coin bitcoin crypto cryptocurrency bank dollar euro stock market chart graph wallet
credit invoice receipt profit investment investor gold treasure vault atm tax salary loan
mortgage auction trader trading billionaire millionaire price
office meeting desk laptop computer startup factory warehouse shop store restaurant cafe
customer employee worker boss team handshake contract signature presentation whiteboard
calendar email spreadsheet phone smartphone website app software server database cloud code
coding programmer developer keyboard screen monitor camera microphone podcast youtube video
billboard newspaper magazine book library school classroom student teacher university
hospital doctor lab laboratory scientist robot ai chip processor circuit satellite rocket
spaceship space planet earth moon galaxy telescope wordpress internet network hacker
car truck train airplane plane airport ship boat port container highway road bridge traffic
bike bicycle motorcycle taxi helicopter drone
city skyline skyscraper building house home apartment street village farm field forest
mountain ocean sea beach river lake desert island sunset sunrise rain storm fire snow tree
flower garden waterfall volcano
key lock door window clock watch map globe flag ladder rope chain hammer tool paper document
pen notebook envelope package box gift bag suitcase cart basket bottle coffee tea food pizza
burger bread cake water glass diamond ring jewelry
crowd family baby child kid couple wedding friend hug smile party dance music concert guitar
piano sport football soccer basketball gym running workout yoga swimming cooking driving
typing writing reading shopping travel vacation hotel passport flight
war soldier police court judge lawyer prison protest election vote government
medicine pill surgery vaccine virus
explosion crash bubble balloon tornado earthquake
""".split())

# What to actually type into the stock site when the spoken word is a poor search term.
ALIASES = {
    "market": "stock market", "crypto": "cryptocurrency", "ai": "artificial intelligence",
    "wordpress": "website design", "startup": "startup office", "coding": "programmer code",
    "app": "smartphone app", "trading": "stock trading", "investor": "investor laptop",
    "government": "government building", "price": "price tag", "chip": "computer chip",
    "credit": "credit card", "hacker": "hacker computer", "friend": "friends together",
    "home": "house", "vote": "voting", "shopping": "shopping mall",
}

FILLERS = {"um", "uh", "uhh", "umm", "erm", "er", "hmm", "mmm", "mm", "ah", "eh"}

LLM_SYSTEM = (
    "You are a short-form video editor adding B-roll cutaways to a clip. You receive "
    "a timestamped transcript of ONE clip. Pick the moments where a stock video or "
    "photo would make the edit feel richer. Rules:\n"
    "- Only for a CONCRETE, filmable thing being talked about (an object, place, "
    "machine, crowd, action). Skip abstract ideas, opinions, feelings and anything "
    "that needs the real person or company on screen.\n"
    "- At most {max_cuts} cutaways, each {min_dur}-{max_dur} seconds long.\n"
    "- Never in the first {intro} seconds or last {outro} seconds of the clip.\n"
    "- Cutaways must not overlap and need at least {min_gap} seconds between them.\n"
    "- 'query' is a 1-3 word search phrase made of common English nouns, the way a "
    "stock-footage tag looks (e.g. 'stock market', 'rocket launch', 'money'). NEVER "
    "use names of people or brands.\n"
    "- 't' is the absolute second the cutaway starts (use the transcript timestamps).\n"
    'Respond with JSON only: {{"cuts": [{{"t": <sec>, "dur": <sec>, "query": "<phrase>"}}]}}'
)


# ------------------------------------------------------------------ helpers

def _clean(w):
    """Lower-case a spoken word: drop possessive 's and every non-alphanumeric char,
    so "don't" -> "dont", "he's" -> "he" and "Bitcoin's" -> "bitcoin"."""
    w = re.sub(r"['’]s$", "", w.strip().lower())
    return re.sub(r"[^a-z0-9]", "", w)


def _lookup(tok):
    """The VISUAL entry a spoken token refers to (handles plurals), or None."""
    if tok in VISUAL:
        return tok
    for cand in ((tok[:-3] + "y") if tok.endswith("ies") else None,
                 tok[:-2] if tok.endswith("es") else None,
                 tok[:-1] if tok.endswith("s") else None):
        if cand and cand in VISUAL:
            return cand
    return None


def make_query(sent_words, max_terms=1):
    """Pick up to `max_terms` filmable keywords from one sentence's words.

    One idea per cutaway by default: gluing two unrelated words together
    ("profit website design") makes stock sites return a mix of both, i.e. neither.

    Returns (query, anchor_time, score) or None when the sentence says nothing
    filmable. anchor_time is when the first chosen keyword is spoken; score is the
    number of distinct filmable words (used to rank candidates best-first).
    """
    hits, seen = [], set()
    for i, w in enumerate(sent_words):
        entry = _lookup(_clean(w["word"]))
        # "a couple of friends" means "a few", not a romantic couple
        if entry == "couple" and i + 1 < len(sent_words) \
                and _clean(sent_words[i + 1]["word"]) == "of":
            continue
        if entry and entry not in seen:
            seen.add(entry)
            hits.append((entry, w["start"]))
    if not hits:
        return None
    chosen = hits[:max_terms]
    phrases = [ALIASES.get(e, e) for e, _ in chosen]
    query = " ".join(dict.fromkeys(" ".join(phrases).split()))     # drop repeated words
    return query, chosen[0][1], float(len(hits))


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
    """Heuristic planner: one candidate per sentence that says something filmable,
    sentences with the most filmable words first, never the same footage idea twice."""
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
    seen, unique = set(), []
    for c in cands:
        words = set(c["query"].split())
        if words <= seen:
            continue
        seen |= words
        unique.append(c)
    return enforce(unique, seg_start, seg_end, **limits)


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
        if not cuts:
            print("      (nothing filmable in this clip - it stays on the speaker)", flush=True)

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
                    help="max fraction of each clip covered by B-roll (default 0.30)")
    ap.add_argument("--max-cuts", type=int, default=DEFAULT_MAX_CUTS)
    ap.add_argument("--min-gap", type=float, default=MIN_GAP,
                    help="min seconds of speaker between cutaways (default 3.0)")
    args = ap.parse_args()
    run(args.workdir, args.llm, args.host, args.max_pct, args.max_cuts, args.min_gap)
    print("B-ROLL PLAN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
