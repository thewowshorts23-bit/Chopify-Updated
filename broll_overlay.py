"""
broll_overlay.py - turn planned B-roll cutaways into an ffmpeg filter graph.

Pure functions (no ffmpeg is run here) used by render_clips.py:

  load_cuts()      cutaways for one segment from work/broll.json (assets on disk)
  retime_cuts()    absolute source seconds -> clip-relative seconds, honouring --tighten
  build_graph()    -filter_complex string + extra ffmpeg input args
  collect_credits() unique credit records for the clip's .meta.json

Layout of the graph:   speaker (crop/track/scale)  ->  cutaway overlays  ->  captions
so the burnt-in subtitles always stay on top of the B-roll and the speaker's audio
keeps playing underneath every cutaway ("full" cutaway style).

Design notes
  * B-roll chains deliberately avoid the `crop` filter: smooth_crop's sendcmd file
    targets the filter *class* name `crop`, so any extra crop in the graph would be
    steered by the speaker tracker. Instead B-roll is scaled to COVER the frame and
    the overlay filter (which clips to the main frame) trims the overflow.
  * Still images get a slow Ken Burns zoom (alternating push-in / pull-out).
  * Every cutaway fades in/out (FADE s) so it dissolves rather than hard-cuts.
"""
import json
import math
from pathlib import Path

import tighten

FADE = 0.15          # fade in/out length of each cutaway (s)
ZOOM = 0.12          # total Ken Burns zoom over a still image (12 %)
MIN_VISIBLE = 0.8    # a cutaway shorter than this after re-timing is dropped (s)


def load_cuts(workdir, seg):
    """Cutaways for `seg` from workdir/broll.json whose asset file exists.

    Matches the plan entry by the segment's start/end. Returns [] when there is
    no plan, no matching entry, or no usable asset. Each returned cut gets an
    absolute `path`.
    """
    workdir = Path(workdir)
    plan_path = workdir / "broll.json"
    if not plan_path.exists():
        return []
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    s, e = float(seg["start"]), float(seg["end"])
    for entry in plan:
        if abs(float(entry["start"]) - s) < 0.05 and abs(float(entry["end"]) - e) < 0.05:
            out = []
            for c in entry.get("cuts", []):
                f = c.get("file")
                if not f:
                    continue
                p = Path(f)
                if not p.is_absolute():
                    p = workdir / p
                if p.exists():
                    out.append({**c, "path": str(p.resolve())})
            return out
    return []


def retime_cuts(cuts, clip_start, dur, keeps=None):
    """Map absolute-time cutaways onto the rendered clip's timeline.

    clip_start: absolute start of the clip in the source video.
    dur:        duration of the rendered clip (after --tighten, if used).
    keeps:      tighten.plan_cuts() keep-ranges, or None when not tightening. Cutaways
                that fall in removed material are dropped; ones that straddle a cut
                are shortened.
    Adds `rt` (start) and `rd` (duration), clip-relative. Overlaps are removed.
    """
    out = []
    for c in cuts:
        a = float(c["t"])
        b = a + float(c["dur"])
        if keeps is None:
            ra, rb = a - clip_start, b - clip_start
        else:
            mapped, _ = tighten.remap_words(
                [{"word": "", "start": a, "end": b}], clip_start, keeps)
            if not mapped:
                continue
            ra, rb = mapped[0]["start"], mapped[0]["end"]
        ra, rb = max(0.0, ra), min(float(dur), rb)
        if rb - ra < MIN_VISIBLE:
            continue
        out.append({**c, "rt": round(ra, 3), "rd": round(rb - ra, 3)})
    out.sort(key=lambda c: c["rt"])
    clean = []
    for c in out:
        if clean and c["rt"] < clean[-1]["rt"] + clean[-1]["rd"]:
            continue
        clean.append(c)
    return clean


def cover_size(w, h, tw, th):
    """Smallest even-sized scale of a w x h image that covers a tw x th frame."""
    s = max(tw / float(w), th / float(h))
    sw, sh = int(math.ceil(w * s)), int(math.ceil(h * s))
    return sw + sw % 2, sh + sh % 2


def build_graph(base_chain, cuts, tw, th, fps, tail_chain, probe=None, fade=FADE):
    """Build the -filter_complex graph.

    base_chain: filters turning input 0 into the framed speaker video, e.g.
                "sendcmd=f=_crop.cmd,crop=...,scale=1080:1920".
    cuts:       output of retime_cuts() (needs rt, rd, kind, path).
    tail_chain: filters applied after the overlays, e.g. "subtitles=_caption.ass".
    probe:      callable(path) -> (w, h) used for still images (falls back to cut w/h).

    Returns (extra_input_args, filter_complex). Map the result with
    ["-map", "[outv]", "-map", "0:a?"] - B-roll audio is never used.
    """
    inputs = []
    parts = [f"[0:v]{base_chain}[base]"]
    last = "base"
    for k, c in enumerate(cuts):
        n = k + 1
        t0, d = float(c["rt"]), float(c["rd"])
        f = min(fade, d / 4.0)
        if c.get("kind") == "image":
            try:
                w, h = probe(c["path"]) if probe else (c["w"], c["h"])
            except Exception:                                # noqa: BLE001
                w, h = c.get("w") or tw, c.get("h") or th
            sw, sh = cover_size(w, h, tw, th)
            frames = max(2, int(round(d * fps)))
            rate = ZOOM / frames
            z = f"1+{rate:.6f}*on" if k % 2 == 0 else f"{1 + ZOOM:.4f}-{rate:.6f}*on"
            inputs += ["-i", c["path"]]
            pre = (f"scale={sw * 2}:{sh * 2},"
                   f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                   f":d={frames}:s={sw}x{sh}:fps={fps:.3f}")
        else:
            inputs += ["-stream_loop", "-1", "-t", f"{d:.3f}", "-i", c["path"]]
            pre = (f"scale={tw}:{th}:force_original_aspect_ratio=increase"
                   f":force_divisible_by=2,fps={fps:.3f}")
        parts.append(
            f"[{n}:v]{pre},setsar=1,format=yuva420p,"
            f"fade=t=in:st=0:d={f:.3f}:alpha=1,"
            f"fade=t=out:st={d - f:.3f}:d={f:.3f}:alpha=1,"
            f"setpts=PTS-STARTPTS+{t0:.3f}/TB[b{k}]")
        parts.append(
            f"[{last}][b{k}]overlay=x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2"
            f":enable='between(t,{t0:.3f},{t0 + d:.3f})'[v{k}]")
        last = f"v{k}"
    parts.append(f"[{last}]{tail_chain},format=yuv420p[outv]")
    return inputs, ";".join(parts)


def collect_credits(cuts):
    """Unique {source, author, url, license} records for the cutaways used."""
    seen, out = set(), []
    for c in cuts:
        cr = c.get("credit") or {}
        key = (cr.get("source"), cr.get("url"), cr.get("author"))
        if cr and key not in seen:
            seen.add(key)
            out.append(cr)
    return out
