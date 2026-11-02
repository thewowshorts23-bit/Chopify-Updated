"""
layout_detect.py - find "picture-in-picture" moments in a source video and plan a
zoom into the speaker's box.

Many podcasts switch between a full-frame close-up and a dark frame that holds a
small framed video box (the "reaction" layout). A plain centre/face-tracked crop to
9:16 or 1:1 cuts that box in half and shows black bars. This module:

  1. samples the clip at SAMPLE_FPS as small grayscale frames,
  2. finds the bright box on a mostly dark frame (find_box),
  3. turns per-frame results into time segments (segments_from_flags),
  4. keeps only segments where the face tracker really found a face INSIDE the box
     (validate_segments) - so dark scenes, letterboxing and full-screen graphics
     never trigger a zoom,
  5. plans a crop window of the target aspect inside the box (plan_window /
     pip_geometry) that render_clips overlays on those segments.

Pure functions plus one ffmpeg call (analyze). Needs numpy (installed with
opencv-python).
"""
import subprocess

import numpy as np

SAMPLE_FPS = 12
GRAY_W, GRAY_H = 640, 360
DARK = 50              # gray level below which a pixel counts as "dark" (black = 16 in tv range)
MIN_SEGMENT = 0.5      # s - shorter picture-in-picture runs are ignored
MAX_GAP = 0.3          # s - detection flicker shorter than this is bridged
INSET = 0.94           # use 94 % of the box so its border / name tag stay out of frame
FACE_RATIO = 0.3       # a segment needs a face inside the box in >= 30 % of tracker samples


def _runs(on, max_gap=0):
    """[(start, end)] inclusive index runs of True in `on`, merging gaps <= max_gap."""
    idx = np.nonzero(on)[0]
    if idx.size == 0:
        return []
    runs = []
    s = p = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - p - 1 > max_gap:
            runs.append((s, p))
            s = i
        p = i
    runs.append((s, p))
    return runs


def find_box(gray):
    """Bright rectangle on a mostly dark frame, or None.

    gray: HxW uint8 array. Returns (x0, y0, x1, y1) as fractions of the frame, only
    when the frame looks like "dark background + one framed video box":
    >= 40 % dark, box 30-92 % of the frame in each direction, landscape-ish aspect,
    clear of the frame edges, and the area outside the box is >= 80 % dark.
    """
    h, w = gray.shape
    nd = gray >= DARK
    if 1.0 - nd.mean() < 0.40:
        return None
    band = nd[int(h * 0.2):int(h * 0.8)]
    runs = _runs(band.mean(axis=0) >= 0.30, max_gap=max(1, int(w * 0.015)))
    if not runs:
        return None
    x0, x1 = max(runs, key=lambda r: r[1] - r[0])      # widest run; ignores edge slivers
    rows = nd[:, x0:x1 + 1].mean(axis=1)
    strong = np.nonzero(rows >= 0.75)[0]                # the box's own top/bottom border rows
    if strong.size < 2:
        return None
    y0, y1 = int(strong[0]), int(strong[-1])

    bw, bh = (x1 - x0 + 1) / w, (y1 - y0 + 1) / h
    if not (0.30 <= bw <= 0.92 and 0.30 <= bh <= 0.92):
        return None
    if not 1.2 <= (bw * w) / (bh * h) <= 2.4:
        return None
    if min(x0, w - 1 - x1) < 0.04 * w or min(y0, h - 1 - y1) < 0.04 * h:
        return None
    box_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    out_area = h * w - box_area
    out_bright = int(nd.sum()) - int(nd[y0:y1 + 1, x0:x1 + 1].sum())
    if out_area <= 0 or out_bright / out_area > 0.20:
        return None
    return x0 / w, y0 / h, (x1 + 1) / w, (y1 + 1) / h


def segments_from_flags(flags, fps=SAMPLE_FPS, min_len=MIN_SEGMENT, max_gap=MAX_GAP):
    """Per-sample booleans -> [(t0, t1)] seconds; bridges short gaps, drops short runs."""
    runs = _runs(np.array(flags, dtype=bool), max_gap=int(round(max_gap * fps)))
    return [(max(0.0, (s - 0.5) / fps), (e + 0.5) / fps)
            for s, e in runs if (e - s + 1) / fps >= min_len]


def analyze(source, start, dur, W, H, runner=subprocess.run):
    """Scan [start, start+dur) of `source`. Returns None or
    {"segments": [(t0, t1)] clip-relative seconds, "rect": (x0, y0, x1, y1) source px}."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(source),
           "-t", f"{dur:.3f}", "-vf",
           f"fps={SAMPLE_FPS},scale={GRAY_W}:{GRAY_H}:flags=area,format=gray",
           "-f", "rawvideo", "-"]
    res = runner(cmd, capture_output=True)
    if res.returncode != 0:
        return None
    size = GRAY_W * GRAY_H
    n = len(res.stdout) // size
    if n == 0:
        return None
    frames = np.frombuffer(res.stdout, np.uint8, n * size).reshape(n, GRAY_H, GRAY_W)
    rects = [find_box(f) for f in frames]
    segs = segments_from_flags([r is not None for r in rects])
    if not segs:
        return None
    inside = [r for i, r in enumerate(rects)
              if r and any(a <= i / SAMPLE_FPS <= b for a, b in segs)]
    med = np.median(np.array(inside), axis=0)
    return {"segments": segs, "rect": (med[0] * W, med[1] * H, med[2] * W, med[3] * H)}


def validate_segments(segments, rect_px, raw_track, min_ratio=FACE_RATIO):
    """Keep segments in which the face tracker found a face inside the box.

    raw_track: [(t, face_cx_or_None, is_cut)] from render_clips.detect_track. If the
    tracker found nothing (no face model) every segment is rejected - the safe default.
    """
    x0, _, x1, _ = rect_px
    kept = []
    for a, b in segments:
        pts = [cx for t, cx, _ in raw_track if a <= t <= b]
        if pts and sum(1 for cx in pts if cx is not None and x0 <= cx <= x1) / len(pts) >= min_ratio:
            kept.append((a, b))
    return kept


def face_center(segments, rect_px, raw_track):
    """Median face x inside the box during `segments`, or None."""
    x0, _, x1, _ = rect_px
    xs = [cx for t, cx, _ in raw_track
          if cx is not None and x0 <= cx <= x1 and any(a <= t <= b for a, b in segments)]
    return float(np.median(xs)) if xs else None


def plan_window(rect_px, tw, th, face_cx=None, inset=INSET):
    """Crop window (x, y, w, h) in source px: target aspect, inside the box, centred on
    the face (or the box centre), as large as the box allows."""
    x0, y0, x1, y1 = rect_px
    rw, rh = x1 - x0, y1 - y0
    ar = tw / float(th)
    wh = min(rh * inset, rw * inset / ar)
    ww = wh * ar
    cx = face_cx if face_cx is not None else (x0 + x1) / 2.0
    cx = min(max(cx, x0 + ww / 2.0), x1 - ww / 2.0)
    cy = (y0 + y1) / 2.0
    return cx - ww / 2.0, cy - wh / 2.0, ww, wh


def pip_geometry(win, W, H, tw, th):
    """Scale + offset that put crop window `win` exactly on the tw x th output when the
    whole source frame is scaled and overlaid at (ox, oy). No `crop` filter is used on
    purpose: smooth_crop's sendcmd steers every filter named `crop`."""
    wx, wy, ww, _ = win
    s = tw / ww
    return {"sw": int(round(W * s)), "sh": int(round(H * s)),
            "ox": -int(round(wx * s)), "oy": -int(round(wy * s))}


def plan(source, start, dur, W, H, tw, th, raw_track, analyzer=analyze):
    """Everything render needs, or None when there is no (validated) picture-in-picture.

    Returns {"segments": [(t0, t1)], "sw", "sh", "ox", "oy"}.
    """
    found = analyzer(source, start, dur, W, H)
    if not found:
        return None
    segs = validate_segments(found["segments"], found["rect"], raw_track)
    if not segs:
        return None
    win = plan_window(found["rect"], tw, th, face_center(segs, found["rect"], raw_track))
    geo = pip_geometry(win, W, H, tw, th)
    print("  layout: picture-in-picture at " +
          ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in segs) +
          " -> zoomed to the speaker's box", flush=True)
    return {"segments": segs, **geo}
