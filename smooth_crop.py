"""
smooth_crop.py - stutter-free speaker tracking for chopify's 9:16 / 1:1 crops.

Why the old version stuttered
-----------------------------
The old code wrote ONE `sendcmd` crop-x update per face detection (every 0.1 s
at DET_FPS=10) and the video is ~30 fps. So the crop position only changed on
every 3rd frame and stood still for the two frames in between: the "camera"
panned at 10 fps inside a 30 fps video. That staircase is the judder you see.
It is not dropped frames (the output timestamps are perfectly regular).

What this module does instead
-----------------------------
1. Cut detection   - a cheap frame-diff on the low-res detection video marks
                     scene cuts, so the crop can snap exactly on a cut (the
                     content jumps anyway, so a snap is invisible).
2. Dead-zone       - inside a shot the camera holds still until the face
                     drifts out of a small zone, like a real camera operator.
3. Gaussian glide  - the path is resampled to the OUTPUT frame rate and
                     low-pass filtered (zero-phase, so no lag), giving eased
                     pans with no jumps.
4. Per-frame cmds  - one sendcmd line per output frame, so the crop moves
                     every frame instead of every 3rd.
"""
from pathlib import Path

import numpy as np

# ---- tuning knobs -----------------------------------------------------------
DEADZONE_FRAC = 0.10   # of crop width; face may wander this far before the camera moves
SMOOTH_SEC = 0.45      # gaussian sigma in seconds: bigger = lazier, smoother pans
CUT_FRAC = 0.22        # face jump (fraction of source width) between detections = cut
CUT_DIFF = 40.0        # mean abs diff (0-255) on 48x27 gray frames = scene cut
MARGIN_FRAC = 0.18     # face is never allowed closer than this to the crop edge
# -----------------------------------------------------------------------------


def _gauss(x, sigma):
    """Zero-phase gaussian smoothing with edge padding."""
    if sigma < 0.5 or len(x) < 3:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def _deadzone(face, dz):
    """Camera stays put while the face is within +-dz, otherwise it is dragged."""
    cam = face[0]
    out = np.empty_like(face)
    for i, f in enumerate(face):
        if f - cam > dz:
            cam = f - dz
        elif cam - f > dz:
            cam = f + dz
        out[i] = cam
    return out


def smooth_track(track, W, crop_w, fps, det_fps):
    """
    track  : [(t_rel, face_center_x or None, is_cut)] from detect_track()
    returns: [(t_rel, crop_center_x)] with one entry PER OUTPUT FRAME.
    """
    half = crop_w / 2.0
    lo, hi = half, W - half
    max_off = half - crop_w * MARGIN_FRAC
    if not track:
        return [(0.0, W / 2.0)]

    times = np.array([t for t, _, _ in track], dtype=float)
    raw = [c for _, c, _ in track]
    cuts = [bool(k) for _, _, k in track]

    known = sorted(c for c in raw if c is not None)
    base = known[len(known) // 2] if known else W / 2.0

    # fill missing detections (hold last, leading gaps use the clip median)
    face = np.empty(len(raw))
    last = base
    for i, c in enumerate(raw):
        last = c if c is not None else last
        face[i] = last

    # 3-tap median: removes single-sample detector glitches but keeps real steps
    if len(face) >= 3:
        pad = np.pad(face, 1, mode="edge")
        face = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)

    # a big face jump between two detections is also a cut
    for i in range(1, len(face)):
        if abs(face[i] - face[i - 1]) > W * CUT_FRAC:
            cuts[i] = True

    step = 1.0 / det_fps
    t_end = times[-1] + step
    T = np.arange(0.0, t_end, 1.0 / fps)

    # shot boundaries: a cut seen at detection i happened in (t[i-1], t[i]]
    bounds = [0.0] + [times[i] - step / 2.0 for i in range(1, len(times)) if cuts[i]] + [t_end + 1]
    cam = np.empty_like(T)
    dz = crop_w * DEADZONE_FRAC
    sigma = SMOOTH_SEC * fps
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = np.where((T >= a) & (T < b))[0]
        if not len(sel):
            continue
        # detections that belong to this shot
        in_shot = [i for i, t in enumerate(times) if (t >= a - step / 2.0 and t < b + step / 2.0)]
        if not in_shot:
            in_shot = [int(np.argmin(np.abs(times - a)))]
        dense = np.interp(T[sel], times[in_shot], face[in_shot])
        path = _gauss(_deadzone(dense, dz), sigma)
        # safety net: never let the face get closer than MARGIN_FRAC to the edge
        path = np.clip(path, dense - max_off, dense + max_off)
        cam[sel] = path

    cam = np.clip(cam, lo, hi)
    return [(float(t), float(c)) for t, c in zip(T, cam)]


def build_sendcmd(track, crop_w, path, fps):
    """
    One `crop x` command per output frame. Each command is scheduled half a
    frame early so it reliably lands on the intended frame.
    """
    lines = []
    for n, (_, cx) in enumerate(track):
        x = int(round(cx - crop_w / 2.0))
        t = max(0.0, (n - 0.5) / fps)
        lines.append(f"{t:.4f} crop x {x};")
    if not lines:
        lines = ["0.0 crop x 0;"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
