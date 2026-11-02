"""Tests for picture-in-picture detection / zoom (layout_detect + its filter graph)."""
import subprocess
import shutil
import subprocess as sp
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import broll_overlay    # noqa: E402
import layout_detect as ld  # noqa: E402
import render_clips     # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
W, H = 1920, 1080


# ------------------------------------------------------------- synthetic frames

def pip_frame():
    """640x360: dark frame, bright framed box in the middle, a sliver of the next box
    at the left edge and a small logo top-right - the typical podcast 'reaction' layout."""
    g = np.zeros((360, 640), np.uint8)
    g[87:274, 153:487] = 150                      # box interior
    g[87:90, 153:487] = 255                       # top border
    g[271:274, 153:487] = 255                     # bottom border
    g[87:274, 153:156] = 255
    g[87:274, 484:487] = 255
    g[100:260, 0:10] = 170                        # neighbour box sliver
    g[13:43, 567:607] = 255                       # logo
    return g


def normal_frame():
    return np.full((360, 640), 170, np.uint8)


def test_find_box_detects_framed_box_and_ignores_sliver_and_logo():
    x0, y0, x1, y1 = ld.find_box(pip_frame())
    assert (x0, y0, x1, y1) == pytest.approx((153 / 640, 87 / 360, 487 / 640, 274 / 360), abs=0.01)


def test_find_box_even_when_the_speaker_wears_black_inside_the_box():
    g = pip_frame()
    g[190:268, 156:484] = 20                      # dark shirt filling the bottom of the box
    x0, y0, x1, y1 = ld.find_box(g)
    assert y1 == pytest.approx(274 / 360, abs=0.01)   # bottom border still found


@pytest.mark.parametrize("name,frame", [
    ("normal bright shot", normal_frame()),
    ("black bars letterbox", np.vstack([np.zeros((45, 640), np.uint8),
                                        np.full((270, 640), 150, np.uint8),
                                        np.zeros((45, 640), np.uint8)])),
])
def test_find_box_rejects_normal_frames(name, frame):
    assert ld.find_box(frame) is None


def test_find_box_rejects_dark_portrait_and_edge_to_edge_graphics():
    portrait = np.zeros((360, 640), np.uint8)
    portrait[60:300, 230:400] = 140               # a face on a black background: tall, narrow
    assert ld.find_box(portrait) is None
    full = np.zeros((360, 640), np.uint8)
    full[40:320, 0:640] = 150                     # touches both frame edges
    assert ld.find_box(full) is None
    tiny = np.zeros((360, 640), np.uint8)
    tiny[150:210, 280:360] = 200                  # a small bright thing
    assert ld.find_box(tiny) is None


def test_segments_bridge_flicker_and_drop_short_runs():
    flags = [False] * 12 + [True] * 12 + [False] * 2 + [True] * 12 + [False] * 20 + [True] * 3 + [False] * 5
    segs = ld.segments_from_flags(flags)
    assert len(segs) == 1                          # the 2-sample gap is bridged, the 3-sample run dropped
    a, b = segs[0]
    assert a == pytest.approx(11.5 / 12) and b == pytest.approx(37.5 / 12)   # last True sample is #37
    assert ld.segments_from_flags([False] * 30) == []
    assert ld.segments_from_flags([True] * 4) == []     # 0.33 s < 0.5 s minimum


# ----------------------------------------------------------- validation + plan

BOX = (460.0, 260.0, 1460.0, 820.0)


def track(cx, secs=(0, 12), step=0.2):
    n = int((secs[1] - secs[0]) / step)
    return [(round(secs[0] + i * step, 2), cx, False) for i in range(n)]


def test_validate_requires_a_face_inside_the_box():
    segs = [(4.0, 8.0)]
    assert ld.validate_segments(segs, BOX, track(960.0)) == segs
    assert ld.validate_segments(segs, BOX, track(None)) == []            # no face model / no face
    assert ld.validate_segments(segs, BOX, track(100.0)) == []           # face outside the box
    assert ld.validate_segments(segs, BOX, []) == []
    # face only in half of the second segment's samples -> still kept (>= 30 %)
    mixed = [(t, 960.0 if t < 10 else None, False) for t, _, _ in track(0.0)]
    assert ld.validate_segments([(8.0, 12.0)], BOX, mixed) == [(8.0, 12.0)]


def test_face_center_uses_median_of_faces_in_box_during_segments():
    t = [(5.0, 900.0, False), (5.2, 1000.0, False), (5.4, 1100.0, False),
         (1.0, 200.0, False), (6.0, None, False)]
    assert ld.face_center([(4.0, 8.0)], BOX, t) == 1000.0
    assert ld.face_center([(4.0, 8.0)], BOX, track(None)) is None


@pytest.mark.parametrize("tw,th", [(1080, 1080), (1080, 1920), (1920, 1080)])
def test_plan_window_has_target_aspect_and_stays_inside_box(tw, th):
    x, y, w, h = ld.plan_window(BOX, tw, th, face_cx=960.0)
    assert w / h == pytest.approx(tw / th, rel=1e-6)
    assert x >= BOX[0] - 1e-6 and x + w <= BOX[2] + 1e-6
    assert y >= BOX[1] - 1e-6 and y + h <= BOX[3] + 1e-6
    assert h <= (BOX[3] - BOX[1]) * ld.INSET + 1e-6


def test_plan_window_follows_face_but_is_clamped_to_the_box():
    x, _, w, _ = ld.plan_window(BOX, 1080, 1080, face_cx=900.0)
    assert x + w / 2 == pytest.approx(900.0)                             # follows the face
    x, _, w, _ = ld.plan_window(BOX, 1080, 1080, face_cx=700.0)
    assert x + w / 2 == pytest.approx(BOX[0] + w / 2)                    # too close to the edge: pushed in
    x, _, w, _ = ld.plan_window(BOX, 1080, 1080, face_cx=100.0)
    assert x == pytest.approx(BOX[0])                                    # clamped at the box edge
    x, _, w, _ = ld.plan_window(BOX, 1080, 1080)
    assert x + w / 2 == pytest.approx((BOX[0] + BOX[2]) / 2)             # default: box centre


@pytest.mark.parametrize("tw,th", [(1080, 1080), (1080, 1920)])
def test_pip_geometry_maps_window_onto_output(tw, th):
    win = ld.plan_window(BOX, tw, th, face_cx=900.0)
    g = ld.pip_geometry(win, W, H, tw, th)
    s = g["sw"] / W
    assert win[0] * s + g["ox"] == pytest.approx(0, abs=1.01)            # window left -> output left
    assert (win[0] + win[2]) * s + g["ox"] == pytest.approx(tw, abs=1.5)  # window right -> output right
    assert win[1] * s + g["oy"] == pytest.approx(0, abs=1.01)
    assert (win[1] + win[3]) * s + g["oy"] == pytest.approx(th, abs=1.5)
    assert g["ox"] <= 0 and g["oy"] <= 0                                  # always covers the frame


def test_plan_uses_analyzer_and_tracker_and_is_conservative():
    found = {"segments": [(4.0, 8.0)], "rect": BOX}
    p = ld.plan("src", 0, 12, W, H, 1080, 1080, track(960.0), analyzer=lambda *a: found)
    assert p["segments"] == [(4.0, 8.0)] and p["sw"] > W and p["ox"] <= 0
    assert ld.plan("src", 0, 12, W, H, 1080, 1080, track(None), analyzer=lambda *a: found) is None
    assert ld.plan("src", 0, 12, W, H, 1080, 1080, track(960.0), analyzer=lambda *a: None) is None


def test_analyze_reads_frames_from_ffmpeg_and_returns_median_box():
    frames = [normal_frame()] * 12 + [pip_frame()] * 12
    out = b"".join(f.tobytes() for f in frames)

    class Res:
        returncode, stdout = 0, out
    found = ld.analyze("src", 0, 2, W, H, runner=lambda *a, **k: Res())
    assert len(found["segments"]) == 1
    a, b = found["segments"][0]
    assert a == pytest.approx(11.5 / 12) and b == pytest.approx(23.5 / 12)
    assert found["rect"] == pytest.approx((461, 261, 1461, 822), abs=6)

    class Fail:
        returncode, stdout = 1, b""
    assert ld.analyze("src", 0, 2, W, H, runner=lambda *a, **k: Fail()) is None
    assert ld.analyze("src", 0, 2, W, H, runner=lambda *a, **k: type("R", (), {"returncode": 0, "stdout": b""})()) is None


# ------------------------------------------------------------------ the graph

def test_build_graph_with_pip_zooms_only_during_segments_and_keeps_one_crop():
    pip = {"segments": [(4.0, 8.0), (10.0, 11.5)], "sw": 2200, "sh": 1238, "ox": -900, "oy": -480}
    cuts = [{"kind": "video", "path": "/b/v.mp4", "rt": 1.0, "rd": 2.0}]
    inputs, fc = broll_overlay.build_graph(
        "sendcmd=f=_crop.cmd,crop=w=1080:h=1080:x=0:y=0:exact=1,scale=1080:1080", cuts,
        1080, 1080, 24.0, "subtitles=_caption.ass", pip=pip)
    assert fc.startswith("[0:v]split=2[s0][s1];[s0]sendcmd=")
    assert "scale=2200:1238:flags=lanczos" in fc
    assert "overlay=x=-900:y=-480:enable='between(t,4.000,8.000)+between(t,10.000,11.500)'" in fc
    assert fc.count("crop=") == 1                       # sendcmd would steer a second crop filter
    # b-roll still sits on top of the zoom, captions on top of everything
    assert fc.index("[vp]") < fc.index("[b0]") < fc.index("subtitles=")
    assert fc.endswith("[outv]")
    # without pip the graph is unchanged
    _, plain = broll_overlay.build_graph("scale=1080:1080", [], 1080, 1080, 24.0, "subtitles=a.ass")
    assert "split" not in plain and plain == "[0:v]scale=1080:1080[base];[base]subtitles=a.ass,format=yuv420p[outv]"


# ---------------------------------------------------- real ffmpeg, end to end

def _ff(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def _pixel(video, t, x, y):
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    cap.release()
    assert ok
    b, g, r = (int(v) for v in fr[y, x])
    return r, g, b


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_pip_zoom_end_to_end(tmp_path, monkeypatch):
    """12 s source: tan close-up (0-4 s) / dark frame with a framed green box (4-8 s) /
    tan close-up (8-12 s). In 1:1 the box must fill the frame during 4-8 s and the
    normal crop must be untouched elsewhere."""
    tan = "color=c=0xC8A060:s=1920x1080:r=24:d=4"
    box = ("[1:v]drawbox=x=460:y=260:w=1000:h=560:c=0x20B040:t=fill,"
           "drawbox=x=460:y=260:w=1000:h=560:c=white:t=3,"
           "drawbox=x=0:y=300:w=30:h=480:c=0xC8A060:t=fill,"
           "drawbox=x=1700:y=40:w=120:h=90:c=white:t=fill,"
           "drawbox=x=910:y=420:w=100:h=140:c=red:t=fill[p];[0:v][p][2:v]concat=n=3:v=1:a=0[v]")
    src = tmp_path / "src.mp4"
    _ff("-f", "lavfi", "-i", tan, "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=24:d=4",
        "-f", "lavfi", "-i", tan, "-f", "lavfi", "-i", "sine=frequency=300:d=12",
        "-filter_complex", box, "-map", "[v]", "-map", "3:a", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src))

    found = ld.analyze(src, 0, 12, W, H)               # real ffmpeg, real detection
    assert found and len(found["segments"]) == 1
    a, b = found["segments"][0]
    assert 3.7 < a < 4.3 and 7.7 < b < 8.3
    assert found["rect"] == pytest.approx((460, 260, 1460, 820), abs=8)

    # no face model in CI: pretend the tracker found a face in the middle of the frame
    monkeypatch.setattr(render_clips, "detect_track",
                        lambda source, start, dur, Wd, wd: track(960.0, (0, 12)))
    (tmp_path / "out").mkdir()
    seg = {"start": 0.0, "end": 12.0, "hook": "pip", "overall": 8.0}
    out = render_clips.render(seg, [], src, W, H, tmp_path, "1:1", out_dir=tmp_path / "out",
                              preview=True)
    # preview is 480x480; sample inside the frame at three moments
    corner = (6, 6)
    for t, label in ((2.0, "tan"), (10.0, "tan")):
        r, g, bl = _pixel(out, t, *corner)
        assert (r, g, bl) == pytest.approx((200, 160, 96), abs=20), f"{label} at {t}s"
    r, g, bl = _pixel(out, 6.0, *corner)               # zoomed: green box fills the corner
    assert g > 140 and r < 90 and bl < 120
    r, g, bl = _pixel(out, 6.0, 240, 215)              # the red marker near the middle
    assert r > 200 and g < 60
    r, g, bl = _pixel(out, 6.0, 474, 474)              # opposite corner: still the box, no black bars
    assert g > 140 and r < 90
