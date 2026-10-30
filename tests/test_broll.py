"""Tests for the B-roll pipeline (planner, fetcher, filter-graph builder) plus
regression tests for bugs found while building it."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import broll_fetch      # noqa: E402
import broll_overlay    # noqa: E402
import broll_plan       # noqa: E402
import chopify          # noqa: E402
import render_clips     # noqa: E402
import score_clips      # noqa: E402
import tighten          # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_word(w, start, dur=0.34):
    return {"word": w, "start": round(start, 3), "end": round(start + dur, 3)}


def sentence_words(text, start, step=0.38):
    return [make_word(w, start + i * step) for i, w in enumerate(text.split())]


# ================================================================ regressions

def test_llm_system_defined_before_main_block():
    """--llm silently fell back to the heuristic because LLM_SYSTEM was defined
    AFTER the `if __name__ == "__main__": main()` block (NameError swallowed by
    the Ollama try/except). It must stay above it."""
    text = (REPO / "score_clips.py").read_text(encoding="utf-8")
    assert text.index("LLM_SYSTEM = (") < text.rindex('if __name__ == "__main__":')
    assert "{min_len}" in score_clips.LLM_SYSTEM            # and it still formats
    score_clips.LLM_SYSTEM.format(min_len=20, max_len=75, max_clips=5, min_score=8)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_tighten_works_with_relative_output_dir(tmp_path, monkeypatch):
    """The concat list used cwd-relative paths, which ffmpeg resolves against the
    list file's folder -> `--tighten` failed for any relative --out (incl. ./clips)."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=160x90:rate=15", "-f", "lavfi", "-i", "sine=frequency=300",
                    "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "src.mp4"], check=True)
    out = tighten.apply_cuts("src.mp4", [(0.0, 1.0), (2.0, 3.0)], 0.0,
                             Path("relout") / "tight.mp4")
    assert out.exists() and out.stat().st_size > 0


# ============================================================= broll_plan

def test_make_query_prefers_proper_nouns_and_skips_stopwords():
    words = sentence_words("Yesterday the Rocket exploded right over the Ocean.", 0.0)
    q, anchor, score = broll_plan.make_query(words)
    assert set(q.split()) == {"rocket", "ocean"}
    assert anchor == words[2]["start"]                      # "Rocket" is spoken first
    assert score > 0


def test_make_query_none_for_empty_content():
    words = sentence_words("Um, so this is what they were just doing.", 0.0)
    assert broll_plan.make_query(words) is None


def test_enforce_keeps_hook_clear_and_no_overlap():
    cuts = [{"t": 100.2, "dur": 2.6, "query": "a"},          # inside the 1.5s intro
            {"t": 101.0, "dur": 2.6, "query": "b"},          # overlaps the first after clamp
            {"t": 110.0, "dur": 2.6, "query": "c"},
            {"t": 111.0, "dur": 2.6, "query": "d"}]          # too close to c
    out = broll_plan.enforce(cuts, 100.0, 140.0)
    assert out and all(c["t"] >= 101.5 for c in out)
    for a, b in zip(out, out[1:]):
        assert b["t"] >= a["t"] + a["dur"] + broll_plan.MIN_GAP - 1e-6
    assert [c["query"] for c in out] == ["a", "c"]


def test_enforce_respects_outro_budget_and_max_cuts():
    cuts = [{"t": 100 + 6 * i, "dur": 3.0, "query": f"q{i}"} for i in range(1, 7)]
    out = broll_plan.enforce(cuts, 100.0, 140.0, max_pct=0.2)    # budget 8s -> 2 cuts of 3s
    assert sum(c["dur"] for c in out) <= 8.0 + 1e-6 and len(out) == 2
    late = broll_plan.enforce([{"t": 139.0, "dur": 3.0, "query": "x"}], 100.0, 140.0)
    assert late == []                                        # would run into the last second
    assert len(broll_plan.enforce(cuts, 100.0, 140.0, max_pct=1.0, max_cuts=3)) <= 3


def test_enforce_skips_garbage_entries():
    cuts = [{"dur": 2}, {"t": "x", "query": "a"}, {"t": 110, "query": "  "},
            {"t": 110, "dur": 2.5, "query": "ok"}]
    out = broll_plan.enforce(cuts, 100.0, 140.0)
    assert [c["query"] for c in out] == ["ok"]


def test_plan_heuristic_stays_inside_clip():
    words = []
    for i, s in enumerate(["The Rocket launch lit the whole Ocean.",
                           "Building a Skyline takes patience and concrete.",
                           "Every Mountain starts with one small stone.",
                           "The Factory produced thousands of engines daily."]):
        words += sentence_words(s, 10 + i * 8)
    sentences = score_clips.sentenceize(words)
    cuts = broll_plan.plan_heuristic(sentences, 10.0, 42.0)
    assert cuts
    for c in cuts:
        assert 10.0 + broll_plan.INTRO <= c["t"] and c["t"] + c["dur"] <= 42.0 - broll_plan.OUTRO


def test_plan_llm_falls_back_when_ollama_down(monkeypatch):
    monkeypatch.setattr(broll_plan, "_ollama_cuts", lambda *a, **k: None)
    words = sentence_words("Yesterday the Rocket exploded right over the Ocean today.", 12.0)
    sentences = score_clips.sentenceize(words)
    assert broll_plan.plan_llm(sentences, 10.0, 40.0, "m") == \
        broll_plan.plan_heuristic(sentences, 10.0, 40.0)


def test_plan_llm_uses_and_enforces_llm_cuts(monkeypatch):
    monkeypatch.setattr(broll_plan, "_ollama_cuts", lambda *a, **k: [
        {"t": 10.2, "dur": 9, "query": "too early and long"},   # clamped by the intro rule
        {"t": 20, "dur": 2.5, "query": "rocket launch"}])
    words = sentence_words("Some words here to make a sentence exist.", 12.0)
    cuts = broll_plan.plan_llm(score_clips.sentenceize(words), 10.0, 40.0, "m")
    assert [c["query"] for c in cuts][-1] == "rocket launch"
    assert all(c["dur"] <= broll_plan.MAX_DUR for c in cuts)


def test_plan_run_writes_broll_json(tmp_path):
    words = []
    for i, s in enumerate(["The Rocket launch lit the whole Ocean.",
                           "Building a Skyline takes patience and concrete.",
                           "Every Mountain starts with one small stone."]):
        words += sentence_words(s, i * 8.0)
    (tmp_path / "transcript.json").write_text(json.dumps(
        {"video": "x.mp4", "duration": 30.0, "words": words, "segments": []}))
    (tmp_path / "segments.json").write_text(json.dumps(
        [{"start": 0.0, "end": 30.0, "hook": "h", "overall": 8}]))
    plan = broll_plan.run(tmp_path)
    saved = json.loads((tmp_path / "broll.json").read_text())
    assert saved == plan and plan[0]["start"] == 0.0 and plan[0]["end"] == 30.0
    assert plan[0]["cuts"]


# ============================================================ broll_fetch

def test_available_sources_and_parse():
    env = {"PEXELS_API_KEY": "k"}
    assert broll_fetch.available_sources(["local", "pexels", "pixabay"], None, env) == ["pexels"]
    assert broll_fetch.available_sources(["local"], ".", env) == ["local"]
    assert broll_fetch.available_sources(["pexels"], None, {}) == []
    assert broll_fetch.parse_sources("pexels, local") == ["pexels", "local"]
    with pytest.raises(SystemExit):
        broll_fetch.parse_sources("pexels,giphy")


def test_pick_video_file_prefers_smallest_that_covers_target():
    files = [{"link": "sd", "width": 640, "height": 360, "file_type": "video/mp4"},
             {"link": "hd", "width": 1920, "height": 1080, "file_type": "video/mp4"},
             {"link": "uhd", "width": 3840, "height": 2160, "file_type": "video/mp4"},
             {"link": "webm", "width": 1920, "height": 1080, "file_type": "video/webm"}]
    assert broll_fetch.pick_video_file(files, (1920, 1080))["link"] == "hd"
    assert broll_fetch.pick_video_file(files[:1], (1920, 1080))["link"] == "sd"
    assert broll_fetch.pick_video_file([]) is None


def test_search_pexels_parses_video_and_photo():
    def fake(url, headers=None):
        assert headers == {"Authorization": "KEY"} and "orientation=portrait" in url
        if "/videos/" in url:
            return {"videos": [{"id": 7, "duration": 12, "url": "https://pexels.com/v7",
                                "user": {"name": "Ann"},
                                "video_files": [{"link": "https://cdn/v7.mp4", "width": 1080,
                                                 "height": 1920, "file_type": "video/mp4"}]}]}
        return {"photos": [{"id": 9, "width": 4000, "height": 3000, "url": "https://pexels.com/p9",
                            "photographer": "Bob", "src": {"large2x": "https://cdn/p9.jpg"}}]}
    v = broll_fetch.search_pexels("rocket", "9:16", "KEY", "video", http=fake)[0]
    assert v["kind"] == "video" and v["id"] == "pexels_v7" and v["credit"]["author"] == "Ann"
    p = broll_fetch.search_pexels("rocket", "9:16", "KEY", "image", http=fake)[0]
    assert p["kind"] == "image" and p["url"] == "https://cdn/p9.jpg" and p["credit"]["license"]


def test_search_pixabay_parses_video_and_photo():
    def fake(url, headers=None):
        if "/videos/" in url:
            return {"hits": [{"id": 3, "duration": 9, "pageURL": "https://pixabay.com/v3",
                              "user": "Cy", "videos": {
                                  "large": {"url": "", "width": 0, "height": 0},
                                  "medium": {"url": "https://cdn/m.mp4", "width": 1920, "height": 1080}}}]}
        return {"hits": [{"id": 4, "largeImageURL": "https://cdn/i.jpg", "imageWidth": 3000,
                          "imageHeight": 2000, "pageURL": "https://pixabay.com/p4", "user": "Di"}]}
    v = broll_fetch.search_pixabay("ocean", "16:9", "KEY", "video", http=fake)[0]
    assert v["url"] == "https://cdn/m.mp4" and v["credit"]["source"] == "Pixabay"
    p = broll_fetch.search_pixabay("ocean", "16:9", "KEY", "image", http=fake)[0]
    assert p["kind"] == "image" and p["w"] == 3000


def _make_library(root):
    (root / "ocean").mkdir(parents=True)
    for rel in ["rocket_launch.mp4", "ocean/waves.mp4", "city_skyline.jpg", "notes.txt"]:
        (root / rel).write_bytes(b"x")


def test_local_index_and_search(tmp_path):
    _make_library(tmp_path)
    idx = broll_fetch.index_local(tmp_path)
    assert len(idx) == 3 and all(p.is_absolute() for p, _, _ in idx)
    names = lambda q, k=None: [Path(c["path"]).name for c in broll_fetch.search_local(q, idx, k)]
    assert names("rocket launches") == ["rocket_launch.mp4"]            # plural stems
    assert names("deep ocean") == ["waves.mp4"]                          # folder name matches
    assert names("city skyline", "image") == ["city_skyline.jpg"]
    assert names("city skyline", "video") == []
    assert names("unrelated words") == []                                # no overlap -> no match


def test_fetcher_downloads_caches_and_falls_back(tmp_path):
    calls = {"dl": [], "http": 0}

    def fake_http(url, headers=None):
        calls["http"] += 1
        if "videos" in url:
            return {"videos": []}                                        # no video results
        return {"photos": [{"id": 1, "width": 100, "height": 100, "url": "u",
                            "photographer": "P", "src": {"large": "https://cdn/1.jpg"}}]}

    def fake_dl(url, dest):
        calls["dl"].append(url)
        Path(dest).write_bytes(b"img")

    f = broll_fetch.Fetcher(tmp_path, ["pexels"], "16:9", env={"PEXELS_API_KEY": "K"},
                            http=fake_http, download=fake_dl, sleep=lambda s: None)
    hit = f.find("rocket")
    assert hit["kind"] == "image" and hit["file"] == "broll_cache/pexels_p1.jpg"
    assert (tmp_path / hit["file"]).exists()
    assert f.find("rocket") is None                                      # same asset not reused
    assert calls["dl"] == ["https://cdn/1.jpg"]


def test_fetcher_skips_failed_downloads(tmp_path):
    def fake_http(url, headers=None):
        return {"photos": [
            {"id": 1, "width": 1, "height": 1, "url": "u", "photographer": "A", "src": {"large": "bad"}},
            {"id": 2, "width": 1, "height": 1, "url": "u", "photographer": "B", "src": {"large": "good"}}]}

    def fake_dl(url, dest):
        if url == "bad":
            raise OSError("boom")
        Path(dest).write_bytes(b"x")

    f = broll_fetch.Fetcher(tmp_path, ["pexels"], "16:9", prefer="image",
                            env={"PEXELS_API_KEY": "K"}, http=fake_http, download=fake_dl,
                            sleep=lambda s: None)
    assert f.find("anything")["file"].endswith("pexels_p2.jpg")


def test_fetch_run_drops_unmatched_and_records_credits(tmp_path):
    lib = tmp_path / "lib"
    _make_library(lib)
    work = tmp_path / "work"
    work.mkdir()
    (work / "broll.json").write_text(json.dumps([{"start": 0, "end": 30, "cuts": [
        {"t": 5, "dur": 2.5, "query": "rocket"}, {"t": 15, "dur": 2.5, "query": "dragon"}]}]))
    plan = broll_fetch.run(work, ["local"], "16:9", broll_dir=lib)
    cuts = plan[0]["cuts"]
    assert [c["query"] for c in cuts] == ["rocket"]
    assert cuts[0]["kind"] == "video" and cuts[0]["credit"]["source"] == "local"
    assert Path(cuts[0]["file"]).is_absolute()


# ========================================================== broll_overlay

def _cut(t, d, kind="video", path="/x/a.mp4", **extra):
    return {"t": t, "dur": d, "kind": kind, "path": path, "query": "q", **extra}


def test_retime_without_tighten_shifts_and_clamps():
    out = broll_overlay.retime_cuts(
        [_cut(110.0, 2.5), _cut(150.0, 2.5), _cut(128.5, 3.0)], 100.0, 30.0)
    assert [(c["rt"], c["rd"]) for c in out] == [(10.0, 2.5), (28.5, 1.5)]   # 150 is past the end


def test_retime_drops_too_short_and_overlapping():
    out = broll_overlay.retime_cuts([_cut(109.5, 2.0), _cut(110.5, 2.0)], 100.0, 30.0)
    assert len(out) == 1                                                     # second overlaps first
    assert broll_overlay.retime_cuts([_cut(129.5, 2.0)], 100.0, 30.0) == []  # only 0.5s visible


def test_retime_with_tighten_maps_through_keep_ranges():
    keeps = [(0.0, 10.0), (14.0, 40.0)]            # clip-relative; seconds 10-14 removed
    out = broll_overlay.retime_cuts([_cut(112.0, 2.0), _cut(121.0, 3.0)], 100.0, 36.0, keeps)
    # first cutaway sits at clip-relative 12-14 -> entirely inside the removed range, dropped;
    # second is at 21-24 -> after the 4s cut, so it moves 4s earlier (17-20)
    assert [(c["rt"], c["rd"]) for c in out] == [(17.0, 3.0)]


def test_retime_with_tighten_shortens_straddling_cutaway():
    keeps = [(0.0, 10.0), (12.0, 40.0)]            # 2s removed at 10-12
    out = broll_overlay.retime_cuts([_cut(108.5, 3.0)], 100.0, 38.0, keeps)   # rel 8.5-11.5
    assert [(c["rt"], c["rd"]) for c in out] == [(8.5, 1.5)]                  # only 8.5-10 kept


def test_cover_size_covers_frame_with_even_dims():
    sw, sh = broll_overlay.cover_size(3000, 2000, 1080, 1920)
    assert sw >= 1080 and sh >= 1920 and sw % 2 == 0 and sh % 2 == 0
    assert broll_overlay.cover_size(1920, 1080, 1920, 1080) == (1920, 1080)


def test_build_graph_structure():
    cuts = [{**_cut(4.0, 2.5, path="/b/v.mp4"), "rt": 4.0, "rd": 2.5},
            {**_cut(12.0, 2.0, kind="image", path="/b/i.jpg"), "rt": 12.0, "rd": 2.0}]
    inputs, fc = broll_overlay.build_graph(
        "sendcmd=f=_crop.cmd,crop=w=608:h=1080:x=0:y=0:exact=1,scale=1080:1920", cuts,
        1080, 1920, 30.0, "subtitles=_caption.ass", probe=lambda p: (3000, 2000))
    assert inputs == ["-stream_loop", "-1", "-t", "2.500", "-i", "/b/v.mp4", "-i", "/b/i.jpg"]
    parts = fc.split(";")
    assert parts[0].startswith("[0:v]sendcmd=") and parts[0].endswith("[base]")
    assert fc.count("overlay=") == 2 and "enable='between(t,4.000,6.500)'" in fc
    assert "zoompan=" in fc and "fade=t=in" in fc and "[outv]" in parts[-1]
    # captions are applied AFTER the overlays so they sit on top of the B-roll
    assert parts[-1].index("subtitles=_caption.ass") > 0 and parts[-1].startswith("[v1]")
    # sendcmd steers every filter named `crop`, so B-roll chains must never use one
    assert all("crop" not in p for p in parts[1:])


def test_load_cuts_matches_segment_and_ignores_missing_files(tmp_path):
    (tmp_path / "broll_cache").mkdir()
    (tmp_path / "broll_cache" / "a.mp4").write_bytes(b"x")
    (tmp_path / "broll.json").write_text(json.dumps([
        {"start": 5.0, "end": 35.0, "cuts": [
            {"t": 10, "dur": 2, "query": "a", "file": "broll_cache/a.mp4", "kind": "video"},
            {"t": 20, "dur": 2, "query": "b", "file": "broll_cache/gone.mp4", "kind": "video"}]}]))
    got = broll_overlay.load_cuts(tmp_path, {"start": 5.0, "end": 35.0})
    assert len(got) == 1 and Path(got[0]["path"]).is_absolute()
    assert broll_overlay.load_cuts(tmp_path, {"start": 6.0, "end": 36.0}) == []
    assert broll_overlay.load_cuts(tmp_path / "nowhere", {"start": 5.0, "end": 35.0}) == []


def test_collect_credits_dedupes():
    a = {"source": "Pexels", "author": "Ann", "url": "u1", "license": "L"}
    out = broll_overlay.collect_credits([{"credit": a}, {"credit": dict(a)}, {}, {"credit": {
        "source": "Pixabay", "author": "Bo", "url": "u2", "license": "L"}}])
    assert len(out) == 2


# =============================================== meta + chopify integration

def test_write_meta_includes_credits_and_source(tmp_path):
    seg = {"hook": "big idea", "overall": 8.0, "start": 1.0, "end": 31.0}
    out = tmp_path / "clip.mp4"
    meta = json.loads(render_clips.write_meta(
        seg, out, credits=[{"source": "Pexels"}], source_url="https://youtu.be/x").read_text())
    assert meta["source_video"] == "https://youtu.be/x" and meta["broll_credits"]
    plain = json.loads(render_clips.write_meta(seg, tmp_path / "p.mp4").read_text())
    assert "broll_credits" not in plain and "source_video" not in plain


def test_broll_problems_preflight(tmp_path):
    assert chopify.broll_problems("local,pexels,pixabay", None, {})                 # nothing usable
    assert chopify.broll_problems("local,pexels", None, {"PEXELS_API_KEY": "k"}) == []
    assert chopify.broll_problems("local", str(tmp_path), {}) == []
    assert "Unknown" in chopify.broll_problems("giphy", None, {})[0]


# ======================================================== real ffmpeg render

@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_with_broll_end_to_end(tmp_path):
    """Real ffmpeg: speaker video + one video cutaway + one still-image cutaway."""
    def ff(*args):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)
    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25", "-f", "lavfi", "-i",
       "sine=frequency=440", "-t", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-c:a", "aac", str(tmp_path / "src.mp4"))
    ff("-f", "lavfi", "-i", "smptebars=size=320x240:rate=25", "-t", "3", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", str(tmp_path / "b.mp4"))
    ff("-f", "lavfi", "-i", "testsrc=size=800x600:rate=1", "-frames:v", "1",
       str(tmp_path / "i.jpg"))
    words = sentence_words("Here is the Rocket story about the Ocean tonight friends.", 0.5, 0.5)
    seg = {"start": 0.0, "end": 9.0, "hook": "e2e", "overall": 8.0}
    (tmp_path / "out").mkdir()
    cuts = [_cut(2.0, 2.0, path=str(tmp_path / "b.mp4")),
            _cut(5.5, 2.0, kind="image", path=str(tmp_path / "i.jpg"))]
    out = render_clips.render(seg, words, tmp_path / "src.mp4", 640, 360, tmp_path, "16:9",
                              out_dir=tmp_path / "out", preview=True, broll=cuts)
    dims = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True).stdout.strip()
    assert dims == "854,480"
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True).stdout)
    assert 8.5 < dur < 9.5
