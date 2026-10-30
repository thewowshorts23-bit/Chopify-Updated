"""
Stage 2.6 of the chopify pipeline - find a stock video / photo for every planned
B-roll cutaway.

Reads work/broll.json (from broll_plan.py), downloads one asset per cutaway into
work/broll_cache/ and rewrites broll.json with the local file, its kind
(video|image) and a credit record (source, author, url, license).

Sources, tried in the order you give them (--sources, default local,pexels,pixabay):
  local    your own footage/images in --broll-dir; matched by filename / folder
           words against the cutaway query. No network, no key.
  pexels   https://www.pexels.com/api/   free key in $PEXELS_API_KEY
  pixabay  https://pixabay.com/api/docs/ free key in $PIXABAY_API_KEY
Both stock sites license their content for commercial use without attribution
(always re-check their current terms). Only the standard library is used, so
there is nothing extra to install.

A cutaway that cannot be matched is DROPPED - no footage beats irrelevant footage.

    set PEXELS_API_KEY=xxxx
    python broll_fetch.py work --aspect 9:16
    python broll_fetch.py work --sources local --broll-dir D:\\my_broll
"""
import os
import re
import json
import time
import hashlib
import argparse
import urllib.parse
import urllib.request
from pathlib import Path

ENV_KEYS = {"pexels": "PEXELS_API_KEY", "pixabay": "PIXABAY_API_KEY"}
ALL_SOURCES = ("local", "pexels", "pixabay")
LICENSES = {"pexels": "Pexels License", "pixabay": "Pixabay Content License",
            "local": "user-provided"}
ASPECT_DIMS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
PEXELS_ORIENT = {"16:9": "landscape", "9:16": "portrait", "1:1": "square"}
PIXABAY_ORIENT = {"16:9": "horizontal", "9:16": "vertical", "1:1": "all"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
UA = "chopify/1.4 (+https://github.com/thewowshorts23-bit/Chopify-Updated)"
MAX_DOWNLOAD = 150 * 1024 * 1024
API_PAUSE = 0.3      # s between API calls - stay well inside free-tier rate limits


# ------------------------------------------------------------------ network
# Both helpers are module-level so tests can monkeypatch them.

def http_json(url, headers=None, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_download(url, dest, timeout=90):
    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        size = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_DOWNLOAD:
                f.close()
                tmp.unlink(missing_ok=True)
                raise ValueError("asset larger than the download cap")
            f.write(chunk)
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------- availability

def available_sources(requested, broll_dir=None, env=None):
    """Subset of `requested` that can actually run (key set / folder exists)."""
    env = os.environ if env is None else env
    out = []
    for s in requested:
        if s == "local" and broll_dir and Path(broll_dir).is_dir():
            out.append(s)
        elif s in ENV_KEYS and env.get(ENV_KEYS[s]):
            out.append(s)
    return out


def parse_sources(spec):
    names = [s.strip().lower() for s in (spec or "").split(",") if s.strip()]
    bad = [s for s in names if s not in ALL_SOURCES]
    if bad:
        raise SystemExit(f"Unknown B-roll source(s) {bad}. Choose from: {', '.join(ALL_SOURCES)}")
    return names or list(ALL_SOURCES)


# ------------------------------------------------------------- remote search

def pick_video_file(files, want=(1080, 1920)):
    """Pick the mp4 whose pixel count is the smallest that still covers `want`
    (avoids multi-GB 4K downloads); if none is big enough, the largest."""
    mp4 = [f for f in files if str(f.get("file_type", "video/mp4")).endswith("mp4")
           and f.get("link") and f.get("width") and f.get("height")]
    if not mp4:
        return None
    need = want[0] * want[1] * 0.8
    big = [f for f in mp4 if f["width"] * f["height"] >= need]
    if big:
        return min(big, key=lambda f: f["width"] * f["height"])
    return max(mp4, key=lambda f: f["width"] * f["height"])


def search_pexels(query, aspect, key, kind, http=None):
    http = http or http_json
    headers = {"Authorization": key}
    orient = PEXELS_ORIENT[aspect]
    q = urllib.parse.urlencode({"query": query, "per_page": 8, "orientation": orient})
    out = []
    if kind == "video":
        data = http(f"https://api.pexels.com/videos/search?{q}", headers)
        for v in data.get("videos", []):
            f = pick_video_file(v.get("video_files", []), ASPECT_DIMS[aspect])
            if f:
                out.append({"kind": "video", "id": f"pexels_v{v['id']}", "url": f["link"],
                            "ext": ".mp4", "duration": v.get("duration"),
                            "credit": {"source": "Pexels", "author": (v.get("user") or {}).get("name"),
                                       "url": v.get("url"), "license": LICENSES["pexels"]}})
    else:
        data = http(f"https://api.pexels.com/v1/search?{q}", headers)
        for p in data.get("photos", []):
            src = p.get("src") or {}
            url = src.get("large2x") or src.get("large") or src.get("original")
            if url:
                out.append({"kind": "image", "id": f"pexels_p{p['id']}", "url": url,
                            "ext": ".jpg", "w": p.get("width"), "h": p.get("height"),
                            "credit": {"source": "Pexels", "author": p.get("photographer"),
                                       "url": p.get("url"), "license": LICENSES["pexels"]}})
    return out


def search_pixabay(query, aspect, key, kind, http=None):
    http = http or http_json
    out = []
    if kind == "video":
        q = urllib.parse.urlencode({"key": key, "q": query, "per_page": 8, "safesearch": "true"})
        data = http(f"https://pixabay.com/api/videos/?{q}")
        for h in data.get("hits", []):
            vids = h.get("videos") or {}
            files = [{"link": v.get("url"), "width": v.get("width"), "height": v.get("height"),
                      "file_type": "video/mp4"} for v in vids.values() if v.get("url")]
            f = pick_video_file(files, ASPECT_DIMS[aspect])
            if f:
                out.append({"kind": "video", "id": f"pixabay_v{h['id']}", "url": f["link"],
                            "ext": ".mp4", "duration": h.get("duration"),
                            "credit": {"source": "Pixabay", "author": h.get("user"),
                                       "url": h.get("pageURL"), "license": LICENSES["pixabay"]}})
    else:
        q = urllib.parse.urlencode({"key": key, "q": query, "per_page": 8, "image_type": "photo",
                                    "orientation": PIXABAY_ORIENT[aspect], "safesearch": "true"})
        data = http(f"https://pixabay.com/api/?{q}")
        for h in data.get("hits", []):
            url = h.get("largeImageURL") or h.get("webformatURL")
            if url:
                out.append({"kind": "image", "id": f"pixabay_p{h['id']}", "url": url,
                            "ext": ".jpg", "w": h.get("imageWidth"), "h": h.get("imageHeight"),
                            "credit": {"source": "Pixabay", "author": h.get("user"),
                                       "url": h.get("pageURL"), "license": LICENSES["pixabay"]}})
    return out


# -------------------------------------------------------------- local library

def _tokens(text):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks if len(t) > 2}


def index_local(broll_dir):
    """[(path, kind, token_set)] for every video/image under broll_dir. Tokens come
    from the file name and its folder names, so 'city/skyline_night.mp4' matches
    'city skyline'."""
    root = Path(broll_dir).resolve()     # absolute, so render can find it from any cwd
    items = []
    for p in sorted(root.rglob("*")):
        ext = p.suffix.lower()
        if ext in VIDEO_EXT or ext in IMAGE_EXT:
            rel = p.relative_to(root)
            toks = _tokens(" ".join([p.stem] + [x for x in rel.parts[:-1]]))
            items.append((p, "video" if ext in VIDEO_EXT else "image", toks))
    return items


def search_local(query, index, kind=None):
    """Local files ranked by query-word overlap (>=1 shared word required)."""
    q = _tokens(query)
    ranked = []
    for path, k, toks in index:
        if kind and k != kind:
            continue
        overlap = len(q & toks)
        if overlap:
            ranked.append((overlap, path, k))
    ranked.sort(key=lambda x: (-x[0], str(x[1])))
    return [{"kind": k, "id": f"local_{hashlib.md5(str(p).encode()).hexdigest()[:10]}",
             "path": str(p), "credit": {"source": "local", "author": None,
                                        "url": None, "license": LICENSES["local"]}}
            for _, p, k in ranked]


# ------------------------------------------------------------------- fetching

def query_variants(query):
    """Full phrase first, then single words (longest first) as fallbacks."""
    terms = query.split()
    out = [query]
    if len(terms) > 1:
        out += sorted(terms, key=len, reverse=True)
    return out


class Fetcher:
    def __init__(self, workdir, sources, aspect, broll_dir=None, prefer="video",
                 env=None, http=None, download=None, sleep=time.sleep):
        self.workdir = Path(workdir)
        self.cache = self.workdir / "broll_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sources = sources
        self.aspect = aspect
        self.kinds = ["video", "image"] if prefer == "video" else ["image", "video"]
        env = os.environ if env is None else env
        self.keys = {s: env.get(k) for s, k in ENV_KEYS.items()}
        self.http = http or http_json
        self.download = download or http_download
        self.sleep = sleep
        self.local = index_local(broll_dir) if "local" in sources and broll_dir else []
        self.used = set()
        self._search_cache = {}

    def _search(self, source, query, kind):
        ck = (source, query, kind, self.aspect)
        if ck in self._search_cache:
            return self._search_cache[ck]
        try:
            if source == "local":
                res = search_local(query, self.local, kind)
            else:
                fn = search_pexels if source == "pexels" else search_pixabay
                res = fn(query, self.aspect, self.keys[source], kind, http=self.http)
                self.sleep(API_PAUSE)
        except Exception as e:                               # noqa: BLE001
            print(f"    {source} search failed for '{query}' ({kind}): {e}", flush=True)
            res = []
        self._search_cache[ck] = res
        return res

    def _materialize(self, cand):
        """Make `cand` a local file under the cache; returns the stored path string."""
        if "path" in cand:                                   # local library file
            return cand["path"]
        dest = self.cache / f"{cand['id']}{cand['ext']}"
        if not dest.exists():
            self.download(cand["url"], dest)
        return os.path.relpath(dest, self.workdir).replace("\\", "/")

    def find(self, query, need_dur=0.0):
        """Best unused asset for `query` across sources/kinds, or None."""
        for variant in query_variants(query):
            for source in self.sources:
                for kind in self.kinds:
                    for cand in self._search(source, variant, kind):
                        if cand["id"] in self.used:
                            continue
                        dur = cand.get("duration")
                        if kind == "video" and dur and dur < need_dur:
                            continue
                        try:
                            f = self._materialize(cand)
                        except Exception as e:               # noqa: BLE001
                            print(f"    download failed ({cand['id']}): {e}", flush=True)
                            continue
                        self.used.add(cand["id"])
                        return {"file": f, "kind": cand["kind"], "credit": cand["credit"],
                                "matched": variant}
        return None


def run(workdir, sources, aspect="16:9", broll_dir=None, prefer="video", fetcher=None):
    """Attach an asset to every cutaway in workdir/broll.json (drops unmatched ones)."""
    workdir = Path(workdir)
    path = workdir / "broll.json"
    if not path.exists():
        raise SystemExit(f"No B-roll plan at {path} - run broll_plan.py first")
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    fetcher = fetcher or Fetcher(workdir, sources, aspect, broll_dir, prefer)

    kept_total = dropped_total = 0
    for i, seg in enumerate(plan, 1):
        kept = []
        for c in seg.get("cuts", []):
            if c.get("file"):                                # already fetched on a prior run
                kept.append(c)
                continue
            hit = fetcher.find(c["query"], need_dur=c.get("dur", 0.0))
            if hit:
                kept.append({**c, **hit})
                print(f"  clip {i}: '{c['query']}' -> {hit['kind']} {hit['file']}"
                      f"  ({hit['credit']['source']})", flush=True)
            else:
                dropped_total += 1
                print(f"  clip {i}: '{c['query']}' -> no match, cutaway dropped", flush=True)
        seg["cuts"] = kept
        kept_total += len(kept)

    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fetched {kept_total} asset(s), dropped {dropped_total} unmatched cutaway(s)", flush=True)
    return plan


def main():
    ap = argparse.ArgumentParser(description="Fetch B-roll assets for chopify cutaways.")
    ap.add_argument("workdir", nargs="?", default="work")
    ap.add_argument("--sources", default="local,pexels,pixabay",
                    help="comma list in priority order: local,pexels,pixabay")
    ap.add_argument("--broll-dir", default=None, help="folder of your own footage/images")
    ap.add_argument("--aspect", default="16:9", choices=list(ASPECT_DIMS))
    ap.add_argument("--prefer", default="video", choices=["video", "image"],
                    help="try stock video first (default) or photos first")
    args = ap.parse_args()

    requested = parse_sources(args.sources)
    usable = available_sources(requested, args.broll_dir)
    if not usable:
        raise SystemExit(
            "No usable B-roll source. Do one of:\n"
            "  - set PEXELS_API_KEY and/or PIXABAY_API_KEY (both are free), or\n"
            "  - pass --broll-dir <folder of your own footage/images> with --sources local")
    print(f"B-roll sources in use: {', '.join(usable)}", flush=True)
    run(args.workdir, usable, args.aspect, args.broll_dir, args.prefer)
    print("B-ROLL FETCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
