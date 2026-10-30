# Changelog

All notable changes to Chopify will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **B-roll cutaways** (`--broll`): clips look edited, not just captioned. A planner
  (`broll_plan.py`, heuristic or local Ollama via `--llm`) picks 2-3 second
  cutaway moments from the transcript; a fetcher (`broll_fetch.py`) finds footage
  in your own folder (`--broll-dir`) or from Pexels / Pixabay (free API keys,
  standard library only); `broll_overlay.py` builds the ffmpeg filter graph.
  Cutaways fade in/out, stills get a Ken Burns zoom, captions stay on top, the
  first 1.5 s (the hook) always stays on the speaker, and B-roll is capped at
  `--broll-max-pct` (default 40 %) of each clip. Works with 16:9 / 9:16 / 1:1
  and with `--tighten` (cutaways are re-timed onto the shortened clip).
- `.meta.json` now records `source_video` and `broll_credits` (source, author,
  URL, license for every asset used).
- Pre-flight check for `--broll` (needs a key or a `--broll-dir`).

### Fixed
- `--llm` silently never worked from the command line: `LLM_SYSTEM` was defined
  below the `if __name__ == "__main__"` block, so the NameError was swallowed by
  the Ollama fallback and the heuristic scorer always ran.
- `--tighten` failed with any relative output folder (including the default
  `./clips`): the ffmpeg concat list and the final render used cwd-relative paths.

## [1.3.0] - 2026-09-06

### Added
- **Complete-thought clips**: every clip is grown or trimmed to end on a sentence
  with terminal punctuation, so clips never start or end mid-thought.
- **Keyword search** (`--search "pricing"`): bypass virality scoring and build
  clips around transcript sentences containing all your keywords.
- **Review workflow** (`--clips 1,3,5`): scoring prints a numbered candidate
  table; render only the rows you pick.
- **Preview drafts** (`--preview`): render fast 480p cuts (no poster or
  metadata) to check your picks before the full-quality render.
- **Dependency pre-flight check**: before any stage, verifies ffmpeg/ffprobe are
  on PATH, the ffmpeg build can encode H.264 (libx264) and AAC, and required
  Python packages are installed — each problem reported with a fix hint. Bypass
  with `--no-preflight`.

### Changed
- Candidate selection now shares a single window-growth routine
  (`_grow_window`) between virality scoring and keyword search, both enforcing
  the complete-thought rule.

[Unreleased]: https://github.com/mehbul/chopify/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/mehbul/chopify/releases/tag/v1.3.0
