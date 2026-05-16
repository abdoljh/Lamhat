#!/usr/bin/env python3
"""
phase3_run.py — Standalone Phase 3 video generator (v2).

Drives the existing phase3/ package from the command line.  v2 adds the
new structural foundation — word-level forced alignment and AI shot
planning — as inspection modes alongside the existing v1 render path.

The v1 render still works exactly as before; the v2 modes only generate
plans for inspection.  Rendering with the v2 plan is the next session's
work.

Usage
-----
    python phase3_run.py --script path/to/script.txt [options]

Required
--------
    --script PATH       Arabic video script (.txt, UTF-8)

Audio (optional — estimated from character count when omitted)
------
    --audio PATH        MP3 file from Phase 2 TTS
    --audio-duration S  Override total duration in seconds

Content context
---------------
    --book-title TEXT       Book title (improves keyword quality)
    --character-name TEXT   Main character / subject name
    --genre TEXT            history | biography | non-fiction | philosophy |
                            science | religion | novel  [default: history]

API keys (CLI > environment variable > .env file in current directory)
--------
    --anthropic-key KEY     ANTHROPIC_API_KEY  — required for v2 planner +
                                                 v1 keywords + vision scoring
    --pexels-key KEY        PEXELS_API_KEY     — optional Pexels video fallback

Visual options
--------------
    --output PATH           Output .mp4  [default: output/phase3_video.mp4]
    --color-grade NAME      warm | cool | neutral  [default: warm]
    --width N               [default: 1280]
    --height N              [default: 720]
    --images-per-section N  Wikimedia images per section  [default: 3]
    --no-subtitles          Skip ASS subtitle generation

Modes
-----
    --dry-run               Parse sections + estimate durations, print plan, exit
    --keywords-only         v1 mode: generate + print keywords, exit
    --align-only            v2 mode: run forced alignment, print word timings,
                            and write JSON to disk; do not call planner
    --plan-only             v2 mode: run alignment + AI shot planner, save plan
                            JSON, print summary, exit (NO video render)
    --verbose               Show DEBUG-level log output

v2 alignment options
--------------------
    --align-backend NAME    auto | whisperx | whisper | interpolated
                            [default: auto — tries each in order]

Output extras
-------------
    --save-keywords PATH    Write keyword JSON (used by --keywords-only)
    --save-plan PATH        Write shot-plan JSON (used by --plan-only)
    --save-alignment PATH   Write word-timings JSON (used by --align-only)
    --thumbnail             Save a thumbnail JPEG beside the output video
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path


# ── Locate the package root so `phase3` is importable ───────────────────── #

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ── .env loader (no dependency on python-dotenv) ─────────────────────────── #

def _load_dotenv(path: Path) -> None:
    """Parse a simple KEY=VALUE .env file and set missing env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ── Progress printer ─────────────────────────────────────────────────────── #

_LAST_PCT: dict[str, float] = {"v": -1.0}


def _make_progress(verbose: bool):
    def _on_progress(label: str, frac: float) -> None:
        pct = int(frac * 100)
        if pct != _LAST_PCT["v"] or verbose:
            _LAST_PCT["v"] = pct
            bar_len = 30
            filled = int(bar_len * frac)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {pct:3d}%  {label:<55}",
                  end="", flush=True)
        if frac >= 1.0:
            print()
    return _on_progress


# ── CLI ──────────────────────────────────────────────────────────────────── #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phase3_run.py",
        description="Standalone Phase 3 video generator (v2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--script", required=True, metavar="PATH")

    # Audio
    p.add_argument("--audio", metavar="PATH")
    p.add_argument("--audio-duration", type=float, metavar="S")

    # Content context
    p.add_argument("--book-title", default="", metavar="TEXT")
    p.add_argument("--character-name", default="", metavar="TEXT")
    p.add_argument("--genre", default="history",
                   choices=["history", "biography", "non-fiction",
                            "philosophy", "science", "religion", "novel"])

    # API keys
    p.add_argument("--anthropic-key", default="", metavar="KEY")
    p.add_argument("--pexels-key", default="", metavar="KEY")

    # Visual options
    p.add_argument("--output", default="output/phase3_video.mp4", metavar="PATH")
    p.add_argument("--color-grade", default="warm",
                   choices=["warm", "cool", "neutral"])
    p.add_argument("--width", type=int, default=1280, metavar="N")
    p.add_argument("--height", type=int, default=720, metavar="N")
    p.add_argument("--images-per-section", type=int, default=3, metavar="N")
    p.add_argument("--no-subtitles", action="store_true")

    # Modes
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keywords-only", action="store_true")
    p.add_argument("--align-only", action="store_true",
                   help="v2: run forced alignment + print word timings, exit")
    p.add_argument("--plan-only", action="store_true",
                   help="v2: run alignment + AI shot planner, save plan JSON, exit")
    p.add_argument("--verbose", action="store_true")

    # v2 alignment
    p.add_argument("--align-backend", default="auto",
                   choices=["auto", "whisperx", "whisper", "interpolated"])

    # Output extras
    p.add_argument("--save-keywords", metavar="PATH")
    p.add_argument("--save-plan", metavar="PATH",
                   default="output/shot_plan.json",
                   help="Path for the v2 shot plan JSON [default: output/shot_plan.json]")
    p.add_argument("--save-alignment", metavar="PATH",
                   default="output/word_timings.json",
                   help="Path for the v2 word-timings JSON [default: output/word_timings.json]")
    p.add_argument("--thumbnail", action="store_true")

    return p


# ── Existing helpers (unchanged) ─────────────────────────────────────────── #

def _print_plan_v1(sections, durations) -> None:
    print("\n── Section Plan " + "─" * 50)
    total = sum(durations)
    for sec, dur in zip(sections, durations):
        bar = int(dur / total * 40)
        print(f"  {sec.section_id:<14}  {dur:6.1f}s  {'█' * bar}")
    print(f"  {'TOTAL':<14}  {total:6.1f}s")
    print("─" * 66 + "\n")


def _run_keywords_only(args, script_text: str, anthropic_key: str) -> dict:
    from phase3.parser import parse_sections, estimate_durations
    from phase3.keywords import generate_keywords, _fallback as _kw_fallback

    sections = parse_sections(script_text)
    durations = estimate_durations(
        sections, args.audio_duration or len(script_text) / 12.0)
    _print_plan_v1(sections, durations)

    print("── Generating keywords " + "─" * 44)
    if anthropic_key:
        keywords = generate_keywords(
            sections, args.genre, anthropic_key,
            book_title=args.book_title,
            character_name=args.character_name,
        )
    else:
        print("  (no Anthropic key — using genre fallbacks)")
        keywords = [_kw_fallback(s, args.genre) for s in sections]

    result = {}
    for kw in keywords:
        result[kw.section_id] = {
            "wikimedia": kw.wikimedia,
            "pexels": kw.pexels,
            "key_phrases": kw.key_phrases,
        }
        print(f"\n  [{kw.section_id}]")
        print(f"    Wikimedia:   {kw.wikimedia}")
        print(f"    Pexels:      {kw.pexels}")
        if kw.key_phrases:
            print(f"    Key phrases: {kw.key_phrases}")

    print()
    return result


# ── v2 helpers ───────────────────────────────────────────────────────────── #

def _resolve_audio_duration(audio_path: Path | None, script_text: str,
                            override: float | None) -> float:
    """Resolve total audio duration: explicit > ffprobe > char-rate estimate."""
    if override:
        return override
    if audio_path and audio_path.exists():
        try:
            from phase3.effects import probe_duration
            return probe_duration(audio_path)
        except Exception as exc:
            print(f"  ffprobe failed ({exc}); falling back to char estimate")
    return min(360.0, max(60.0, len(script_text.strip()) / 12.0))


def _run_align_only(args, script_text: str) -> None:
    """Run forced alignment and print/save the word timings."""
    from phase3.align import align, tokenize_script
    from phase3.parser import parse_sections

    audio_path = Path(args.audio) if args.audio else None
    total_dur = _resolve_audio_duration(
        audio_path, script_text, args.audio_duration)

    sections = parse_sections(script_text)
    tokens = tokenize_script(script_text)
    print(f"\nScript : {len(tokens)} word tokens, {len(sections)} sections")
    print(f"Audio  : {audio_path or '(none — will interpolate)'}")
    print(f"Total  : {total_dur:.1f} s")
    print(f"Backend: {args.align_backend}\n")

    print("── Running alignment " + "─" * 46)
    t0 = time.perf_counter()
    timings = align(
        script_text, audio_path, total_dur,
        prefer_backend=args.align_backend,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Aligned {len(timings)} words in {elapsed:.1f} s")
    print(f"  Backend used: {timings[0].source if timings else 'n/a'}")
    print()

    # Print first 30 word timings as a sanity check
    print("── First 30 word timings " + "─" * 42)
    for w in timings[:30]:
        print(f"  {w.start:7.2f}s → {w.end:7.2f}s   ({w.duration:.2f}s)   {w.word}")
    if len(timings) > 30:
        print(f"  … and {len(timings) - 30} more")
    print()

    if args.save_alignment:
        out_path = Path(args.save_alignment)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"word": w.word, "start": w.start, "end": w.end, "source": w.source}
            for w in timings
        ]
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Word timings → {out_path}")


def _run_plan_only(args, script_text: str, anthropic_key: str) -> None:
    """Run alignment + AI shot planner, print summary, save JSON."""
    if not anthropic_key:
        print("ERROR: --plan-only requires an Anthropic API key", file=sys.stderr)
        sys.exit(1)

    from phase3.align import align, tokenize_script
    from phase3.parser import parse_sections
    from phase3.plan import build_shot_plan, save_plan, summarise_plan

    audio_path = Path(args.audio) if args.audio else None
    total_dur = _resolve_audio_duration(
        audio_path, script_text, args.audio_duration)

    sections = parse_sections(script_text)
    if not sections:
        print("ERROR: no sections parsed from script", file=sys.stderr)
        sys.exit(1)

    tokens = tokenize_script(script_text)
    print(f"\nScript : {len(tokens)} word tokens, {len(sections)} sections")
    print(f"Audio  : {audio_path or '(none — will interpolate)'}")
    print(f"Total  : {total_dur:.1f} s\n")

    # Step 1: align
    print("── Step 1: Forced alignment " + "─" * 39)
    t0 = time.perf_counter()
    timings = align(
        script_text, audio_path, total_dur,
        prefer_backend=args.align_backend,
    )
    elapsed = time.perf_counter() - t0
    backend = timings[0].source if timings else "n/a"
    print(f"  {len(timings)} words aligned via {backend} ({elapsed:.1f}s)")
    print()

    # Step 2: plan
    print("── Step 2: AI shot planner " + "─" * 40)
    t0 = time.perf_counter()
    plan_path = Path(args.save_plan) if args.save_plan else None
    debug_dir = plan_path.parent if plan_path else Path("output")
    shots = build_shot_plan(
        sections=sections,
        word_timings=timings,
        book_title=args.book_title,
        character_name=args.character_name,
        genre=args.genre,
        total_duration_sec=total_dur,
        anthropic_api_key=anthropic_key,
        debug_dir=debug_dir,
    )
    elapsed = time.perf_counter() - t0
    print(f"  {len(shots)} shots planned in {elapsed:.1f}s")
    print()

    # Print summary
    print(summarise_plan(shots))
    print()

    # Save
    if args.save_plan:
        out_path = Path(args.save_plan)
        save_plan(shots, out_path)
        print(f"Shot plan → {out_path}")

    if args.save_alignment:
        out_path = Path(args.save_alignment)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"word": w.word, "start": w.start, "end": w.end, "source": w.source}
            for w in timings
        ]
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Word timings → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────── #

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s  %(name)s  %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("phase3").setLevel(level)

    _load_dotenv(Path(".env"))
    _load_dotenv(_HERE / ".env")

    anthropic_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")
    pexels_key = args.pexels_key or os.environ.get("PEXELS_API_KEY", "")

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: script file not found: {script_path}", file=sys.stderr)
        return 1
    script_text = script_path.read_text(encoding="utf-8")
    print(f"\nScript : {script_path}  ({len(script_text):,} chars)")

    # Dispatch on mode
    if args.dry_run:
        from phase3.parser import parse_sections, estimate_durations
        dur_hint = args.audio_duration or len(script_text) / 12.0
        sections = parse_sections(script_text)
        durations = estimate_durations(sections, dur_hint)
        print(f"Sections found: {len(sections)}")
        _print_plan_v1(sections, durations)
        return 0

    if args.keywords_only:
        kw_data = _run_keywords_only(args, script_text, anthropic_key)
        if args.save_keywords:
            kw_path = Path(args.save_keywords)
            kw_path.parent.mkdir(parents=True, exist_ok=True)
            kw_path.write_text(
                json.dumps(kw_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Keywords saved → {kw_path}")
        return 0

    if args.align_only:
        _run_align_only(args, script_text)
        return 0

    if args.plan_only:
        _run_plan_only(args, script_text, anthropic_key)
        return 0

    # ── Full v1 render ───────────────────────────────────────────────── #
    audio_bytes: bytes | None = None
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
            return 1
        audio_bytes = audio_path.read_bytes()
        print(f"Audio  : {audio_path}  ({len(audio_bytes) / 1024:.0f} KB)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output : {output_path}")
    print(f"Genre  : {args.genre}   Grade: {args.color_grade}   "
          f"{args.width}×{args.height}")
    print(f"Claude : {'✓  (keywords + vision scoring)' if anthropic_key else '✗  (genre fallbacks only)'}")
    print(f"Pexels : {'✓' if pexels_key else '✗  (Wikimedia images only)'}")
    print()

    from phase3 import generate_background_video

    t0 = time.perf_counter()
    try:
        result = generate_background_video(
            script_text=script_text,
            output_path=output_path,
            audio_bytes=audio_bytes,
            audio_duration_sec=args.audio_duration,
            anthropic_api_key=anthropic_key,
            pexels_api_key=pexels_key,
            genre=args.genre,
            color_grade=args.color_grade,
            width=args.width,
            height=args.height,
            images_per_section=args.images_per_section,
            book_title=args.book_title,
            character_name=args.character_name,
            add_subtitles=not args.no_subtitles,
            on_progress=_make_progress(args.verbose),
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    elapsed = time.perf_counter() - t0
    size_mb = result.stat().st_size / (1024 * 1024)
    print(f"\nDone in {elapsed:.0f}s  —  {result}  ({size_mb:.1f} MB)")

    if args.thumbnail:
        from phase3.compositor import extract_thumbnail
        thumb = result.with_suffix(".jpg")
        try:
            extract_thumbnail(result, thumb, time=5.0)
            print(f"Thumbnail → {thumb}")
        except Exception as exc:
            print(f"Thumbnail failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
