#!/usr/bin/env python3
"""
render_plan.py — Render a saved shot plan JSON to MP4.

Stage 2 driver: real image fetching from LoC/Wikimedia/Internet
Archive/Pexels, with optional user uploads and Phase 1a book extracts.
Image-shot fallback to placeholder cards when fetching fails.

Usage
-----
    python render_plan.py --plan PATH [options]

Required
--------
    --plan PATH         Shot plan JSON (from phase3_run.py --plan-only)

Audio
-----
    --audio PATH        Phase 2 TTS MP3.  Optional.

Output
------
    --output PATH       Output MP4  [default: output/rough_cut.mp4]
    --width N           [default: 1920]
    --height N          [default: 1080]
    --fps N             [default: 25]
    --no-captions       Skip caption burn-in

Image fetching (Stage 2)
------------------------
    --anthropic-key KEY     Enables vision scoring     (or ANTHROPIC_API_KEY)
    --pexels-key KEY        Enables Pexels source      (or PEXELS_API_KEY)
    --user-dir PATH         User-supplied images (overrides automatic)
    --book-extracts PATH    Phase 1a photos directory or ZIP
    --book-title TEXT       Used in vision-scoring rubric
    --character-name TEXT   Used in vision-scoring rubric
    --cache-dir PATH        Image cache  [default: ~/.cache/lamahat/images]
    --no-cache              Disable disk caching
    --no-vision             Disable Claude vision scoring

Modes
-----
    --build-manifest PATH   Write required-images manifest and exit

Logging
-------
    --verbose               Show DEBUG-level logs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from phase3.plan import load_plan
from phase3.render import RenderConfig, render_video
from phase3.sources import Fetcher, FetcherConfig


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader — no external dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_LAST_PCT = {"v": -1}


def _make_progress(verbose: bool):
    def _on_progress(label: str, frac: float) -> None:
        pct = int(frac * 100)
        if pct != _LAST_PCT["v"] or verbose:
            _LAST_PCT["v"] = pct
            bar_len = 30
            filled = int(bar_len * frac)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {pct:3d}%  {label:<70}",
                  end="", flush=True)
        if frac >= 1.0:
            print()
    return _on_progress


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="render_plan.py",
        description="Render a saved shot plan to MP4 with optional image fetching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--plan", required=True, metavar="PATH")
    ap.add_argument("--audio", metavar="PATH")
    ap.add_argument("--output", default="output/rough_cut.mp4", metavar="PATH")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps",    type=int, default=25)
    ap.add_argument("--no-captions", action="store_true")

    ap.add_argument("--anthropic-key", default="", metavar="KEY")
    ap.add_argument("--pexels-key", default="", metavar="KEY")
    ap.add_argument("--user-dir", metavar="PATH")
    ap.add_argument("--book-extracts", metavar="PATH")
    ap.add_argument("--book-title", default="", metavar="TEXT")
    ap.add_argument("--character-name", default="", metavar="TEXT")
    ap.add_argument("--cache-dir", metavar="PATH")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-vision", action="store_true")

    ap.add_argument("--build-manifest", metavar="PATH",
                    help="Write required-images manifest and exit")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

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
    pexels_key    = args.pexels_key    or os.environ.get("PEXELS_API_KEY", "")

    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"ERROR: plan file not found: {plan_path}", file=sys.stderr)
        return 1
    shots = load_plan(plan_path)
    print(f"\nPlan   : {plan_path}  ({len(shots)} shots)")

    cache_dir = (
        None if args.no_cache
        else (Path(args.cache_dir) if args.cache_dir else None)
    )
    fc = FetcherConfig(
        anthropic_api_key=anthropic_key,
        pexels_api_key=pexels_key,
        cache_dir=cache_dir,
        user_dir=Path(args.user_dir) if args.user_dir else None,
        book_extracts=Path(args.book_extracts) if args.book_extracts else None,
        book_title=args.book_title,
        character_name=args.character_name,
        enable_vision=not args.no_vision,
    )

    # --build-manifest mode: write the manifest and exit
    if args.build_manifest:
        manifest_path = Path(args.build_manifest)
        fetcher = Fetcher(fc)
        fetcher.build_manifest(shots, manifest_path)
        print(f"Manifest written → {manifest_path}")
        return 0

    audio_path = None
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
            return 1
        print(f"Audio  : {audio_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output : {out_path}")
    print(f"Config : {args.width}×{args.height} @ {args.fps} fps  "
          f"captions={'off' if args.no_captions else 'on'}")
    print(f"Vision : {'on' if fc.vision_enabled else 'off'}  "
          f"Cache: {'off' if args.no_cache else 'on'}  "
          f"User dir: {args.user_dir or '–'}  "
          f"Book extracts: {args.book_extracts or '–'}")
    print()

    fetcher = Fetcher(fc)
    cfg = RenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        add_captions=not args.no_captions,
        fetcher=fetcher,
    )

    t0 = time.perf_counter()
    try:
        render_video(
            shots, out_path,
            audio_path=audio_path,
            config=cfg,
            on_progress=_make_progress(args.verbose),
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nDone in {elapsed:.0f}s  —  {out_path}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
