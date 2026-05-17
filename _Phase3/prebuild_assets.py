#!/usr/bin/env python3
"""
Phase 3 — pre-render asset review pass.

Reads a saved shot plan, runs the full image-fetch waterfall (LoC →
Wikimedia → Internet Archive → Pexels) for every image-needing shot,
downloads all candidates, vision-scores them, and writes a *review
dossier* to disk that the user can audit and edit before the actual
render burns anything to video.

After this finishes, the user opens the review directory in a file
browser, looks at the candidate thumbnails, edits `decisions.json` to
swap candidates / pin a portrait / drop personal images into
`overrides/`, then re-runs `render_plan.py --review-dir <same-dir>`.

The render pass consumes the dossier: for each shot it uses the
override → pinned portrait → chosen candidate, falling back to the
live fetcher only if nothing was pre-resolved.

Usage
-----
  python prebuild_assets.py \\
      --plan          output/al_askari_plan_v2.json \\
      --script        samples/al_askari_script.txt \\
      --audio         output/al_askari_audio.mp3 \\
      --book-title    "مذكرات جعفر العسكري" \\
      --character-name "Jafar al-Askari" \\
      --anthropic-key "$ANTHROPIC_API_KEY" \\
      --pexels-key    "$PEXELS_API_KEY" \\
      --review-dir    output/review/ \\
      --character-portrait /path/to/jafar.jpg  # optional but recommended

Cost
----
~28 image-needing shots × 3 sources × ~3 candidates per source × 1
Haiku vision call each = ~250 Haiku calls.  Plus pooled scoring per
shot.  Empirically ~$0.40-$0.60 with current pricing.  This is paid
once per plan and reused by every subsequent re-render.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Repo-root imports — assumes this file sits at _Phase3/prebuild_assets.py
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3.plan import load_plan
from phase3.sources import Fetcher, FetcherConfig
from phase3.sources.base import is_free_license
from phase3.sources.decisions import (
    CandidateEntry,
    Decisions,
    DECISIONS_FILENAME,
    OVERRIDES_SUBDIR,
    ShotDecision,
    is_image_shot,
    shot_folder_name,
    write_readme,
)


log = logging.getLogger("phase3.prebuild")


# ── Helpers ───────────────────────────────────────────────────────────── #

def _read_script(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _arabic_excerpt_for_shot(shot, full_script: str) -> str:
    """
    Best-effort: pull a short Arabic phrase the shot is "about".
    Prefers shot.typography_text (already the planner's pick).  Falls
    back to shot.caption_text.  Truncates to ~120 chars.
    """
    txt = (getattr(shot, "typography_text", "") or "").strip()
    if not txt:
        txt = (getattr(shot, "caption_text", "") or "").strip()
    if len(txt) > 120:
        txt = txt[:117] + "…"
    return txt


def _short_source_label(source: str, index_in_source: int) -> str:
    """Filename token used in candidate filenames: pexels_a, pexels_b, ..."""
    letter = chr(ord("a") + index_in_source) if index_in_source < 26 else f"{index_in_source}"
    return f"{source}_{letter}"


def _copy_into(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


# ── Per-shot processing ───────────────────────────────────────────────── #

def _process_shot(idx: int, shot, *, fetcher: Fetcher, review_dir: Path,
                  script_text: str) -> ShotDecision | None:
    """Run the waterfall for one shot and write its review folder."""

    if not is_image_shot(shot.visual):
        return None

    shot_dir_name = shot_folder_name(idx, shot.visual)
    shot_dir = review_dir / shot_dir_name
    shot_dir.mkdir(parents=True, exist_ok=True)

    query = (shot.search_query or "").strip()
    duration = float(shot.end - shot.start)

    log.info("Shot %d/%s: query=%r duration=%.1fs",
             idx, shot.visual, query[:60], duration)

    # Run the fetcher — this writes candidates to the on-disk cache, vision-
    # scores them, and picks a winner.  We replay the data back into our
    # review-dir layout.
    try:
        result = fetcher.fetch_for_shot(query=query, shot_index=idx)
    except Exception as exc:
        log.warning("Shot %d: fetcher raised %s — emitting empty decision", idx, exc)
        result = None

    # Build the candidate entry list.  We copy each downloaded cache file
    # into the shot folder so the user can browse without traversing the
    # ~/.cache hierarchy.
    candidates: list[CandidateEntry] = []
    seen_per_source: dict[str, int] = {}
    chosen_entry: CandidateEntry | None = None

    if result is not None:
        for cand in result.candidates:
            n_so_far = seen_per_source.get(cand.source, 0)
            seen_per_source[cand.source] = n_so_far + 1

            label = _short_source_label(cand.source, n_so_far)
            rel_file = ""
            if cand.local_path and Path(cand.local_path).exists():
                ext = Path(cand.local_path).suffix or ".jpg"
                dest = shot_dir / f"{label}{ext}"
                try:
                    _copy_into(Path(cand.local_path), dest)
                    rel_file = f"{shot_dir_name}/{dest.name}"
                except OSError as exc:
                    log.warning("Shot %d: couldn't copy %s → %s: %s",
                                idx, cand.local_path, dest, exc)

            score = cand.total_score if cand.is_scored else -1
            score_breakdown = None
            if cand.is_scored:
                score_breakdown = {
                    "subject":   cand.score_subject,
                    "quality":   cand.score_quality,
                    "cinematic": cand.score_cinematic,
                }

            entry = CandidateEntry(
                source=cand.source,
                title=cand.title,
                url=cand.url,
                file=rel_file,
                score=score,
                score_breakdown=score_breakdown,
                vision_reason=cand.vision_reason,
                width=cand.width,
                height=cand.height,
                license_short=cand.license_short,
            )
            candidates.append(entry)

            # The fetcher's `best` is the one that won; mark it.
            if result.best is not None and (
                cand.url == result.best.url and cand.source == result.best.source
            ):
                chosen_entry = entry

    # Write per-shot artefacts (context, candidates copy).
    arabic = _arabic_excerpt_for_shot(shot, script_text)
    context_lines = [
        f"Shot {idx} — {shot.visual}",
        f"Duration: {duration:.2f} s   (timeline {shot.start:.2f} → {shot.end:.2f} s)",
        f"Search query (English): {query}",
        f"Spoken / typography excerpt (Arabic): {arabic}" if arabic else "",
        "",
        "Candidates:",
    ]
    if not candidates:
        context_lines.append("  (no candidates returned by any source)")
    else:
        for c in candidates:
            score_str = f"score {c.score}/9" if c.score >= 0 else "unscored"
            context_lines.append(
                f"  [{c.source:<16}] {score_str:<12} {c.title[:80]}"
            )
            if c.vision_reason:
                context_lines.append(f"      → {c.vision_reason[:120]}")
    (shot_dir / "context.txt").write_text(
        "\n".join(L for L in context_lines if L is not None) + "\n",
        encoding="utf-8",
    )
    (shot_dir / "candidates.json").write_text(
        json.dumps([c.__dict__ for c in candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    chosen_str = ""
    chosen_url = ""
    chosen_file = ""
    if chosen_entry is not None:
        chosen_str  = f"{chosen_entry.source}:{chosen_entry.title}"
        chosen_url  = chosen_entry.url
        chosen_file = chosen_entry.file

    return ShotDecision(
        visual=shot.visual,
        query=query,
        duration_sec=duration,
        arabic_caption_excerpt=arabic,
        chosen=chosen_str,
        chosen_url=chosen_url,
        chosen_file=chosen_file,
        override=None,
        candidates=candidates,
    )


# ── Main ──────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(
        description="Pre-fetch and score candidate images, then emit a "
                    "review dossier for the user to edit before render.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--plan", type=Path, required=True,
                    help="Saved shot plan JSON (from phase3_run --plan-only)")
    ap.add_argument("--script", type=Path, default=None,
                    help="Original Arabic script (for context.txt excerpts)")
    ap.add_argument("--review-dir", type=Path, required=True,
                    help="Where to write the dossier (created if absent)")
    ap.add_argument("--book-title", default="",
                    help="Arabic book title — used as vision-scoring context")
    ap.add_argument("--character-name", default="",
                    help="Main character name in Latin — vision context "
                         "and search query disambiguation")
    ap.add_argument("--character-portrait", type=Path, default=None,
                    help="Path to a personal photo of the main character.  "
                         "Copied into overrides/character.jpg and pinned "
                         "as `pinned_portrait` in decisions.json — every "
                         "`portrait` shot will then use this single image.")
    ap.add_argument("--anthropic-key", default="",
                    help="ANTHROPIC_API_KEY for Haiku vision scoring")
    ap.add_argument("--pexels-key", default="",
                    help="PEXELS_API_KEY (optional)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Disk cache root.  Defaults to ~/.cache/lamahat/images")
    ap.add_argument("--book-extracts", type=Path, default=None,
                    help="Phase 1a photos.zip or directory.  Vision-scored "
                         "against each shot's query.")
    ap.add_argument("--n-candidates", type=int, default=3,
                    help="Candidates to request per source (default 3)")
    ap.add_argument("--no-vision", action="store_true",
                    help="Skip vision scoring entirely (faster, no API cost; "
                         "candidates are pooled unscored)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s  %(message)s",
    )
    # Don't dump base64 image data into the log on --verbose
    for noisy in ("anthropic", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Load plan ─────────────────────────────────────────────── #
    if not args.plan.exists():
        log.error("Plan file not found: %s", args.plan)
        return 2
    shots = load_plan(args.plan)
    log.info("Loaded plan: %d shots", len(shots))

    n_image_shots = sum(1 for s in shots if is_image_shot(s.visual))
    log.info("Image-needing shots: %d (the rest are typography)", n_image_shots)

    script_text = _read_script(args.script) if args.script else ""

    # ── Prepare review directory ──────────────────────────────── #
    review_dir = args.review_dir.resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    overrides_dir = review_dir / OVERRIDES_SUBDIR
    overrides_dir.mkdir(parents=True, exist_ok=True)
    write_readme(review_dir)

    pinned_portrait_rel: str | None = None
    if args.character_portrait is not None:
        src = args.character_portrait.expanduser().resolve()
        if not src.exists():
            log.error("--character-portrait path does not exist: %s", src)
            return 2
        ext = src.suffix.lower() or ".jpg"
        dest = overrides_dir / f"character{ext}"
        _copy_into(src, dest)
        pinned_portrait_rel = f"{OVERRIDES_SUBDIR}/{dest.name}"
        log.info("Pinned portrait copied → %s", dest)

    # ── Configure the fetcher ─────────────────────────────────── #
    cfg = FetcherConfig(
        anthropic_api_key=args.anthropic_key,
        pexels_api_key=args.pexels_key,
        cache_dir=args.cache_dir,
        user_dir=None,
        book_extracts=args.book_extracts,
        book_title=args.book_title,
        character_name=args.character_name,
        n_candidates_per_source=args.n_candidates,
        enable_vision=(False if args.no_vision else None),
    )
    fetcher = Fetcher(config=cfg)

    # ── Walk shots, build decisions ───────────────────────────── #
    decisions = Decisions(
        book={"title": args.book_title, "character": args.character_name},
        pinned_portrait=pinned_portrait_rel,
        shots={},
    )

    processed = 0
    for idx0, shot in enumerate(shots):
        idx = idx0 + 1                # decisions.json uses 1-indexed
        decision = _process_shot(
            idx=idx, shot=shot,
            fetcher=fetcher,
            review_dir=review_dir,
            script_text=script_text,
        )
        if decision is None:
            continue
        decisions.shots[idx] = decision
        processed += 1

    # ── Persist ───────────────────────────────────────────────── #
    decisions.save(review_dir)

    # ── Summary ───────────────────────────────────────────────── #
    have_chosen     = sum(1 for d in decisions.shots.values() if d.chosen)
    no_candidates   = sum(1 for d in decisions.shots.values() if not d.candidates)
    by_source = {}
    for d in decisions.shots.values():
        if d.chosen:
            src = d.chosen.split(":", 1)[0]
            by_source[src] = by_source.get(src, 0) + 1

    print()
    print("─" * 64)
    print(f"Review dossier ready: {review_dir}")
    print(f"Image shots processed:  {processed}")
    print(f"  with a chosen winner: {have_chosen}")
    print(f"  with no candidates:   {no_candidates}")
    if by_source:
        print("Winner source breakdown:")
        for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"  {src:<16} {n}")
    if pinned_portrait_rel:
        print(f"Pinned portrait:        {pinned_portrait_rel}")
        n_portraits = sum(1 for d in decisions.shots.values()
                          if d.visual == "portrait")
        print(f"  affects {n_portraits} portrait shot(s) at render time")
    print()
    print("Next:")
    print(f"  1. Open {review_dir}/ and review each shot folder.")
    print(f"  2. Edit {review_dir/DECISIONS_FILENAME} to swap candidates,")
    print(f"     drop overrides into {review_dir/OVERRIDES_SUBDIR}/,")
    print(f"     or set 'pinned_portrait'.")
    print(f"  3. Render:")
    print(f"     python render_plan.py --plan {args.plan} \\")
    print(f"         --review-dir {review_dir} \\")
    print(f"         --audio <audio> --output <output.mp4>")
    print("─" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
