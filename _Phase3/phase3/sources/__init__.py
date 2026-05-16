"""
Phase 3 sources — orchestrator.

Top-level API
-------------
fetch_for_shot(query, shot_index, ...) → FetchResult

The orchestrator runs a priority waterfall:

  1. User upload by shot index / manifest
  2. Phase 1a book extract (vision-scored against the query)
  3. Cached web result for this query
  4. Live web fetch (LoC → Wikimedia → Internet Archive → Pexels),
     vision-scored, top-ranked candidate kept
  5. Returns FetchResult with .best = None → renderer uses placeholder

Each layer is optional.  When `anthropic_api_key` is empty, vision
scoring is skipped (candidates are kept in source priority order).
When `cache_dir` is None, no caching.  When `user_dir` is None or
`book_extracts` is None, those layers are skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .base import FetchResult, ImageCandidate, Source, ext_from_url
from .book_extract import BookExtractSource
from .cache import ImageCache
from .internet_archive import InternetArchive
from .loc import LibraryOfCongress
from .pexels import Pexels
from .user_upload import UserUploadSource
from .vision import VisionScorer, passes_threshold, rank_candidates
from .wikimedia import WikimediaCommons

log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────── #

@dataclass
class FetcherConfig:
    """All configuration needed to instantiate the fetcher."""
    anthropic_api_key: str = ""
    pexels_api_key: str = ""
    cache_dir: Path | None = None
    user_dir: Path | None = None
    book_extracts: Path | None = None   # Phase 1a photos.zip or directory
    book_title: str = ""
    character_name: str = ""

    # Per-shot fetching params
    n_candidates_per_source: int = 3
    enable_vision: bool | None = None   # None = enable iff API key provided

    @property
    def vision_enabled(self) -> bool:
        if self.enable_vision is False:
            return False
        return bool(self.anthropic_api_key)


# ── Orchestrator ──────────────────────────────────────────────────── #

@dataclass
class Fetcher:
    """Stateful image-fetching orchestrator."""
    config: FetcherConfig

    def __post_init__(self):
        self.cache = ImageCache(self.config.cache_dir) if self.config.cache_dir else None
        self.user_source = UserUploadSource(self.config.user_dir)
        self.book_source = BookExtractSource(self.config.book_extracts)
        self.web_sources: list[Source] = [
            LibraryOfCongress(),
            WikimediaCommons(),
            InternetArchive(),
            Pexels(self.config.pexels_api_key),
        ]
        self.scorer = (
            VisionScorer(self.config.anthropic_api_key)
            if self.config.vision_enabled else None
        )

    # ── Required-images manifest ────────────────────────────────── #

    def build_manifest(self, shots: list, out_path: Path) -> None:
        """
        Write a text manifest listing every image-kind shot so the
        user can review and optionally provide images.

        Format:
          shot_05  portrait      "Jafar al-Askari Iraqi general 1920s"
          shot_08  archive       "Ottoman Empire 1918"
          ...
        """
        from ..render import TYPOGRAPHY_VISUALS  # avoid circular

        lines = []
        lines.append("# Required images for Phase 3 video")
        lines.append("# Drop matching files into your --user-dir to override.")
        lines.append("# Filename pattern: shot_NN.jpg (NN = shot number, 01-indexed)")
        lines.append("# OR use a manifest.json in --user-dir.")
        lines.append("")
        lines.append(f"{'Shot':<10}{'Visual':<14}{'Duration':<10}{'Query':<60}")
        lines.append("─" * 95)

        for i, shot in enumerate(shots, start=1):
            if shot.visual in TYPOGRAPHY_VISUALS:
                continue
            lines.append(
                f"shot_{i:02d}    {shot.visual:<14}"
                f"{shot.duration:>5.1f}s    "
                f'"{shot.search_query}"'
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Manifest written → %s", out_path)

    # ── Main entry point ───────────────────────────────────────── #

    def fetch_for_shot(self, query: str, shot_index: int) -> FetchResult:
        """
        Resolve an image for one shot.  Returns FetchResult; the
        caller uses result.best.local_path (or falls back to placeholder
        if result.has_image is False).
        """

        # 1. User upload — highest priority, bypasses everything
        user_cand = self.user_source.lookup_for_shot(shot_index, query)
        if user_cand:
            log.info("Shot %d: user-supplied image %s",
                     shot_index, user_cand.title)
            return FetchResult(query=query, candidates=[user_cand], best=user_cand)

        # 2. Book extract — second priority, vision-scored against query
        book_cands = self.book_source.all_candidates(query)
        if book_cands and self.scorer:
            for c in book_cands:
                self.scorer.score(c,
                                  book_title=self.config.book_title,
                                  character_name=self.config.character_name,
                                  query=query)
            kept = [c for c in book_cands if passes_threshold(c)]
            if kept:
                best = rank_candidates(kept)[0]
                log.info("Shot %d: book-extracted photo (score %d/9)",
                         shot_index, best.total_score)
                return FetchResult(query=query, candidates=book_cands, best=best)
        elif book_cands and not self.scorer:
            # Book extracts provided but no vision scoring — can't pick
            # the right one for this shot.  Warn once, then skip.
            if not getattr(self, "_book_warned", False):
                log.warning(
                    "Book extracts provided (%d photos) but vision scoring "
                    "is disabled.  Cannot match book photos to specific "
                    "shots without vision — set --anthropic-key or remove "
                    "--no-vision to use them.  Falling back to web sources.",
                    len(book_cands)
                )
                self._book_warned = True

        # 3. Cached web result
        if self.cache:
            cached = self.cache.get(query)
            if cached and cached.has_image:
                log.info("Shot %d: cache hit", shot_index)
                return cached

        # 4. Live web fetch
        result = self._fetch_live(query, shot_index)

        # Store to cache for next time
        if self.cache:
            self.cache.put(result)

        return result

    # ── Live web fetch with vision scoring ──────────────────────── #

    def _fetch_live(self, query: str, shot_index: int) -> FetchResult:
        all_candidates: list[ImageCandidate] = []
        n = self.config.n_candidates_per_source

        for src in self.web_sources:
            try:
                cands = src.search(query, n=n)
            except Exception as exc:
                log.warning("Source %s raised: %s", src.name, exc)
                cands = []
            all_candidates.extend(cands)

        if not all_candidates:
            log.warning("Shot %d: no candidates from any web source for %r",
                        shot_index, query)
            return FetchResult(query=query, candidates=[], best=None)

        # Download each candidate so we can vision-score it.
        # Each download is guarded — one corrupt URL doesn't kill the run.
        if self.cache:
            for i, c in enumerate(all_candidates):
                if c.local_path:
                    continue
                try:
                    dest = self.cache.candidate_path(
                        query, i, ext=ext_from_url(c.url))
                    src = self._source_by_name(c.source)
                    if src:
                        src.download(c, dest)
                except Exception as exc:
                    log.warning("Download failed for candidate %d of %r: %s",
                                i, query, exc)
        else:
            # No cache — use a temp dir
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp(prefix="lamahat_fetch_"))
            for i, c in enumerate(all_candidates):
                try:
                    dest = tmp_dir / f"cand_{i:02d}{ext_from_url(c.url)}"
                    src = self._source_by_name(c.source)
                    if src:
                        src.download(c, dest)
                except Exception as exc:
                    log.warning("Download failed for candidate %d of %r: %s",
                                i, query, exc)

        # Drop candidates that failed to download
        downloaded = [c for c in all_candidates if c.local_path]
        if not downloaded:
            log.warning("Shot %d: all downloads failed for %r",
                        shot_index, query)
            return FetchResult(query=query, candidates=all_candidates, best=None)

        # Vision-score (also guarded — score() is fail-open internally
        # but extra belt-and-braces here)
        if self.scorer:
            for c in downloaded:
                try:
                    self.scorer.score(
                        c,
                        book_title=self.config.book_title,
                        character_name=self.config.character_name,
                        query=query,
                    )
                except Exception as exc:
                    log.warning("Scoring failed for %s: %s", c.title[:40], exc)
                    # Apply neutral score so the candidate stays in the pool
                    c.score_subject = 2
                    c.score_quality = 2
                    c.score_cinematic = 1
                    c.vision_reason = f"[error] {exc!s}"[:200]
            kept = [c for c in downloaded if passes_threshold(c)]
            if not kept:
                log.warning("Shot %d: all %d candidates failed threshold",
                            shot_index, len(downloaded))
                return FetchResult(query=query, candidates=downloaded, best=None)
            ranked = rank_candidates(kept)
        else:
            # No vision — keep source-priority order
            ranked = downloaded

        best = ranked[0]
        log.info("Shot %d: best=%s (score %s)",
                 shot_index, best.title[:50],
                 best.total_score if best.is_scored else "n/a")
        return FetchResult(query=query, candidates=downloaded, best=best)

    def _source_by_name(self, name: str) -> Source | None:
        for src in self.web_sources:
            if src.name == name:
                return src
        return None


# ── Convenience export ─────────────────────────────────────────── #

def fetch_for_shot(query: str, shot_index: int, config: FetcherConfig) -> FetchResult:
    """One-shot wrapper.  Build a Fetcher and run one query."""
    return Fetcher(config).fetch_for_shot(query, shot_index)


__all__ = [
    "Fetcher", "FetcherConfig", "FetchResult", "ImageCandidate",
    "fetch_for_shot",
]
