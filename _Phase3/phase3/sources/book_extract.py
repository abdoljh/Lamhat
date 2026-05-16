"""
Book-extracted images source — uses Phase 1a's photo extractions.

Phase 1a extracts photographs from the source PDF.  For biography
content, *the book's own photographs* are usually far better than
anything web search returns — they're the photographs the book's
editor chose for that subject, scanned and cropped already.

The matching strategy: vision-score each extracted photo against the
shot's query, keep the best.  This source doesn't search by query
text; it returns the full set of book photos and lets the orchestrator
ask the vision scorer to pick the right one.

Input format: Phase 1a writes extracted photos to a ZIP or a
directory.  This source can read either.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from .base import ImageCandidate

log = logging.getLogger(__name__)


_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


class BookExtractSource:
    """
    Not a regular Source.  Returns the full book-photo bank; the
    orchestrator decides which to use.

    Construct with either a directory or a ZIP file path (typically
    Phase 1a's photos_zip_path output).  If a ZIP is provided, photos
    are extracted to a working directory on first access.
    """

    name = "book_extract"

    def __init__(self,
                 source: Path | None,
                 work_dir: Path | None = None):
        self.source = Path(source) if source else None
        self.work_dir = (Path(work_dir) if work_dir
                         else (Path.home() / ".cache" / "lamahat"
                               / "book_extracts"))
        self._photos: list[Path] | None = None

    def _materialize(self) -> list[Path]:
        """Extract ZIP if needed and return list of image paths."""
        if self._photos is not None:
            return self._photos
        if not self.source or not self.source.exists():
            self._photos = []
            return self._photos

        if self.source.is_dir():
            photos = [
                p for p in sorted(self.source.iterdir())
                if p.suffix.lower() in _SUPPORTED_EXTS
            ]
        elif self.source.suffix.lower() == ".zip":
            # Extract to a stable location keyed on the ZIP's basename
            extract_dir = self.work_dir / self.source.stem
            extract_dir.mkdir(parents=True, exist_ok=True)
            if not any(extract_dir.iterdir()):
                with zipfile.ZipFile(self.source) as zf:
                    for name in zf.namelist():
                        fname = Path(name).name
                        if fname and Path(fname).suffix.lower() in _SUPPORTED_EXTS:
                            (extract_dir / fname).write_bytes(zf.read(name))
            photos = [
                p for p in sorted(extract_dir.iterdir())
                if p.suffix.lower() in _SUPPORTED_EXTS
            ]
        else:
            log.warning("BookExtract: %s is neither a directory nor ZIP",
                        self.source)
            photos = []

        self._photos = photos
        log.info("BookExtract: %d photos available from %s",
                 len(photos), self.source.name if self.source else "(none)")
        return photos

    def all_candidates(self, query: str) -> list[ImageCandidate]:
        """
        Return all book-extracted photos as candidates for the query.
        The orchestrator vision-scores them against the query and
        keeps the best.

        The vision scorer is the matching layer here — we don't try to
        filter by query text (book photos rarely have useful captions).
        """
        candidates: list[ImageCandidate] = []
        for i, photo in enumerate(self._materialize()):
            c = ImageCandidate(
                url=photo.as_uri(),
                title=f"Book photo {i+1}: {photo.name}",
                source="book_extract",
                license_short="from-source",   # licensing inherited from book
                source_url=str(photo),
                source_query=query,
            )
            c.local_path = photo
            candidates.append(c)
        return candidates
