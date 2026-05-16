"""
Phase 3 sources — shared types and base classes.

ImageCandidate is the unit of currency.  Every source returns these;
the vision scorer enriches them with scores; the orchestrator picks
the best for each shot.

The Source abstract base class defines the contract every concrete
source (LoC, Wikimedia, Internet Archive, Pexels) must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)


SourceName = Literal[
    "loc",            # Library of Congress
    "wikimedia",      # Wikimedia Commons
    "internet_archive",
    "pexels",
    "user_upload",    # User-supplied image
    "book_extract",   # Phase 1a-extracted photo from the source PDF
]


@dataclass
class ImageCandidate:
    """
    One candidate image for a shot.

    All sources return ImageCandidate instances.  Fields are populated
    progressively:

    - Source returns:        url, title, license_short, source, width, height
    - Cache layer adds:      local_path
    - Vision scorer adds:    score_subject, score_quality, score_cinematic,
                             vision_reason
    """

    # Required at construction
    url: str                              # Direct URL to the bitmap
    title: str                            # Human-readable description
    source: SourceName                    # Where it came from

    # Optional metadata
    license_short: str = ""               # e.g. "PD", "CC-BY-4.0"
    license_url: str = ""                 # Link to license terms
    width: int = 0                        # Pixel width (0 = unknown)
    height: int = 0                       # Pixel height
    source_url: str = ""                  # URL of the source page (for attribution)
    source_query: str = ""                # The query that found this candidate

    # Populated by the cache layer after download
    local_path: Path | None = None

    # Populated by the vision scorer
    score_subject: int = -1               # 0-3, -1 = not scored
    score_quality: int = -1               # 0-3
    score_cinematic: int = -1             # 0-3
    vision_reason: str = ""               # One-line rationale from Claude

    @property
    def total_score(self) -> int:
        """Sum of vision scores. -1 if not scored."""
        if self.score_subject < 0:
            return -1
        return self.score_subject + self.score_quality + self.score_cinematic

    @property
    def is_scored(self) -> bool:
        return self.score_subject >= 0

    def __str__(self) -> str:
        s = (self.total_score if self.is_scored
             else "unscored")
        return f"{self.source}:{self.title[:40]} [{s}]"


@dataclass
class FetchResult:
    """Result of a fetch_for_shot() call — multiple candidates, ranked."""
    query: str
    candidates: list[ImageCandidate]
    best: ImageCandidate | None = None    # Top-scored, downloaded, kept

    @property
    def has_image(self) -> bool:
        return self.best is not None and self.best.local_path is not None


class Source(ABC):
    """Abstract base for any image source."""

    name: SourceName

    @abstractmethod
    def search(self, query: str, n: int = 4) -> list[ImageCandidate]:
        """Search the source for up to n images matching `query`."""

    def download(self, candidate: ImageCandidate, dest: Path) -> Path | None:
        """
        Download the candidate's image to dest.  Returns the path on
        success, None on failure.

        Default implementation handles most HTTP sources via urllib;
        sources with auth (Pexels) override this.
        """
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(
                candidate.url,
                headers={"User-Agent":
                         "Lamahat/1.0 (https://github.com/abdoljh/Lamahat)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 1024:
                log.debug("Source %s: %s too small (%d bytes), skipping",
                          self.name, candidate.title, len(data))
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            candidate.local_path = dest
            log.debug("Source %s ↓ %s → %s (%d KB)",
                      self.name, candidate.title[:40], dest.name,
                      len(data) // 1024)
            return dest
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Source %s: download failed for %s: %s",
                        self.name, candidate.url[:80], exc)
            return None


# ── Utilities ─────────────────────────────────────────────────────────── #

def query_hash(query: str, prefix_len: int = 16) -> str:
    """Stable short hash of a query string for use in cache keys."""
    import hashlib
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:prefix_len]


def ext_from_url(url: str, default: str = ".jpg") -> str:
    """Pick a sensible file extension based on URL path."""
    import urllib.parse
    path = urllib.parse.urlparse(url).path.lower().split("?")[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return default


# License classification — what counts as "free" for our purposes.
# We accept anything Creative Commons, public domain, or explicitly
# permissive.  We reject anything with NC (non-commercial) or ND
# (no-derivatives) restrictions.
_FREE_LICENSE_PREFIXES = (
    "cc-", "cc0", "pd", "public domain", "attribution",
    "no known", "no restrictions",
)
_NONFREE_TERMS = (
    "nc", "non-commercial", "noncommercial",
    "nd", "no derivative", "no-derivative",
    "all rights reserved",
)


def is_free_license(license_str: str) -> bool:
    """Return True if the license string indicates a freely-usable image."""
    ls = (license_str or "").lower().strip()
    if any(term in ls for term in _NONFREE_TERMS):
        return False
    if not ls:
        return True   # Treat unknown as free; assume good faith from APIs
    return any(ls.startswith(p) for p in _FREE_LICENSE_PREFIXES)
