"""
Phase 3 sources — disk cache.

Layout
------
~/.cache/lamahat/images/
    {query_hash}/
        meta.json            # Query string, scoring metadata, picks
        cand_00.jpg          # All downloaded candidates
        cand_01.jpg
        cand_02.jpg

The cache is content-addressed by the query string (not the shot ID),
so two shots that happen to use the same query share the same cache
entry.  Cache hits are instant; cache misses fall through to live
fetching.

Cache invalidation is intentionally manual — re-running with different
queries naturally creates new entries; if you want to force a re-fetch,
delete the relevant subdirectory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from .base import FetchResult, ImageCandidate, query_hash

log = logging.getLogger(__name__)


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "lamahat" / "images"


class ImageCache:
    """Disk-backed cache for image search results."""

    def __init__(self, root: Path | None = None):
        self.root = root or DEFAULT_CACHE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Path helpers ─────────────────────────────────────────────── #

    def _query_dir(self, query: str) -> Path:
        return self.root / query_hash(query)

    def _meta_path(self, query: str) -> Path:
        return self._query_dir(query) / "meta.json"

    # ── Lookup ────────────────────────────────────────────────────── #

    def get(self, query: str) -> FetchResult | None:
        """
        Return a cached FetchResult for `query`, or None if not cached.

        The cached result includes vision scores and the chosen best
        image, so a cache hit completely bypasses both the API calls
        and the vision scoring pass.
        """
        meta_path = self._meta_path(query)
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Corrupt cache entry %s: %s — ignoring", meta_path, exc)
            return None

        qdir = self._query_dir(query)
        candidates: list[ImageCandidate] = []
        for c in data.get("candidates", []):
            cand = ImageCandidate(**{
                k: v for k, v in c.items()
                if k != "local_path"     # path needs special handling
            })
            local = c.get("local_path")
            if local:
                p = qdir / local
                if p.exists():
                    cand.local_path = p
            candidates.append(cand)

        best_idx = data.get("best_index")
        best = candidates[best_idx] if (
            best_idx is not None and 0 <= best_idx < len(candidates)
        ) else None

        result = FetchResult(
            query=query,
            candidates=candidates,
            best=best,
        )
        log.info("Cache hit for %r: %d candidates, best=%s",
                 query, len(candidates),
                 best.title[:40] if best else "none")
        return result

    # ── Storage ───────────────────────────────────────────────────── #

    def put(self, result: FetchResult) -> None:
        """Persist a FetchResult to disk.  Mutates candidate paths to
        be relative to the query directory for portability."""
        qdir = self._query_dir(result.query)
        qdir.mkdir(parents=True, exist_ok=True)

        best_index: int | None = None
        cand_dicts: list[dict] = []

        for i, cand in enumerate(result.candidates):
            d = asdict(cand)
            # Store local_path as a name within the query dir, not absolute
            if cand.local_path:
                d["local_path"] = cand.local_path.name
            else:
                d["local_path"] = None
            cand_dicts.append(d)
            if result.best is cand:
                best_index = i

        meta = {
            "query": result.query,
            "candidates": cand_dicts,
            "best_index": best_index,
        }
        self._meta_path(result.query).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.debug("Cached %d candidates for %r", len(cand_dicts), result.query)

    # ── Path for new candidate downloads ─────────────────────────── #

    def candidate_path(self, query: str, index: int, ext: str = ".jpg") -> Path:
        """Where a not-yet-downloaded candidate should be saved."""
        qdir = self._query_dir(query)
        qdir.mkdir(parents=True, exist_ok=True)
        return qdir / f"cand_{index:02d}{ext}"

    # ── Stats / inspection ──────────────────────────────────────── #

    def list_queries(self) -> list[str]:
        """Return all cached query strings (for inspection / debugging)."""
        out: list[str] = []
        for d in self.root.iterdir():
            if d.is_dir():
                meta = d / "meta.json"
                if meta.exists():
                    try:
                        out.append(json.loads(meta.read_text())["query"])
                    except Exception:
                        pass
        return out

    def clear(self) -> None:
        """Wipe the entire cache.  Use sparingly."""
        import shutil
        if self.root.exists():
            shutil.rmtree(self.root)
            self.root.mkdir(parents=True)
        log.info("Cleared cache at %s", self.root)
