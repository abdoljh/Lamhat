"""
User-supplied image source.

The user can override automatic fetching for any shot by placing an
image in a designated directory.  Two ways to associate images with
shots:

1. **By shot index** — file named `shot_05.jpg` matches shot 5
   (1-indexed, zero-padded).
2. **By manifest** — a `manifest.json` file mapping shot indices or
   queries to filenames, e.g.:

       {
         "5":  "askari_portrait.jpg",
         "12": "mosul_1904.png",
         "query:Arab Revolt 1916": "revolt_photo.tif"
       }

This source isn't queried by query string like the web sources.  The
orchestrator calls `lookup_for_shot(idx, query)` directly with the
shot's identity, and the source returns either a match or nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import ImageCandidate

log = logging.getLogger(__name__)


_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


class UserUploadSource:
    """
    Not a regular Source — doesn't search by query.

    Instead, lookup_for_shot() takes (shot_index, query) and returns a
    candidate if the user has provided one.
    """

    name = "user_upload"

    def __init__(self, user_dir: Path | None):
        """user_dir: directory containing user-supplied images, or None
        to disable."""
        self.user_dir = Path(user_dir) if user_dir else None
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if not self.user_dir or not self.user_dir.exists():
            return {}
        m = self.user_dir / "manifest.json"
        if not m.exists():
            return {}
        try:
            return json.loads(m.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("User manifest at %s is invalid: %s", m, exc)
            return {}

    def lookup_for_shot(self,
                        shot_index: int,
                        query: str) -> ImageCandidate | None:
        """
        Return a candidate for this shot, or None.

        Resolution order:
        1. manifest by shot index ("5": "askari.jpg")
        2. manifest by query ("query:Arab Revolt 1916": "...")
        3. filename pattern shot_NN.{ext} where NN is zero-padded
        """
        if not self.user_dir or not self.user_dir.exists():
            return None

        # 1. Manifest by shot index
        manifest_path = self._manifest.get(str(shot_index))
        if manifest_path:
            p = self.user_dir / manifest_path
            if p.exists():
                return self._candidate(p, f"shot {shot_index} (manifest)")

        # 2. Manifest by query
        manifest_path = self._manifest.get(f"query:{query}")
        if manifest_path:
            p = self.user_dir / manifest_path
            if p.exists():
                return self._candidate(p, f"query match: {query[:40]}")

        # 3. shot_NN.<ext> filename pattern
        for ext in _SUPPORTED_EXTS:
            p = self.user_dir / f"shot_{shot_index:02d}{ext}"
            if p.exists():
                return self._candidate(p, f"shot_{shot_index:02d}{ext}")

        return None

    def _candidate(self, path: Path, title: str) -> ImageCandidate:
        c = ImageCandidate(
            url=path.as_uri(),
            title=f"User: {title}",
            source="user_upload",
            license_short="user-supplied",
            source_url=str(path),
        )
        c.local_path = path
        return c

    def list_available(self) -> list[str]:
        """Return all image filenames in the user dir for inspection."""
        if not self.user_dir or not self.user_dir.exists():
            return []
        return sorted(
            p.name for p in self.user_dir.iterdir()
            if p.suffix.lower() in _SUPPORTED_EXTS
        )
