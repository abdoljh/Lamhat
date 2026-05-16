"""
Pexels source — refactored from phase3/pexels.py.

Pexels is the wrong source for historical content (it's modern stock
photography) but the right source for cinematic b-roll: atmospheric
landscapes, contemplative reading shots, generic period-feel imagery.

Requires PEXELS_API_KEY.  Photos endpoint, not videos — Stage 1
removed the video-clip path; here we just fetch still photos and
apply motion in the renderer like any other image.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from .base import ImageCandidate, Source

log = logging.getLogger(__name__)

_API = "https://api.pexels.com/v1/search"


class Pexels(Source):
    name = "pexels"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY", "")

    def search(self, query: str, n: int = 4) -> list[ImageCandidate]:
        if not self.api_key:
            log.debug("Pexels: no API key configured — skipping")
            return []

        params = {
            "query": query,
            "per_page": str(min(n * 2, 20)),
            "orientation": "landscape",
            "size": "large",
        }
        url = _API + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={
                "Authorization": self.api_key,
                "User-Agent":
                    "Lamahat/1.0 (https://github.com/abdoljh/Lamahat)",
            })
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            log.warning("Pexels search failed for %r: %s", query, exc)
            return []

        photos = data.get("photos", []) or []
        candidates: list[ImageCandidate] = []

        for p in photos:
            src = p.get("src", {}) or {}
            # Prefer "large2x" (1920w) then "large" (940w)
            url_str = src.get("large2x") or src.get("large") or src.get("original")
            if not url_str:
                continue

            candidates.append(ImageCandidate(
                url=url_str,
                title=(p.get("alt") or p.get("photographer", "Pexels"))[:120],
                source="pexels",
                license_short="Pexels License",   # Permissive
                license_url="https://www.pexels.com/license/",
                width=p.get("width", 0),
                height=p.get("height", 0),
                source_url=p.get("url", ""),
                source_query=query,
            ))
            if len(candidates) >= n:
                break

        log.info("Pexels: %d candidates for %r", len(candidates), query)
        return candidates
