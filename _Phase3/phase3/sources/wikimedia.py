"""
Wikimedia Commons source — refactored to the Source ABC.

The MediaWiki API is well-known.  We filter for bitmap images,
exclude diagrams/anatomy/charts via search syntax, require a free
license, and require a minimum 400px dimension.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from .base import ImageCandidate, Source, is_free_license

log = logging.getLogger(__name__)

_API = "https://commons.wikimedia.org/w/api.php"
_HEADERS = {
    "User-Agent": "Lamahat/1.0 (https://github.com/abdoljh/Lamahat; bot)",
}

_SEARCH_EXCLUSIONS = "-diagram -anatomy -chart -schematic"
_MIN_DIMENSION = 400


class WikimediaCommons(Source):
    name = "wikimedia"

    def search(self, query: str, n: int = 4,
               thumb_width: int = 1280) -> list[ImageCandidate]:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap {_SEARCH_EXCLUSIONS}",
            "gsrnamespace": "6",                       # File:
            "gsrlimit": str(min(n * 3, 30)),
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|mime",
            "iiurlwidth": str(thumb_width),
            "format": "json",
        }
        url = _API + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            log.warning("Wikimedia search failed for %r: %s", query, exc)
            return []

        pages = data.get("query", {}).get("pages", {}) or {}
        candidates: list[ImageCandidate] = []

        for page in pages.values():
            ii_list = page.get("imageinfo") or []
            if not ii_list:
                continue
            ii = ii_list[0]

            mime = ii.get("mime", "")
            if not mime.startswith("image/") or "gif" in mime:
                continue

            url_str = ii.get("thumburl") or ii.get("url") or ""
            if not url_str:
                continue

            # License check
            meta = ii.get("extmetadata", {}) or {}
            lic_val = meta.get("LicenseShortName", {})
            lic = lic_val.get("value", "") if isinstance(lic_val, dict) else ""
            if not is_free_license(lic):
                continue

            w = ii.get("thumbwidth") or ii.get("width", 0)
            h = ii.get("thumbheight") or ii.get("height", 0)
            if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
                continue

            candidates.append(ImageCandidate(
                url=url_str,
                title=page.get("title", "Untitled")[:120],
                source="wikimedia",
                license_short=lic or "PD",
                width=w,
                height=h,
                source_url=ii.get("descriptionurl", ""),
                source_query=query,
            ))

            if len(candidates) >= n:
                break

        log.info("Wikimedia: %d candidates for %r", len(candidates), query)
        return candidates

    def download(self, candidate, dest):
        # Override to use the Wikimedia User-Agent
        try:
            req = urllib.request.Request(candidate.url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 1024:
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            candidate.local_path = dest
            return dest
        except Exception as exc:
            log.warning("Wikimedia: download failed for %s: %s",
                        candidate.url[:80], exc)
            return None
