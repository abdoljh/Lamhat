"""
Internet Archive source.

IA's advanced search returns JSON when given fl[] (field list) and
output=json.  Great for period book illustrations, newspapers, and
historical context; rarely the best source for named-person portraits
(LoC and Wikimedia win there).

Useful for queries like "Baghdad 1920 historical street" or
"Ottoman cavalry illustration".
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from .base import ImageCandidate, Source

log = logging.getLogger(__name__)

_API = "https://archive.org/advancedsearch.php"


class InternetArchive(Source):
    name = "internet_archive"

    def search(self, query: str, n: int = 4) -> list[ImageCandidate]:
        # IA's query syntax supports field-specific search.  Restrict
        # to image mediatype to skip audio/video/text results.
        ia_q = f'({query}) AND mediatype:(image)'
        params = [
            ("q", ia_q),
            ("fl[]", "identifier"),
            ("fl[]", "title"),
            ("fl[]", "creator"),
            ("fl[]", "date"),
            ("fl[]", "licenseurl"),
            ("fl[]", "rights"),
            ("rows", str(min(n * 2, 20))),
            ("page", "1"),
            ("output", "json"),
        ]
        url = _API + "?" + urllib.parse.urlencode(params, doseq=True)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent":
                    "Lamahat/1.0 (https://github.com/abdoljh/Lamahat)",
            })
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            log.warning("Internet Archive search failed for %r: %s", query, exc)
            return []

        docs = data.get("response", {}).get("docs", []) or []
        candidates: list[ImageCandidate] = []

        for d in docs:
            ident = d.get("identifier", "")
            if not ident:
                continue

            # IA's image item has a derived JPEG at this URL pattern
            img_url = f"https://archive.org/download/{ident}/{ident}.jpg"

            license_str = d.get("licenseurl") or d.get("rights") or ""
            if isinstance(license_str, list):
                license_str = license_str[0] if license_str else ""

            title = d.get("title", "Untitled")
            if isinstance(title, list):
                title = title[0] if title else "Untitled"

            candidates.append(ImageCandidate(
                url=img_url,
                title=str(title)[:120],
                source="internet_archive",
                license_short=str(license_str)[:30] or "PD",
                source_url=f"https://archive.org/details/{ident}",
                source_query=query,
            ))
            if len(candidates) >= n:
                break

        log.info("Internet Archive: %d candidates for %r",
                 len(candidates), query)
        return candidates
