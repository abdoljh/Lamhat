"""
Phase 3 — Pexels Video API client.

Downloads stock video clips for sections where Wikimedia has
insufficient images. Free Pexels API key required (pexels.com/api).
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

_VIDEO_SEARCH = "https://api.pexels.com/videos/search"
_UA           = "Bk2Video/1.0 (https://github.com/abdoljh/Bk2Video)"


def _get(url: str, api_key: str, params: dict) -> dict:
    full = url + "?" + urllib.parse.urlencode(params)
    req  = urllib.request.Request(full, headers={
        "Authorization": api_key,
        "User-Agent":    _UA,
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())


def _best_mp4(video: dict, prefer_width: int = 1280) -> dict | None:
    """Return the MP4 video_file entry closest to prefer_width."""
    files = [
        f for f in video.get("video_files", [])
        if f.get("file_type") == "video/mp4" and f.get("link")
    ]
    if not files:
        return None
    files.sort(key=lambda f: abs(f.get("width", 0) - prefer_width))
    return files[0]


def search_videos(
    query: str,
    api_key: str,
    per_page: int = 8,
    min_duration: int = 10,
    max_duration: int = 90,
) -> list[dict]:
    """Return Pexels video entries matching the query and duration range."""
    try:
        data = _get(_VIDEO_SEARCH, api_key, {
            "query":       query,
            "per_page":    per_page,
            "size":        "medium",       # up to 1920 px wide
            "orientation": "landscape",
        })
    except Exception as exc:
        log.warning("Pexels search failed for %r: %s", query, exc)
        return []

    return [
        v for v in data.get("videos", [])
        if min_duration <= v.get("duration", 0) <= max_duration
    ]


def fetch_section_clip(
    queries: list[str],
    api_key: str,
    dest: Path,
    min_duration: int = 15,
    max_duration: int = 90,
) -> Path | None:
    """
    Try each query in sequence until a suitable clip is downloaded.

    Returns the local path on success, None if every query fails.
    """
    for query in queries:
        videos = search_videos(
            query, api_key,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        if not videos:
            continue

        best_file = _best_mp4(videos[0])
        if not best_file:
            continue

        link = best_file["link"]
        try:
            req = urllib.request.Request(link, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            log.info("Pexels ↓ %r → %s (%d KB)",
                     query, dest.name, dest.stat().st_size // 1024)
            return dest
        except Exception as exc:
            log.warning("Pexels download failed for %r: %s", query, exc)

    return None
