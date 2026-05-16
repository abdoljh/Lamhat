"""
Phase 3 — Per-section visual keyword generator.

Uses Claude Haiku to produce, in a single call per section:
  1. Wikimedia Commons search terms (historical/documentary photographs)
  2. Pexels video search terms (cinematic fallback clips)
  3. Key phrases — 1-2 impactful Arabic sentences from the section text,
     used as full-screen on-screen text overlays in the final video.

Falls back to genre-based defaults if the API call fails.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .parser import ScriptSection

log = logging.getLogger(__name__)

_SYSTEM = (
    "You generate visual search keywords and key phrases for Arabic book summary videos. "
    "Return ONLY valid JSON — no markdown fences, no explanation."
)

_USER_TMPL = """\
Book title: {book_title}
Main character / subject: {character_name}
Book genre: {genre}
Section ID: {section_id}
Section title: {title}
Section text (first 400 chars): {excerpt}

Produce three lists:

1. "wikimedia": 3-4 specific search terms for Wikimedia Commons historical/documentary photographs
2. "pexels":    2-3 cinematic search terms for Pexels stock video footage
3. "key_phrases": 1-2 short, impactful Arabic phrases (8-15 Arabic words each) taken verbatim
   or lightly condensed from the section text above. These will be displayed as large full-screen
   text overlays in the video — choose the most emotionally resonant or thought-provoking sentences.

Rules for wikimedia / pexels:
- Terms must be in English (Wikimedia and Pexels index in English)
- For history/biography genres prefer real historical photographs of people and places
- If a main character is named, the FIRST wikimedia term must be the character's full name
  (e.g. "Jafar al-Askari" or "Jafar Pasha") to retrieve a portrait photograph
- Be specific: "Arab Revolt 1916" beats "revolution"; "Jafar al-Askari 1921" beats "Arab officer"
- NEVER use a single generic word alone — always combine with a person, place, event, or year:
  BAD: "horse", "army", "soldier"   GOOD: "Arab cavalry 1916", "Ottoman officer uniform WWI"
- Avoid terms that return anatomical diagrams, manuscript illustrations, or charts
- Pexels terms should be cinematic: "desert landscape dawn", "Baghdad historical street"

Rules for key_phrases:
- Must be in Arabic (the language of the section text)
- Each phrase must be a complete, self-contained thought — a sentence, not a fragment
- 8-15 Arabic words each
- Choose sentences that provoke curiosity or convey the section's central insight
- Do NOT translate — use Arabic exactly as it appears in the text (or lightly condensed)

Return ONLY this JSON (no other text):
{{"wikimedia": ["...", "..."], "pexels": ["...", "..."], "key_phrases": ["...", "..."]}}"""


@dataclass
class KeywordSet:
    section_id:  str
    wikimedia:   list[str]
    pexels:      list[str]
    key_phrases: list[str] = field(default_factory=list)


def generate_keywords(
    sections: list[ScriptSection],
    genre: str,
    anthropic_api_key: str,
    book_title: str = "",
    character_name: str = "",
) -> list[KeywordSet]:
    """
    Call Claude Haiku once per section to produce search terms + key phrases.
    Falls back to genre-based defaults on any failure.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=anthropic_api_key)
    results: list[KeywordSet] = []

    for section in sections:
        try:
            prompt = _USER_TMPL.format(
                book_title=book_title or "unknown",
                character_name=character_name or "not specified",
                genre=genre,
                section_id=section.section_id,
                title=section.title,
                excerpt=section.text[:400],
            )
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            raw = (raw
                   .removeprefix("```json")
                   .removeprefix("```")
                   .removesuffix("```")
                   .strip())
            data = json.loads(raw)
            results.append(KeywordSet(
                section_id=section.section_id,
                wikimedia=data.get("wikimedia", [])[:4],
                pexels=data.get("pexels", [])[:3],
                key_phrases=data.get("key_phrases", [])[:2],
            ))
            log.debug("Keywords for %s: wikimedia=%s key_phrases=%s",
                      section.section_id, data.get("wikimedia"), data.get("key_phrases"))
        except Exception as exc:
            log.warning("Keyword gen failed for '%s': %s", section.section_id, exc)
            results.append(_fallback(section, genre))

    return results


# ── Fallback keyword sets by genre ──────────────────────────────────────── #
_FALLBACKS: dict[str, dict] = {
    "history": {
        "wikimedia": ["Arabic history photograph", "Ottoman Empire historical", "historical Arab leader"],
        "pexels":    ["history documentary", "ancient ruins"],
        "key_phrases": [],
    },
    "biography": {
        "wikimedia": ["historical portrait photograph", "Arab leader 20th century"],
        "pexels":    ["biography", "person contemplating"],
        "key_phrases": [],
    },
    "non-fiction": {
        "wikimedia": ["book library historical", "Arabic manuscript"],
        "pexels":    ["library reading", "knowledge"],
        "key_phrases": [],
    },
    "philosophy": {
        "wikimedia": ["Islamic philosophy", "Arabic calligraphy art"],
        "pexels":    ["philosophy thinking", "meditation"],
        "key_phrases": [],
    },
    "science": {
        "wikimedia": ["Islamic golden age science", "Arabic astronomy manuscript"],
        "pexels":    ["science discovery", "laboratory"],
        "key_phrases": [],
    },
    "religion": {
        "wikimedia": ["Islamic art calligraphy", "mosque architecture"],
        "pexels":    ["mosque spiritual", "prayer"],
        "key_phrases": [],
    },
    "novel": {
        "wikimedia": ["Arabic literature", "storytelling art"],
        "pexels":    ["storytelling dramatic", "narrative"],
        "key_phrases": [],
    },
}
_DEFAULT_FALLBACK = {
    "wikimedia": ["Arabic manuscript", "library historical"],
    "pexels":    ["library", "books"],
    "key_phrases": [],
}


def _fallback(section: ScriptSection, genre: str) -> KeywordSet:
    kw = _FALLBACKS.get(genre, _DEFAULT_FALLBACK)
    return KeywordSet(
        section_id=section.section_id,
        wikimedia=list(kw["wikimedia"]),
        pexels=list(kw["pexels"]),
        key_phrases=list(kw.get("key_phrases", [])),
    )
