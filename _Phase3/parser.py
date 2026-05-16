"""
Phase 3 — Script section parser.

Splits an Arabic video script into its structural sections and
estimates per-section video durations proportional to character count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Section header patterns (Arabic) ────────────────────────────────────── #
# Each tuple: (compiled regex matching the *start* of a line, section_id)
# Order matters — more specific patterns first.
_SECTION_HEADERS: list[tuple[str, str]] = [
    (r"^النقطة\s+الأولى",   "point_1"),
    (r"^النقطة\s+الثانية",  "point_2"),
    (r"^النقطة\s+الثالثة",  "point_3"),
    (r"^النقطة\s+الرابعة",  "point_4"),
    (r"^النقطة\s+الخامسة",  "point_5"),
    (r"^الخاتمة",           "closing"),
    (r"^تقديم\s+الكتاب",    "cta"),
]

_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.UNICODE), sid) for p, sid in _SECTION_HEADERS
]


@dataclass
class ScriptSection:
    section_id: str   # e.g. "opening", "point_1", "closing", "cta"
    title: str        # first line of the section (the header text)
    text: str         # full section text including header
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text.strip())


def parse_sections(script_text: str) -> list[ScriptSection]:
    """
    Split a Phase-1 Arabic video script into structural sections.

    The opening block (everything before the first recognised header)
    is always returned as section_id='opening'.

    Returns sections in document order.
    """
    lines = script_text.splitlines()

    # Find line indices where a recognised header begins
    boundaries: list[tuple[int, str]] = []          # (line_index, section_id)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern, sid in _COMPILED:
            if pattern.match(stripped):
                boundaries.append((i, sid))
                break

    # Sentinel at end
    boundaries.append((len(lines), "_end"))

    sections: list[ScriptSection] = []

    # Opening: everything before the first named section
    first = boundaries[0][0] if boundaries else len(lines)
    opening_text = "\n".join(lines[:first]).strip()
    if opening_text:
        opening_title = lines[0].strip() if lines else "مقدمة"
        sections.append(ScriptSection("opening", opening_title, opening_text))

    # Named sections
    for idx, (start, sid) in enumerate(boundaries[:-1]):
        end = boundaries[idx + 1][0]
        text = "\n".join(lines[start:end]).strip()
        if not text:
            continue
        title = lines[start].strip()
        sections.append(ScriptSection(sid, title, text))

    return sections


def estimate_durations(
    sections: list[ScriptSection],
    total_duration_sec: float,
) -> list[float]:
    """
    Distribute total_duration_sec across sections proportionally to
    their character counts.

    Returns a list of floats (seconds) in the same order as `sections`.
    Each section gets at least 5 seconds.
    """
    total_chars = sum(s.char_count for s in sections) or 1
    return [
        max(5.0, total_duration_sec * s.char_count / total_chars)
        for s in sections
    ]
