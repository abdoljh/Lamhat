"""
Phase 3 — Multi-layer Arabic subtitle / text-overlay generator.

Produces a single ASS file that drives four visual layers, all rendered
by FFmpeg's libass (which handles Arabic RTL / bidi correctly):

  TitleCard   — book title + author, full-screen centred, first 5 seconds
  SectionMark — Arabic section heading, top-centre at each section boundary
  KeyPhrase   — most impactful sentence from each section, large text at screen bottom
  Arabic      — regular caption flow, readable text at screen bottom

Why ASS over SRT or FFmpeg drawtext:
  - libass has full Unicode bidi support → Arabic displays RTL correctly
  - Multiple named styles allow per-event sizing, colour, and position
  - FFmpeg drawtext has no Arabic shaping / bidi engine — text is scrambled

Font: Amiri (package fonts-hosny-amiri in Debian trixie).
      Installed via packages.txt; font family name for ASS is "Amiri".
"""

from __future__ import annotations

import re
from pathlib import Path

from .parser import ScriptSection

# ── Timing constants ─────────────────────────────────────────────────────── #
TITLE_SEC       = 5.0    # book title card duration at t=0
SECTION_MARK_SEC = 2.5   # section header card duration
KEY_PHRASE_SEC  = 3.5    # each key-phrase overlay duration (shortened to leave time for captions)
MIN_CAPTION_SEC = 2.0    # minimum duration for a regular caption block
CAPTION_GAP_SEC = 0.08   # tiny gap between consecutive caption blocks

# Max Arabic words per regular caption block
MAX_CAPTION_WORDS = 10

# ── Section ID → display name (Arabic) ───────────────────────────────────── #
_SECTION_LABELS: dict[str, str] = {
    "opening": "الخطاف الافتتاحي",
    "point_1": "النقطة الأولى",
    "point_2": "النقطة الثانية",
    "point_3": "النقطة الثالثة",
    "point_4": "النقطة الرابعة",
    "point_5": "النقطة الخامسة",
    "closing": "الخاتمة",
    "cta":     "تقديم الكتاب",
}

# ── ASS header template ───────────────────────────────────────────────────── #
# Colours in ASS are &HAABBGGRR (alpha, blue, green, red — little-endian).
# &H00FFFFFF = opaque white    &H00C9A84C = opaque gold
# &H80000000 = 50% black bg   &HC0000000 = 75% black bg
_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: {width}
PlayResY: {height}
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding

Style: TitleCard,Amiri,{title_sz},&H00FFFFFF,&H000000FF,&H00000000,&HC0000000,1,0,0,0,100,100,0,0,3,0,0,5,60,60,60,1
Style: TitleSub,Amiri,{titlesub_sz},&H00C9A84C,&H000000FF,&H00000000,&HC0000000,0,1,0,0,100,100,0,0,3,0,0,5,60,60,{titlesub_v},1
Style: SectionMark,Amiri,{section_sz},&H00C9A84C,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,3,0,0,8,40,40,40,1
Style: KeyPhrase,Amiri,{keyphrase_sz},&H00FFFACD,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,3,1,2,60,60,{keyphrase_v},1
Style: Arabic,Amiri,{caption_sz},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,{caption_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ── Helpers ───────────────────────────────────────────────────────────────── #

def _ts(sec: float) -> str:
    """Seconds → ASS timestamp H:MM:SS.cc"""
    sec = max(0.0, sec)
    h   = int(sec // 3600)
    m   = int((sec % 3600) // 60)
    s   = int(sec % 60)
    cs  = min(99, int(round((sec - int(sec)) * 100)))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(text: str) -> str:
    """Escape ASS special characters."""
    return text.replace("\\", "\\\\").replace("{", r"\{")


def _split_captions(text: str, max_words: int = MAX_CAPTION_WORDS) -> list[str]:
    """
    Split Arabic text into caption blocks of ≤ max_words words.
    Prefers sentence boundaries; sub-divides long sentences at word level.
    """
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<=[.؟!\n])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    for sent in sentences:
        words = sent.split()
        while len(words) > max_words:
            chunks.append(' '.join(words[:max_words]))
            words = words[max_words:]
        if words:
            chunks.append(' '.join(words))

    # Merge very short trailing chunks
    merged: list[str] = []
    for chunk in chunks:
        wc = len(chunk.split())
        if merged and wc < 4 and len(merged[-1].split()) + wc <= max_words:
            merged[-1] += ' ' + chunk
        else:
            merged.append(chunk)

    return merged or [text]


# ── Main generator ────────────────────────────────────────────────────────── #

def generate_ass(
    sections: list[ScriptSection],
    section_durations: list[float],
    *,
    book_title: str = "",
    author_name: str = "",
    key_phrases_map: dict[str, list[str]] | None = None,
    width: int = 1280,
    height: int = 720,
) -> str:
    """
    Build a complete ASS subtitle file and return it as a string.

    Visual timeline per section
    ---------------------------
    Opening section (first only):
      [0 → TITLE_SEC]                TitleCard: book title
      [0 → TITLE_SEC]                TitleSub:  author name (if provided)
      [TITLE_SEC → +SECTION_MARK_SEC] SectionMark: section label
      [after mark → +KEY_PHRASE_SEC each] KeyPhrase overlays
      [remaining time]               Arabic captions (regular)

    All other sections:
      [section_start → +SECTION_MARK_SEC] SectionMark
      [after mark → +KEY_PHRASE_SEC each] KeyPhrase overlays
      [remaining time]               Arabic captions

    Parameters
    ----------
    sections          Parsed script sections.
    section_durations Duration in seconds for each section.
    book_title        Displayed in the opening TitleCard.
    author_name       Displayed below the title (optional).
    key_phrases_map   {section_id: [phrase, ...]} from keyword generator.
    width / height    Must match the output video resolution.
    """
    if key_phrases_map is None:
        key_phrases_map = {}

    # ── Compute font sizes proportional to resolution ─────────────────── #
    # Sizes are chosen for comfortable readability on a 720p screen.
    title_sz     = max(72,  height // 9)     # big title card at t=0
    titlesub_sz  = max(44,  height // 16)    # author subtitle line
    section_sz   = max(52,  height // 13)    # section heading at top
    keyphrase_sz = max(62,  height // 11)    # key phrase — large, at bottom
    caption_sz   = max(48,  height // 14)    # regular captions — clearly readable

    titlesub_v   = title_sz + titlesub_sz + 20  # TitleSub drops below TitleCard centre

    # KeyPhrase sits higher at the bottom than regular captions so they are distinct.
    keyphrase_v  = max(80,  height // 7)         # px from screen bottom (alignment 2)
    caption_v    = max(40,  height // 18)        # px from screen bottom for captions

    header = _ASS_HEADER.format(
        width=width, height=height,
        title_sz=title_sz, titlesub_sz=titlesub_sz, titlesub_v=titlesub_v,
        section_sz=section_sz,
        keyphrase_sz=keyphrase_sz, keyphrase_v=keyphrase_v,
        caption_sz=caption_sz, caption_v=caption_v,
    )

    events: list[str] = []

    def dlg(start: float, end: float, style: str, text: str) -> None:
        end = max(start + 0.5, end)
        events.append(
            f"Dialogue: 0,{_ts(start)},{_ts(end)},{style},,0,0,0,,{_esc(text)}"
        )

    cursor = 0.0   # running clock

    for idx, (section, duration) in enumerate(zip(sections, section_durations)):
        is_opening = (idx == 0)
        section_start = cursor
        overlay_t = section_start   # pointer for overlay events

        # ── Title card (opening section only) ──────────────────────────── #
        if is_opening and book_title:
            title_end = section_start + TITLE_SEC
            dlg(section_start, title_end, "TitleCard", book_title)
            if author_name:
                dlg(section_start, title_end, "TitleSub", author_name)
            overlay_t = title_end

        # ── Section marker ─────────────────────────────────────────────── #
        label = _SECTION_LABELS.get(section.section_id, section.title[:40])
        mark_end = min(overlay_t + SECTION_MARK_SEC, section_start + duration - 1.0)
        if mark_end > overlay_t:
            dlg(overlay_t, mark_end, "SectionMark", label)
            overlay_t = mark_end

        # ── Key phrase overlays ─────────────────────────────────────────── #
        phrases = key_phrases_map.get(section.section_id, [])
        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue
            phrase_end = min(overlay_t + KEY_PHRASE_SEC, section_start + duration - 0.5)
            if phrase_end <= overlay_t:
                break
            dlg(overlay_t, phrase_end, "KeyPhrase", phrase)
            overlay_t = phrase_end

        # ── Regular captions ────────────────────────────────────────────── #
        caption_start = overlay_t
        caption_budget = (section_start + duration) - caption_start

        if caption_budget > MIN_CAPTION_SEC:
            blocks = _split_captions(section.text)
            wc_list = [max(1, len(b.split())) for b in blocks]
            total_wc = sum(wc_list)

            t = caption_start
            for block, wc in zip(blocks, wc_list):
                raw_dur = caption_budget * wc / total_wc
                blk_dur = max(MIN_CAPTION_SEC, raw_dur)
                end_t   = min(t + blk_dur - CAPTION_GAP_SEC,
                              section_start + duration - CAPTION_GAP_SEC)
                end_t   = max(end_t, t + MIN_CAPTION_SEC)
                dlg(t, end_t, "Arabic", block)
                t = end_t + CAPTION_GAP_SEC
                if t >= section_start + duration:
                    break

        cursor += duration

    return header + "\n".join(events) + "\n"


def write_ass(
    sections: list[ScriptSection],
    section_durations: list[float],
    dest: Path,
    *,
    book_title: str = "",
    author_name: str = "",
    key_phrases_map: dict[str, list[str]] | None = None,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Write the ASS file to `dest` and return its path."""
    content = generate_ass(
        sections, section_durations,
        book_title=book_title,
        author_name=author_name,
        key_phrases_map=key_phrases_map,
        width=width,
        height=height,
    )
    dest.write_text(content, encoding="utf-8")
    return dest
