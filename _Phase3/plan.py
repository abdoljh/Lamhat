"""
Phase 3 — Shot planner.

Takes:
  - The Arabic script (parsed sections)
  - Word-level timings from align.py
  - Book context (title, character, genre)

Produces:
  - A list of Shot records that fully specify the video, shot by shot.

The compositor (next session's work) executes shots without making any
creative decisions — all decisions live in the plan.

Design principles
-----------------
* ONE Claude call per video.  Sonnet 4.6 is the right model: it has the
  reasoning depth to make varied creative choices across 60–80 shots,
  and at ~$0.10/video the cost is negligible.
* The plan is JSON.  Inspectable, diffable, regeneratable.  When the
  output video is disappointing you debug the plan first, the renderer
  second.
* Word timings are the truth.  Shot boundaries always fall on word
  boundaries — never mid-word — so cuts feel intentional.
* Variety is mandated by the prompt.  We tell Sonnet explicitly that
  shots over 6 s without overlay text are forbidden in long-form, and
  that typography cards must account for ~25–35 % of shots.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from .align import WordTiming, assign_words_to_sections, tokenize_script
from .parser import ScriptSection

log = logging.getLogger(__name__)


# ── Shot type taxonomy ───────────────────────────────────────────────────── #

ShotVisual = Literal[
    "portrait",       # primary subject's face — use static_hold or slow_push
    "location",       # places, landscapes, architecture
    "object",         # documents, weapons, artefacts, books
    "archive",        # period photographs, newspapers, maps
    "broll",          # generic atmospheric footage (Pexels)
    "typography",     # full-screen Arabic text card (no image)
    "title_card",     # opens the video: book title + author
    "section_mark",   # short interstitial naming the section
]

ShotMotion = Literal[
    "static_hold",    # no camera move (best for portraits, typography)
    "slow_push",      # 1.00 → 1.08 over the shot
    "fast_push",      # 1.00 → 1.20 (emphasis, 1.5–3 s shots)
    "slow_pull",      # 1.08 → 1.00 (reveal)
    "pan_left",       # horizontal pan
    "pan_right",
    "ken_burns",      # mixed zoom + pan (only for long landscapes)
]

TypographyTemplate = Literal[
    "pull_quote",     # large centred quote with quotation marks
    "name_reveal",    # "{character_name} —" with a date underneath
    "date_stamp",     # full-screen year/date in massive serif
    "chapter_heading", # section interstitial
]


@dataclass
class Shot:
    """A single shot in the video.  All time is in seconds from t=0."""

    # Timing
    start: float
    end: float

    # Visual selection
    visual: ShotVisual
    search_query: str = ""              # English query for the image source
    source_hint: str = "auto"           # "wikimedia" | "loc" | "pexels" | "auto"

    # Motion
    motion: ShotMotion = "slow_push"
    motion_intensity: float = 1.0        # 0.5 = subtle, 1.5 = exaggerated

    # Typography (only for visual="typography")
    typography_template: TypographyTemplate | None = None
    typography_text: str = ""            # the Arabic text to display

    # Caption overlay (Arabic words spoken during this shot)
    caption_text: str = ""               # auto-filled by build_shot_plan
    show_caption: bool = True            # off for title_card / hero moments

    # Optional creative note from the planner (free-form, for debugging)
    note: str = ""

    # Section this shot belongs to (auto-filled)
    section_id: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ── Prompt construction ─────────────────────────────────────────────────── #

_SYSTEM_PROMPT = """\
You are a documentary video editor planning the visual flow of an Arabic \
book-summary video.  You will receive the script, word-by-word audio \
timings, and book context.  Your job is to output a JSON shot list that \
makes the video genuinely cinematic.

Hard rules
----------
1.  Output JSON only.  No markdown, no commentary.
2.  Shot boundaries must fall on word boundaries — start/end times must \
    match one of the word timings provided.
3.  Shot duration limits by visual type:
    - title_card:  2.5 – 6.0 s
    - section_mark:  2.5 – 7.0 s
    - typography, portrait:  2.5 – 10.0 s (need reading / contemplation time)
    - archive, broll, location, object:  2.5 – 8.0 s
    Exceeding these limits is FORBIDDEN.  Documentary pacing favours \
    longer holds (5–8 s) on typography and portraits — don't rush them.
4.  The shot list must cover the entire script with no gaps and no \
    overlaps.  shot[i].end == shot[i+1].start.  The first shot must \
    start at 0.0, and the last shot must end at exactly the total \
    audio duration provided — not later.  Do not plan shots beyond \
    the audio.
5.  Each shot has exactly one `visual` type.  Use the full vocabulary — \
    avoid stacking 5 "location" shots in a row.  Aim for 25–35 % \
    typography shots, distributed across the video.
6.  Search queries are in English, specific, and named.  Bad: "horse". \
    Good: "Arab cavalry Faisal 1916".  For people, use full names.
7.  Typography shots have `typography_text` filled with Arabic text \
    taken VERBATIM from the script — exact wording, including diacritics. \
    Do not summarise, condense, or paraphrase.  If a sentence is too \
    long for typography, pick a shorter contiguous sub-phrase from it. \
    No translation, no invention.  Pull quotes should be the most \
    resonant lines in the script — the lines that would make someone \
    stop scrolling.
8.  Motion choice matters:
    - `static_hold` for portraits and typography (let them land)
    - `slow_push` for most B-roll
    - `fast_push` only on emphasis beats (rare, ≤3 per video)
    - `pan_*` for landscapes and wide compositions
9.  `title_card` and `section_mark` shots are TYPOGRAPHY-ONLY: \
    fill `typography_text` with the relevant Arabic text and leave \
    `search_query` empty.  The renderer composes them from text + \
    designed background, not from an image search.
10. Always open with a `title_card` (3–5 s) showing the book title.
11. Optionally use `section_mark` shots between sections (1.5–2.5 s).

Output schema
-------------
{
  "shots": [
    {
      "start": float,
      "end": float,
      "visual": "portrait" | "location" | "object" | "archive" | \
                "broll" | "typography" | "title_card" | "section_mark",
      "search_query": "English search terms, specific and named",
      "motion": "static_hold" | "slow_push" | "fast_push" | "slow_pull" |\
                "pan_left" | "pan_right" | "ken_burns",
      "motion_intensity": 1.0,
      "typography_template": null | "pull_quote" | "name_reveal" | \
                             "date_stamp" | "chapter_heading",
      "typography_text": "Arabic text or empty string",
      "show_caption": true | false,
      "note": "one-line creative reasoning, optional"
    }
  ]
}
"""

_USER_PROMPT_TMPL = """\
Book title: {book_title}
Main character / subject: {character_name}
Genre: {genre}
Total audio duration: {total_duration:.1f} seconds

Script sections (with word-level timings):
{sections_with_timings}

Notes for the planner:
- Open with a 4-second title_card showing the book title.
- TARGET SHOT COUNT: {target_shots} shots (≈ {avg_duration:.1f} s average). \
  This is a documentary, not social media — favour slightly longer shots \
  over too many short ones.  Returning fewer than {target_shots} shots is \
  acceptable; returning MORE than {target_shots} is not.
- Typography shots: choose the most emotionally resonant lines from the \
  script — quotes that would stand alone as social-media graphics.
- For portrait searches of "{character_name}", use the name in English \
  (e.g. "Jafar al-Askari portrait") plus any known historical context \
  from the script.
- For location searches, ground them: "Baghdad 1920 historical photo" \
  beats "Iraq".
- Avoid stacking same-visual shots: vary portrait → location → typography\
  → archive → location, etc.

Return JSON only.
"""


def _format_sections_for_prompt(
    sections: list[ScriptSection],
    section_word_map: dict[str, tuple[float, float, list[WordTiming]]],
) -> str:
    """
    Build a compact textual representation of each section with its
    word-level timings, suitable for inclusion in the planner prompt.

    Each section is rendered as:
        [section_id]  T=12.30–37.85s  (62 words)
        Arabic text body...
        Word timings: word@12.30 word@12.65 word@13.10 ...
    """
    lines: list[str] = []
    for section in sections:
        info = section_word_map.get(section.section_id)
        if not info:
            continue
        sec_start, sec_end, words = info
        lines.append(
            f"\n[{section.section_id}]  T={sec_start:.2f}–{sec_end:.2f}s  "
            f"({len(words)} words)"
        )
        # Cap section text at ~600 chars for prompt size control
        text_excerpt = section.text.strip()
        if len(text_excerpt) > 600:
            text_excerpt = text_excerpt[:600] + "…"
        lines.append(text_excerpt)
        # Word timings: include every word so the planner can pick exact
        # cut points.  Compact format: "word@1.23".
        timing_line = " ".join(f"{w.word}@{w.start:.2f}" for w in words)
        lines.append(f"WORDS: {timing_line}")

    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────── #

def _sized_target_shots(total_duration_sec: float,
                        target_shot_duration: float) -> int:
    """
    Pick a reasonable shot count given total duration.

    Documentary pacing on biographical content averages 5–6 s per shot.
    We scale linearly for short videos and sub-linearly past 3 minutes
    to keep the output token budget bounded.  Cap at 65.
    """
    if total_duration_sec <= 180:
        target = int(total_duration_sec / target_shot_duration)
    else:
        base = int(180 / target_shot_duration)
        extra = int((total_duration_sec - 180) / 5.5)
        target = base + extra
    return max(8, min(65, target))


def _extract_json_resilient(raw: str) -> dict:
    """
    Parse JSON from a Sonnet response that may be truncated or have
    trailing-comma artefacts.  Strategy:

    1. Try plain json.loads.
    2. If that fails, find the outermost "shots" array and parse shot
       objects one at a time, stopping at the first malformed one.
       This salvages an N-1 shot plan when Sonnet cut off mid-final-shot.
    """
    import re

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Locate "shots": [ … and walk objects manually.
    m = re.search(r'"shots"\s*:\s*\[', raw)
    if not m:
        raise ValueError("No 'shots' array found in response")
    body = raw[m.end():]

    # Walk the body tracking brace depth, extracting top-level shot
    # objects (depth 1 → 0 closes one shot).
    shots: list[dict] = []
    depth = 0
    in_string = False
    escape = False
    obj_start = -1

    for i, ch in enumerate(body):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                obj_text = body[obj_start:i + 1]
                try:
                    shots.append(json.loads(obj_text))
                except json.JSONDecodeError:
                    # First malformed object — stop here, return what we have
                    log.warning(
                        "Salvaged %d shot(s) before encountering "
                        "unparseable object at offset %d", len(shots), i)
                    break
                obj_start = -1
        elif ch == "]" and depth == 0:
            # Clean close of the shots array — done
            break

    if not shots:
        raise ValueError("Could not salvage any shots from response")
    return {"shots": shots}


def build_shot_plan(
    sections: list[ScriptSection],
    word_timings: list[WordTiming],
    *,
    book_title: str,
    character_name: str,
    genre: str,
    total_duration_sec: float,
    anthropic_api_key: str,
    target_shot_duration: float = 5.0,
    model: str = "claude-sonnet-4-6",
    debug_dir: Path | None = None,
) -> list[Shot]:
    """
    Generate a complete shot plan in a single Claude call.

    Parameters
    ----------
    sections             From parser.parse_sections().
    word_timings         From align.align().
    book_title           For prompt context and title card.
    character_name       For portrait queries and name_reveal cards.
    genre                Steers visual choices ("history" prefers archive
                         photographs; "philosophy" prefers typography).
    total_duration_sec   Authoritative total duration.
    anthropic_api_key    Required.
    target_shot_duration Desired average shot length.  Default 4.5 s is
                         documentary pacing; lower → faster cutting.
    model                Sonnet 4.6 by default.  Override for testing.
    debug_dir            If provided, write the raw Sonnet response and
                         (on parse failure) the partial JSON to this
                         directory for inspection.
    """
    from anthropic import Anthropic

    section_word_map = assign_words_to_sections(word_timings, sections)
    target_shots = _sized_target_shots(total_duration_sec, target_shot_duration)

    user_prompt = _USER_PROMPT_TMPL.format(
        book_title=book_title or "unknown",
        character_name=character_name or "not specified",
        genre=genre,
        total_duration=total_duration_sec,
        sections_with_timings=_format_sections_for_prompt(
            sections, section_word_map),
        target_shots=target_shots,
        avg_duration=target_shot_duration,
    )

    log.info("Calling %s for shot plan (target=%d shots, %.1fs avg, %.1fs total)",
             model, target_shots, target_shot_duration, total_duration_sec)

    # Use streaming so very long responses don't timeout client-side and
    # so we get partial output even if the model hits a stop condition
    # before producing the final closing brackets.
    client = Anthropic(api_key=anthropic_api_key)
    raw_parts: list[str] = []
    with client.messages.stream(
        model=model,
        max_tokens=24000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text in stream.text_stream:
            raw_parts.append(text)
        final_message = stream.get_final_message()

    raw = "".join(raw_parts).strip()
    raw = (raw
           .removeprefix("```json")
           .removeprefix("```")
           .removesuffix("```")
           .strip())

    # Always dump the raw response if debug_dir is given — useful for
    # iterating on the prompt without re-running the API call.
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "planner_raw_response.txt").write_text(raw, encoding="utf-8")

    stop_reason = getattr(final_message, "stop_reason", None)
    if stop_reason and stop_reason != "end_turn":
        log.warning("Planner stopped early: stop_reason=%s — may need to "
                    "raise max_tokens or lower target_shots", stop_reason)

    try:
        data = _extract_json_resilient(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        # Always dump on failure so the user can inspect what Sonnet said
        fail_dir = debug_dir or Path("output")
        fail_dir.mkdir(parents=True, exist_ok=True)
        fail_path = fail_dir / "planner_raw_response_FAILED.txt"
        fail_path.write_text(raw, encoding="utf-8")
        log.error("Planner returned unparseable JSON (%s). "
                  "Raw response saved to %s", exc, fail_path)
        raise ValueError(
            f"Shot planner returned invalid JSON: {exc}.  "
            f"Raw response written to {fail_path}"
        ) from exc

    shots_data = data.get("shots", [])
    if not shots_data:
        raise ValueError("Shot planner returned no shots")

    shots = [_shot_from_dict(s) for s in shots_data]

    # Post-process: snap to word boundaries, fill captions, assign sections
    shots = _snap_to_word_boundaries(shots, word_timings)
    shots = _fill_captions(shots, word_timings)
    shots = _assign_sections(shots, section_word_map)
    shots = _normalise_fields(shots)
    shots = _validate_plan(shots, total_duration_sec)

    log.info("Plan produced: %d shots", len(shots))
    return shots


# ── Shot construction & post-processing ─────────────────────────────────── #

def _shot_from_dict(d: dict) -> Shot:
    """Build a Shot from the planner's JSON dict, with defaulting."""
    return Shot(
        start=float(d.get("start", 0.0)),
        end=float(d.get("end", 0.0)),
        visual=d.get("visual", "broll"),
        search_query=d.get("search_query", "") or "",
        source_hint=d.get("source_hint", "auto") or "auto",
        motion=d.get("motion", "slow_push"),
        motion_intensity=float(d.get("motion_intensity", 1.0)),
        typography_template=d.get("typography_template") or None,
        typography_text=d.get("typography_text", "") or "",
        caption_text=d.get("caption_text", "") or "",
        show_caption=bool(d.get("show_caption", True)),
        note=d.get("note", "") or "",
        section_id=d.get("section_id", "") or "",
    )


def _normalise_fields(shots: list[Shot]) -> list[Shot]:
    """
    Enforce field exclusivity rules.  Sonnet sometimes sets both
    typography_text and search_query on title cards; the renderer
    needs each shot to have exactly one source of visual content.

    Rules:
    - title_card, section_mark, typography → keep typography_text,
      clear search_query
    - all other visual types → keep search_query, clear typography_text
      and typography_template
    """
    TYPOGRAPHY_KINDS = {"title_card", "section_mark", "typography"}
    cleaned = 0
    for shot in shots:
        if shot.visual in TYPOGRAPHY_KINDS:
            if shot.search_query:
                shot.search_query = ""
                cleaned += 1
        else:
            if shot.typography_text:
                shot.typography_text = ""
                shot.typography_template = None
                cleaned += 1
    if cleaned:
        log.info("Normalised %d shot(s) with conflicting fields", cleaned)
    return shots


def _snap_to_word_boundaries(
    shots: list[Shot],
    word_timings: list[WordTiming],
) -> list[Shot]:
    """
    Snap each shot's start/end to the nearest actual word boundary.
    Prevents cuts mid-syllable when the planner is slightly off.
    """
    if not word_timings:
        return shots

    word_starts = [w.start for w in word_timings]
    word_ends = [w.end for w in word_timings]

    def _snap(t: float, anchors: list[float]) -> float:
        # Linear scan is fine — a few hundred words at most
        return min(anchors, key=lambda a: abs(a - t))

    snapped: list[Shot] = []
    for shot in shots:
        s = _snap(shot.start, word_starts)
        e = _snap(shot.end, word_ends)
        if e <= s:
            e = s + 1.5  # minimum shot duration
        snapped.append(Shot(**{**asdict(shot), "start": s, "end": e}))
    return snapped


def _fill_captions(
    shots: list[Shot],
    word_timings: list[WordTiming],
) -> list[Shot]:
    """Concatenate the Arabic words spoken during each shot into its caption."""
    for shot in shots:
        words_in_shot = [
            w.word for w in word_timings
            if w.start >= shot.start - 0.05 and w.end <= shot.end + 0.05
        ]
        shot.caption_text = " ".join(words_in_shot)
    return shots


def _assign_sections(
    shots: list[Shot],
    section_word_map: dict[str, tuple[float, float, list[WordTiming]]],
) -> list[Shot]:
    """Tag each shot with the section_id whose time range contains it."""
    section_ranges = [(sid, s, e) for sid, (s, e, _) in section_word_map.items()]
    for shot in shots:
        midpoint = (shot.start + shot.end) / 2
        for sid, s, e in section_ranges:
            if s <= midpoint <= e:
                shot.section_id = sid
                break
    return shots


def _validate_plan(shots: list[Shot], total_duration_sec: float) -> list[Shot]:
    """
    Enforce structural invariants on the shot plan.

    - Sort by start time
    - Eliminate overlaps and gaps by snapping consecutive shots together
    - Trim or extend the final shot to cover the full audio
    - Clip any shots that extend beyond the audio
    - Split only shots that exceed 8.0 s (documentary shots routinely
      run 4–6 s; the 8 s ceiling is for true runaways).  Splits are
      annotated but not duplicated — the renderer can vary motion
      across pieces of a split shot to keep the eye engaged.
    """
    if not shots:
        return shots

    shots = sorted(shots, key=lambda s: s.start)

    # 1. Drop or clip shots that start beyond the audio
    in_bounds: list[Shot] = []
    for shot in shots:
        if shot.start >= total_duration_sec:
            log.warning("Dropping shot at %.2fs (audio ends at %.2fs)",
                        shot.start, total_duration_sec)
            continue
        if shot.end > total_duration_sec:
            log.info("Clipping shot %.2f-%.2fs to %.2fs (audio end)",
                     shot.start, shot.end, total_duration_sec)
            shot = Shot(**{**asdict(shot), "end": total_duration_sec})
        in_bounds.append(shot)
    shots = in_bounds
    if not shots:
        return shots

    # 2. Stitch into a contiguous timeline
    fixed: list[Shot] = []
    for shot in shots:
        if fixed:
            prev_end = fixed[-1].end
            if shot.start != prev_end:
                shot = Shot(**{**asdict(shot), "start": prev_end})
            if shot.end <= shot.start:
                shot = Shot(**{**asdict(shot), "end": shot.start + 2.0})
        fixed.append(shot)

    # 3. Pin the last shot to the audio end
    fixed[-1] = Shot(**{**asdict(fixed[-1]), "end": total_duration_sec})

    # 4. Split only TRUE runaways.  Caps are visual-type aware because
    #    documentary holds for different content types are inherently
    #    different.  Typography needs reading time; portraits need
    #    contemplation time; archive/B-roll can rotate faster.
    HARD_CAPS = {
        "typography":   12.0,
        "portrait":     12.0,
        "archive":      10.0,
        "broll":        10.0,
        "location":     10.0,
        "object":       10.0,
        "section_mark":  7.0,
        "title_card":    7.0,
    }
    TARGET_PIECE = 5.0
    # Floating-point tolerance — shots within 0.1s of the cap are kept
    # intact rather than micro-split.
    TOLERANCE = 0.1
    final: list[Shot] = []
    for shot in fixed:
        cap = HARD_CAPS.get(shot.visual, 8.0)
        if shot.duration <= cap + TOLERANCE:
            final.append(shot)
            continue
        n_pieces = max(2, int(round(shot.duration / TARGET_PIECE)))
        piece_dur = shot.duration / n_pieces
        for k in range(n_pieces):
            piece = Shot(**asdict(shot))
            piece.start = shot.start + k * piece_dur
            piece.end = shot.start + (k + 1) * piece_dur
            piece.note = (piece.note +
                          f" [auto-split {k + 1}/{n_pieces}]").strip()
            final.append(piece)

    # 5. Merge adjacent shots with identical content.
    #    Sometimes the splitter, or the planner itself, produces back-
    #    to-back shots that show the same thing — e.g. two 4.1s archive
    #    shots with the same search_query.  Documentary practice says
    #    those should be one 8.2s shot, not a frame-accurate edit
    #    point with no visual change.  Merging here also consolidates
    #    the caption windows so subtitles don't cut in/out at the
    #    invisible boundary.
    merged: list[Shot] = []
    for shot in final:
        if merged and _shots_can_merge(merged[-1], shot):
            prev = merged[-1]
            # Build the merged shot.  Use the earlier shot's metadata;
            # extend its end time; concatenate the caption text.
            combined = Shot(**asdict(prev))
            combined.end = shot.end
            # Merge captions — only when both are non-empty and differ
            if shot.caption_text and shot.caption_text != prev.caption_text:
                combined.caption_text = (
                    f"{prev.caption_text} {shot.caption_text}".strip()
                )
            merged[-1] = combined
            log.debug("Merged adjacent identical shots: %s [%.2fs]",
                      shot.visual, combined.duration)
        else:
            merged.append(shot)

    if len(merged) != len(final):
        log.info("Merged %d adjacent identical shot(s) — final plan %d shots",
                 len(final) - len(merged), len(merged))
    return merged


def _shots_can_merge(a: Shot, b: Shot) -> bool:
    """
    Return True iff two adjacent shots are identical enough to merge.

    Criteria: same visual type, same source (search_query for image
    shots, typography_text for typography shots).  Also require that
    they're temporally adjacent (b.start == a.end within 0.05s).
    """
    if abs(b.start - a.end) > 0.05:
        return False
    if a.visual != b.visual:
        return False
    # Image-kind shots share when their search_query is identical
    if a.search_query or b.search_query:
        if a.search_query != b.search_query:
            return False
    # Typography-kind shots share when their text is identical
    if a.typography_text or b.typography_text:
        if a.typography_text != b.typography_text:
            return False
    return True


# ── Serialisation ────────────────────────────────────────────────────────── #

def shots_to_json(shots: list[Shot]) -> str:
    """Serialise the plan as pretty JSON for inspection or caching."""
    return json.dumps(
        [asdict(s) for s in shots],
        ensure_ascii=False,
        indent=2,
    )


def shots_from_json(text: str) -> list[Shot]:
    """Deserialise a plan from JSON (round-trip safe with shots_to_json)."""
    data = json.loads(text)
    return [_shot_from_dict(d) for d in data]


def save_plan(shots: list[Shot], path: Path) -> Path:
    """Write the plan to disk; returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(shots_to_json(shots), encoding="utf-8")
    return path


def load_plan(path: Path | str) -> list[Shot]:
    """Load a previously-saved plan."""
    return shots_from_json(Path(path).read_text(encoding="utf-8"))


# ── Plan summary printer ─────────────────────────────────────────────────── #

def summarise_plan(shots: list[Shot]) -> str:
    """Human-readable one-line-per-shot summary, for the CLI."""
    if not shots:
        return "(empty plan)"

    lines: list[str] = []
    lines.append(f"Plan: {len(shots)} shots, "
                 f"{sum(s.duration for s in shots):.1f}s total")
    lines.append("─" * 92)

    # Histogram of visual types
    from collections import Counter
    visual_counts = Counter(s.visual for s in shots)
    histogram = "  ".join(f"{v}:{n}" for v, n in visual_counts.most_common())
    lines.append(f"Visuals: {histogram}")
    lines.append("─" * 92)

    for i, shot in enumerate(shots):
        section_tag = f"[{shot.section_id:>9}]" if shot.section_id else "[         ]"
        timing = f"{shot.start:6.2f}-{shot.end:6.2f}s"
        dur = f"({shot.duration:4.1f}s)"
        kind = f"{shot.visual:<14}"
        motion = f"{shot.motion:<12}"

        if shot.visual == "typography":
            extra = f'"{shot.typography_text[:50]}"'
        else:
            extra = shot.search_query[:60]

        lines.append(
            f"  {i+1:>3}. {section_tag} {timing} {dur} {kind} {motion}  {extra}"
        )
    return "\n".join(lines)
