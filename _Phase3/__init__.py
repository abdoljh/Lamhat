"""
Phase 3 — Visual Generation.

Entry point: generate_background_video()

Pipeline
--------
1. parse_sections()        Split script text into structural sections.
2. estimate_durations()    Distribute audio duration across sections by char count.
3. generate_keywords()     Claude Haiku → Wikimedia + Pexels search terms per section.
4. fetch_section_images()  Download freely-licensed photos from Wikimedia Commons.
5. fetch_section_clip()    Download Pexels stock clip as fallback (if key supplied).
6. assemble_background_video()
   └─ Ken Burns effect on each image  →  concat mini-clips per section
   └─ xfade crossfade between sections
   └─ colour-grade the assembled video
   └─ write silent background .mp4
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable

from .compositor import assemble_background_video, extract_thumbnail, mux_final_video
from .keywords   import generate_keywords, _fallback as _kw_fallback
from .parser     import ScriptSection, estimate_durations, parse_sections
from .pexels     import fetch_section_clip
from .subtitler  import write_ass
from .wikimedia  import fetch_section_images, score_images

log = logging.getLogger(__name__)

__all__ = ["generate_background_video"]


def generate_background_video(
    script_text: str,
    output_path: Path,
    *,
    audio_bytes: bytes | None = None,
    audio_duration_sec: float | None = None,
    anthropic_api_key: str = "",
    pexels_api_key: str = "",
    genre: str = "history",
    color_grade: str = "warm",
    width: int = 1280,
    height: int = 720,
    images_per_section: int = 3,
    book_title: str = "",
    character_name: str = "",
    add_subtitles: bool = True,
    on_progress: Callable[[str, float], None] | None = None,
) -> Path:
    """
    Convert an Arabic video script into a complete video with visuals,
    audio (if provided) and burned-in Arabic subtitles.

    Parameters
    ----------
    script_text          Full Arabic script text (plain or diacritized).
    output_path          Where to write the final .mp4.
    audio_bytes          MP3 bytes from Phase 2 TTS (muxed into final video).
    audio_duration_sec   Override: total audio duration in seconds.
                         If None, derived from audio_bytes or estimated from chars.
    anthropic_api_key    For Claude Haiku keyword generation (optional; falls
                         back to genre defaults if absent).
    pexels_api_key       For Pexels video clips (optional; Wikimedia images
                         are used without any key).
    genre                Book genre — affects keyword fallbacks and colour grade.
    color_grade          'warm' | 'cool' | 'neutral'.
    width / height       Output resolution (default 1280×720).
    images_per_section   Max Wikimedia images fetched per section (default 3).
    book_title           Book title passed to keyword generator for context.
    character_name       Main character / subject name for portrait searches.
    add_subtitles        Burn Arabic subtitles into the video (default True).
    on_progress          Callback(step_label: str, fraction: float).

    Returns
    -------
    output_path (Path) — the finished video (visuals + audio + subtitles).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _prog(label: str, frac: float) -> None:
        log.info("[P3 %.0f%%] %s", frac * 100, label)
        if on_progress:
            on_progress(label, frac)

    # ── Step 1: Parse script sections ──────────────────────────────── #
    _prog("Parsing script sections…", 0.02)
    sections = parse_sections(script_text)
    if not sections:
        raise ValueError("No recognisable sections found in script text.")
    log.info("Parsed %d sections: %s",
             len(sections), [s.section_id for s in sections])

    # ── Step 2: Resolve audio duration ─────────────────────────────── #
    if audio_duration_sec is None:
        audio_duration_sec = _resolve_duration(audio_bytes, script_text)
    durations = estimate_durations(sections, audio_duration_sec)
    log.info("Section durations (sec): %s",
             [f"{d:.1f}" for d in durations])

    # ── Step 3: Generate keywords ───────────────────────────────────── #
    _prog("Generating visual keywords…", 0.06)
    if anthropic_api_key:
        keywords = generate_keywords(
            sections, genre, anthropic_api_key,
            book_title=book_title,
            character_name=character_name,
        )
    else:
        keywords = [_kw_fallback(s, genre) for s in sections]
    kw_map = {kw.section_id: kw for kw in keywords}

    # ── Steps 4 & 5: Fetch visuals ──────────────────────────────────── #
    with tempfile.TemporaryDirectory(prefix="bk2v_assets_") as _assets:
        assets_dir = Path(_assets)

        images_map: dict[str, list[Path]] = {}
        clips_map:  dict[str, Path | None] = {}

        n_sections = len(sections)

        for i, section in enumerate(sections):
            kw        = kw_map.get(section.section_id)
            wiki_q    = kw.wikimedia if kw else [genre, "Arabic manuscript"]
            pexels_q  = kw.pexels    if kw else [genre, "library"]
            sec_dir   = assets_dir / section.section_id
            sec_dir.mkdir()

            # Log the actual search terms so the user can see what's happening
            log.info("Section %s | wikimedia queries: %s",
                     section.section_id, wiki_q)
            log.info("Section %s | pexels queries: %s",
                     section.section_id, pexels_q)
            if kw and kw.key_phrases:
                log.info("Section %s | key phrases: %s",
                         section.section_id, kw.key_phrases)

            # Wikimedia images — over-fetch when vision scoring is active
            # so the scorer has extra candidates to filter from.
            fetch_max = images_per_section * 2 if anthropic_api_key else images_per_section
            _prog(
                f"[{i+1}/{n_sections}] Wikimedia search · {section.section_id}"
                f" | {wiki_q[0] if wiki_q else '?'}…",
                0.10 + 0.30 * i / n_sections,
            )
            imgs = fetch_section_images(
                queries=wiki_q,
                dest_dir=sec_dir,
                n_per_query=2,
                max_total=fetch_max,
            )
            log.info("Section %s: %d Wikimedia image(s) downloaded",
                     section.section_id, len(imgs))

            # Vision scoring: discard images Claude says are irrelevant
            if anthropic_api_key and imgs:
                _prog(
                    f"[{i+1}/{n_sections}] Checking image relevance · {section.section_id}…",
                    0.10 + 0.30 * (i + 0.3) / n_sections,
                )
                imgs = score_images(
                    imgs,
                    book_title=book_title,
                    character_name=character_name,
                    api_key=anthropic_api_key,
                )
                imgs = imgs[:images_per_section]   # keep at most the target count
                log.info("Section %s: %d image(s) after vision scoring",
                         section.section_id, len(imgs))

            images_map[section.section_id] = imgs
            _prog(
                f"[{i+1}/{n_sections}] {section.section_id}: "
                f"{len(imgs)} relevant image(s)",
                0.10 + 0.30 * (i + 0.5) / n_sections,
            )

            # Pexels video clip: use proactively alongside images if key supplied,
            # not only as a last-resort fallback.
            clip: Path | None = None
            if pexels_api_key:
                _prog(f"[{i+1}/{n_sections}] Pexels clip · {section.section_id}…",
                      0.10 + 0.30 * (i + 0.7) / n_sections)
                clip_dest = assets_dir / f"pexels_{section.section_id}.mp4"
                clip = fetch_section_clip(
                    queries=pexels_q,
                    api_key=pexels_api_key,
                    dest=clip_dest,
                )
                if clip:
                    log.info("Section %s: Pexels clip downloaded → %s",
                             section.section_id, clip.name)
            clips_map[section.section_id] = clip

        # ── Step 6: Assemble background video ────────────────────────── #
        def _asm_prog(label: str, frac: float) -> None:
            _prog(label, 0.40 + 0.42 * frac)   # 40%–82% of total

        bg_path = assets_dir / "background.mp4"
        assemble_background_video(
            sections=sections,
            section_durations=durations,
            images_per_section=images_map,
            clips_per_section=clips_map,
            output_path=bg_path,
            width=width,
            height=height,
            color_grade=color_grade,
            on_progress=_asm_prog,
        )
        log.info("Background video assembled: %s", bg_path)

        # ── Step 7: Write ASS subtitles ───────────────────────────────── #
        ass_path: Path | None = None
        if add_subtitles and script_text.strip():
            _prog("Generating Arabic subtitles…", 0.84)
            ass_path = assets_dir / "subtitles.ass"
            key_phrases_map = {
                kw.section_id: kw.key_phrases
                for kw in keywords
                if kw.key_phrases
            } if anthropic_api_key else {}
            write_ass(
                sections, durations, ass_path,
                book_title=book_title,
                author_name=character_name,
                key_phrases_map=key_phrases_map,
                width=width,
                height=height,
            )
            log.info("ASS subtitle file written: %s", ass_path)

        # ── Step 8: Save audio bytes to temp file ─────────────────────── #
        audio_tmp: Path | None = None
        if audio_bytes:
            audio_tmp = assets_dir / "audio.mp3"
            audio_tmp.write_bytes(audio_bytes)
            log.info("Audio written to temp file: %s", audio_tmp)

        # ── Step 9: Mux background + audio + subtitles ────────────────── #
        # Pass the resolved audio duration so the muxer can hard-trim the
        # output to exactly that length, preventing a silent tail when the
        # background video is longer than the audio.
        _prog("Muxing audio and subtitles into final video…", 0.88)
        mux_final_video(
            background_video=bg_path,
            output_path=output_path,
            audio_path=audio_tmp,
            subtitle_file=ass_path,
            max_duration=audio_duration_sec,
        )

    _prog("Final video complete ✓", 1.0)
    return output_path


# ── Helpers ──────────────────────────────────────────────────────────────── #

def _resolve_duration(audio_bytes: bytes | None, script_text: str) -> float:
    """
    Determine total audio duration.

    Priority:
    1. ffprobe on audio_bytes (most accurate — always used when audio is present)
    2. Character-count estimate for Arabic TTS (~12 chars/sec, max 360 s)

    Arabic TTS rate: gTTS reads Arabic at roughly 120 words/min ≈ 2 words/sec,
    average 6 chars/word including diacritics ≈ 12 chars/sec.
    The hard cap of 360 s (6 min) prevents run-away estimates from over-long
    script files.
    """
    if audio_bytes:
        try:
            import tempfile
            from .effects import probe_duration
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                tmp = Path(f.name)
            try:
                dur = probe_duration(tmp)
                log.info("Audio duration from ffprobe: %.1f s", dur)
                return dur
            finally:
                tmp.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("ffprobe duration failed: %s", exc)

    # Fallback: character-count estimate for Arabic TTS
    # ~12 chars/sec accounts for gTTS Arabic reading speed; cap at 6 min.
    n_chars = len(script_text.strip())
    est     = min(360.0, max(60.0, n_chars / 12.0))
    log.info("Estimated audio duration from chars (%d): %.1f s", n_chars, est)
    return est
