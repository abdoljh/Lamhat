"""
Phase 3 — Forced alignment of TTS audio to script text.

Produces a flat list of WordTiming records: every Arabic word in the
script paired with its start/end time in the audio.

Why this exists
---------------
The Phase 2 TTS reads a *known* script.  We don't need ASR to discover
what was said — we already have the text.  What we need is to know
*when* each word is spoken.  This is the "forced alignment" problem
and it's solved by phoneme-level alignment models.

Backends (auto-selected in this order)
--------------------------------------
1. WhisperX with the Arabic wav2vec2 model
   (`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`).
   Sub-100 ms accuracy.  CPU-only is workable for 3–5 min clips.

2. Whisper-only (without WhisperX's phoneme alignment).
   Word timings via Whisper's `word_timestamps=True`.
   Drift of 200–500 ms is typical, still useful.

3. Interpolation fallback.
   When no audio-aligning library is installed, distribute time across
   the script by character count.  Quality is poor but the pipeline
   keeps working.

The WhisperX/Whisper backends transcribe the audio independently and
align *their* transcript to *their* timing.  We then map the script's
known words onto that timeline by token order, not by string match —
TTS systems sometimes pronounce a word with a slight phonetic variation
that breaks string equality.  Order is much more reliable.

Output
------
A WordTiming has:
  word    str   — the original script token (with diacritics)
  start   float — seconds from audio start
  end     float — seconds from audio start
  source  str   — "whisperx" | "whisper" | "interpolated"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# Arabic Unicode block matchers — these are what we count as "a word"
# in the script.  Latin tokens (e.g. dates, names quoted in Latin script)
# also count.
_ARABIC_WORD_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\w]+",
    re.UNICODE,
)


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    source: str = "interpolated"   # "whisperx" | "whisper" | "interpolated"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ── Public API ───────────────────────────────────────────────────────────── #

def tokenize_script(script_text: str) -> list[str]:
    """Tokenise the Arabic script into a flat list of word forms."""
    return _ARABIC_WORD_RE.findall(script_text)


def align(
    script_text: str,
    audio_path: Path,
    total_duration_sec: float,
    *,
    prefer_backend: str = "auto",
) -> list[WordTiming]:
    """
    Return one WordTiming per word in `script_text`.

    Parameters
    ----------
    script_text         The full script (any diacritisation).
    audio_path          MP3/WAV from Phase 2 TTS.  May be None for
                        interpolation-only mode.
    total_duration_sec  Authoritative total audio duration in seconds.
                        Used by the interpolation fallback and as a
                        sanity bound on aligner output.
    prefer_backend      "auto" | "whisperx" | "whisper" | "interpolated".
                        "auto" tries whisperx → whisper → interpolated.
    """
    tokens = tokenize_script(script_text)
    if not tokens:
        return []

    backends_to_try: list[str]
    if prefer_backend == "auto":
        backends_to_try = ["whisperx", "whisper", "interpolated"]
    else:
        backends_to_try = [prefer_backend]

    if audio_path is None:
        log.info("No audio path provided — using interpolation fallback")
        return _interpolate(tokens, total_duration_sec)
    if not Path(audio_path).exists():
        log.warning("Audio file not found at %s (cwd=%s) — using interpolation fallback",
                    audio_path, Path.cwd())
        return _interpolate(tokens, total_duration_sec)

    for backend in backends_to_try:
        try:
            if backend == "whisperx":
                timings = _align_whisperx(tokens, Path(audio_path))
            elif backend == "whisper":
                timings = _align_whisper(tokens, Path(audio_path))
            elif backend == "interpolated":
                timings = _interpolate(tokens, total_duration_sec)
            else:
                continue

            if timings:
                if backend == "interpolated":
                    log.warning(
                        "Using interpolated timings — character-rate "
                        "estimates only. For real word-level accuracy, "
                        "install whisperx (pip install whisperx) or "
                        "whisper (pip install openai-whisper)."
                    )
                else:
                    log.info("Aligned %d words via %s backend",
                             len(timings), backend)
                return timings
        except Exception as exc:
            log.warning("Backend %s failed: %s", backend, exc)
            continue

    # Last resort
    log.warning("All aligners failed — interpolating")
    return _interpolate(tokens, total_duration_sec)


# ── WhisperX backend ─────────────────────────────────────────────────────── #

def _align_whisperx(tokens: list[str], audio_path: Path) -> list[WordTiming]:
    """
    Use WhisperX with the Arabic phoneme alignment model for sub-100 ms
    word-level timestamps.

    Notes
    -----
    - WhisperX needs to load both Whisper itself (for transcription) and
      a phoneme model for alignment.  We pin the Arabic phoneme model
      explicitly because WhisperX's auto-detect picks an English model
      for any non-English language it doesn't know about.
    - Runs CPU-only.  Expect ~30–60 s for a 3-minute audio file.
    - Memory: ~600 MB RSS for the small Whisper model + Arabic w2v.
    """
    import whisperx   # type: ignore

    device = "cpu"
    compute_type = "int8"

    # Step 1 — transcribe (Whisper's own timestamps, segment-level)
    log.debug("WhisperX: loading Whisper small model")
    model = whisperx.load_model("small", device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language="ar", batch_size=8)

    # Step 2 — phoneme-level forced alignment
    log.debug("WhisperX: loading Arabic alignment model")
    align_model, metadata = whisperx.load_align_model(
        language_code="ar",
        device=device,
        # The default arabic phoneme model on HuggingFace; WhisperX may
        # already know this string, but pinning is safer.
        model_name="jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    )
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    # Step 3 — flatten to (word, start, end) triples
    asr_words: list[tuple[str, float, float]] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if text and start is not None and end is not None:
                asr_words.append((text, float(start), float(end)))

    if not asr_words:
        raise RuntimeError("WhisperX returned no aligned words")

    return _map_tokens_to_asr(tokens, asr_words, source="whisperx")


# ── Whisper-only backend ─────────────────────────────────────────────────── #

def _align_whisper(tokens: list[str], audio_path: Path) -> list[WordTiming]:
    """
    Use openai-whisper directly (no WhisperX).  Less accurate timestamps
    (Whisper interpolates word timings from its decoder), but doesn't
    require the phoneme model download.
    """
    import whisper   # type: ignore

    log.debug("Whisper: loading small model")
    model = whisper.load_model("small")
    result = model.transcribe(
        str(audio_path),
        language="ar",
        word_timestamps=True,
        verbose=False,
    )

    asr_words: list[tuple[str, float, float]] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if text and start is not None and end is not None:
                asr_words.append((text, float(start), float(end)))

    if not asr_words:
        raise RuntimeError("Whisper returned no word timestamps")

    return _map_tokens_to_asr(tokens, asr_words, source="whisper")


# ── Interpolation fallback ──────────────────────────────────────────────── #

def _interpolate(tokens: list[str], total_duration_sec: float) -> list[WordTiming]:
    """
    Distribute total_duration_sec across tokens by character count.
    This is what the v1 pipeline does implicitly — but exposing it here
    lets the v2 planner consume the same WordTiming shape regardless of
    which backend was used.
    """
    if not tokens:
        return []

    lengths = [max(1, len(t)) for t in tokens]
    total_chars = sum(lengths)
    timings: list[WordTiming] = []
    cursor = 0.0
    for tok, n in zip(tokens, lengths):
        dur = total_duration_sec * n / total_chars
        timings.append(WordTiming(word=tok, start=cursor, end=cursor + dur,
                                  source="interpolated"))
        cursor += dur
    return timings


# ── Token → ASR word alignment ───────────────────────────────────────────── #

def _map_tokens_to_asr(
    tokens: list[str],
    asr_words: list[tuple[str, float, float]],
    *,
    source: str,
) -> list[WordTiming]:
    """
    Pair each script token with an ASR word by position.

    The ASR may produce a slightly different word count than the script
    (mismatched tokenisation, dropped/added words, fillers).  Strategy:

    - If counts match exactly: 1-to-1 pairing.
    - If ASR has fewer words: distribute the missing tokens proportionally
      across the nearest neighbouring ASR words.
    - If ASR has more words: collapse extra ASR words into the nearest
      script token (keep the union of timestamps).

    This is a much simpler heuristic than dynamic-programming alignment
    but works well in practice for TTS audio (which faithfully reads the
    script in order).
    """
    n_script = len(tokens)
    n_asr = len(asr_words)
    if n_script == 0 or n_asr == 0:
        return []

    # Exact match — easy path
    if n_script == n_asr:
        return [
            WordTiming(word=tok, start=s, end=e, source=source)
            for tok, (_, s, e) in zip(tokens, asr_words)
        ]

    # Proportional remapping
    timings: list[WordTiming] = []
    for i, tok in enumerate(tokens):
        # Project script index i ∈ [0, n_script) onto ASR index range
        asr_idx_low = int(i * n_asr / n_script)
        asr_idx_high = int((i + 1) * n_asr / n_script)
        asr_idx_low = max(0, min(n_asr - 1, asr_idx_low))
        asr_idx_high = max(asr_idx_low + 1, min(n_asr, asr_idx_high))

        start = asr_words[asr_idx_low][1]
        end = asr_words[asr_idx_high - 1][2]
        timings.append(WordTiming(word=tok, start=start, end=end, source=source))

    log.info("Token/ASR count mismatch (%d script vs %d ASR) — "
             "used proportional mapping", n_script, n_asr)
    return timings


# ── Convenience: bucket words into sections ─────────────────────────────── #

def assign_words_to_sections(
    word_timings: list[WordTiming],
    sections: list,        # list[ScriptSection] from parser.py
) -> dict[str, tuple[float, float, list[WordTiming]]]:
    """
    Group word timings by the script section they belong to.

    Returns
    -------
    dict[section_id, (section_start_sec, section_end_sec, [WordTiming, ...])]

    The section_start / section_end come from the first and last word's
    timestamps respectively — these are *measured* durations, far more
    accurate than the v1 estimate_durations() output.
    """
    # Build a flat list of words *per section* by re-tokenising each
    # section's text in order — the word_timings list mirrors that order.
    result: dict[str, tuple[float, float, list[WordTiming]]] = {}
    cursor = 0
    for section in sections:
        section_tokens = tokenize_script(section.text)
        n = len(section_tokens)
        if n == 0 or cursor >= len(word_timings):
            continue
        slice_ = word_timings[cursor:cursor + n]
        cursor += n
        if slice_:
            result[section.section_id] = (
                slice_[0].start,
                slice_[-1].end,
                slice_,
            )
    return result
