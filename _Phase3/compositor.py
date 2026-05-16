"""
Phase 3 — Background video compositor.

Turns per-section image sets (and optional Pexels fallback clips)
into a single silent background .mp4 with crossfade transitions
and an optional colour grade.

All heavy work is done by FFmpeg subprocesses; Python holds no large
buffers in memory, keeping RAM well below the 1 GB Streamlit Cloud limit.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from .effects import get_effect, ken_burns, probe_duration, trim_clip
from .parser import ScriptSection

log = logging.getLogger(__name__)

# ── Colour grade presets (FFmpeg curves filter) ───────────────────────── #
_GRADES: dict[str, str] = {
    # Warm amber — good for history / biography
    "warm": (
        "curves=r='0/0 0.5/0.56 1/1'"
        ":g='0/0 0.5/0.50 1/0.96'"
        ":b='0/0 0.5/0.38 1/0.82'"
    ),
    # Cool blue — good for science / philosophy
    "cool": (
        "curves=r='0/0 0.5/0.44 1/0.88'"
        ":g='0/0 0.5/0.50 1/0.96'"
        ":b='0/0 0.5/0.60 1/1'"
    ),
    # Neutral pass-through
    "neutral": "curves=all='0/0 1/1'",
}


# ── Section clip builder ─────────────────────────────────────────────────── #

def build_section_clip(
    section: ScriptSection,
    images: list[Path],
    fallback_clip: Path | None,
    section_duration: float,
    work_dir: Path,
    width: int,
    height: int,
) -> Path:
    """
    Build one video clip covering `section_duration` seconds for a single
    script section.

    Strategy (in priority order):
    1. Ken Burns effect on each downloaded image → concat mini-clips
    2. Trim / rescale a Pexels video clip to section_duration
    3. Black frame fallback (ensures pipeline never hard-fails)
    """
    out = work_dir / f"sec_{section.section_id}.mp4"

    # ── 1. Ken Burns on images ──────────────────────────────────────── #
    if images:
        per_img    = max(8.0, section_duration / len(images))
        mini_clips: list[Path] = []

        for idx, img_path in enumerate(images):
            mini_out = work_dir / f"mini_{section.section_id}_{idx:02d}.mp4"
            effect   = get_effect(idx)
            try:
                ken_burns(
                    img_path, mini_out,
                    duration=per_img,
                    width=width, height=height,
                    effect=effect,
                )
                mini_clips.append(mini_out)
                log.debug("Ken Burns ✓  %s img %d (%s)", section.section_id, idx, effect)
            except Exception as exc:
                log.warning("Ken Burns failed %s img %d: %s", section.section_id, idx, exc)

        if mini_clips:
            _concat_copy(mini_clips, out)
            return out

    # ── 2. Pexels fallback clip ─────────────────────────────────────── #
    if fallback_clip and fallback_clip.exists():
        try:
            trim_clip(fallback_clip, out,
                      duration=section_duration,
                      width=width, height=height)
            log.debug("Pexels fallback ✓  %s", section.section_id)
            return out
        except Exception as exc:
            log.warning("Pexels trim failed %s: %s", section.section_id, exc)

    # ── 3. Black frame fallback ─────────────────────────────────────── #
    log.warning("Using black frame for section %s", section.section_id)
    _black_clip(out, section_duration, width, height)
    return out


def _concat_copy(clips: list[Path], output: Path) -> None:
    """Concatenate clips that share identical codec/resolution using stream copy."""
    if len(clips) == 1:
        shutil.copy(clips[0], output)
        return

    list_file = output.parent / f"_list_{output.stem}.txt"
    list_file.write_text(
        "\n".join(f"file '{c.resolve()}'" for c in clips),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"concat_copy failed:\n{result.stderr[-800:]}")


def _black_clip(output: Path, duration: float, width: int, height: int) -> None:
    """
    Generate a silent dark-navy video clip (last-resort fallback).
    Dark navy (#1a1a2e) looks cinematic and lets white ASS text stand out clearly.
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x1a1a2e:s={width}x{height}:r=25",
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        str(output),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60)


# ── Cross-fade assembly ──────────────────────────────────────────────────── #

def concat_with_crossfade(
    clips: list[Path],
    output: Path,
    fade_duration: float = 1.0,
) -> Path:
    """
    Concatenate clips with a smooth `fade` crossfade between each pair.

    Uses FFmpeg's xfade filter chained for N clips.
    Falls back to plain stream-copy concat if xfade fails.
    """
    if len(clips) == 1:
        shutil.copy(clips[0], output)
        return output

    # Get clip durations for xfade offset calculation
    durations: list[float] = []
    for clip in clips:
        try:
            durations.append(probe_duration(clip))
        except Exception:
            durations.append(30.0)

    # Build -filter_complex chain:
    # [0:v][1:v]xfade=...:offset=d0-fade[v1];
    # [v1][2:v]xfade=...:offset=d0+d1-2*fade[v2]; ...
    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]

    filter_parts: list[str] = []
    cumulative = 0.0
    prev       = "0:v"

    for i in range(1, len(clips)):
        cumulative += durations[i - 1] - fade_duration
        is_last    = (i == len(clips) - 1)
        out_label  = "vout" if is_last else f"v{i}"
        filter_parts.append(
            f"[{prev}][{i}:v]"
            f"xfade=transition=fade:duration={fade_duration}"
            f":offset={max(0.0, cumulative):.3f}"
            f"[{out_label}]"
        )
        prev = out_label

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        log.warning("xfade concat failed, falling back to stream copy:\n%s",
                    result.stderr[-600:])
        _concat_copy(clips, output)
    return output


# ── Colour grade ─────────────────────────────────────────────────────────── #

def apply_color_grade(
    input_path: Path,
    output_path: Path,
    grade: str = "warm",
) -> Path:
    """Re-encode the assembled video with a colour-grade filter."""
    vf = _GRADES.get(grade, _GRADES["neutral"])
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Color grade failed:\n{result.stderr[-800:]}")
    return output_path


# ── Thumbnail extractor ──────────────────────────────────────────────────── #

def extract_thumbnail(video_path: Path, output_path: Path, time: float = 2.0) -> Path:
    """Extract a single JPEG frame from the video for UI preview."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(time),
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "3",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return output_path


# ── Final video muxer ────────────────────────────────────────────────────── #

def mux_final_video(
    background_video: Path,
    output_path: Path,
    audio_path: Path | None = None,
    subtitle_file: Path | None = None,
    max_duration: float | None = None,
) -> Path:
    """
    Combine a silent background video with optional audio and ASS subtitles
    into a finished MP4.

    The function does everything in a single FFmpeg pass:
      - Video: re-encoded at crf=22 (better quality than intermediate 26).
      - Audio: AAC 192 kbps if audio_path is provided; otherwise silent.
      - Subtitles: burned-in via libass if subtitle_file is provided.
      - Duration: hard-capped at max_duration seconds (-t flag) so the
        output is never longer than the audio, even if the background video
        was assembled for a slightly longer duration.

    Parameters
    ----------
    background_video   Silent .mp4 produced by assemble_background_video().
    output_path        Where to write the finished video.
    audio_path         MP3/AAC audio file (Phase 2 TTS output).  Optional.
    subtitle_file      ASS subtitle file from subtitler.py.  Optional.
    max_duration       Hard output duration cap in seconds.  Typically the
                       resolved audio duration.  None = no cap.
    """
    inputs: list[str] = ["-i", str(background_video)]
    if audio_path and audio_path.exists():
        inputs += ["-i", str(audio_path)]

    # Build -vf chain: subtitle burn (if requested)
    vf_parts: list[str] = []
    if subtitle_file and subtitle_file.exists():
        # Escape colons and backslashes in path for FFmpeg filtergraph
        safe_path = str(subtitle_file).replace("\\", "/").replace(":", "\\:")
        vf_parts.append(f"ass={safe_path}")

    cmd = ["ffmpeg", "-y", *inputs]

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    else:
        cmd += ["-c:v", "copy"]   # no re-encode if no subtitle burn needed

    if vf_parts:
        # Re-encode video for subtitle burn
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p"]

    if audio_path and audio_path.exists():
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-an"]

    # Hard duration cap: prevents silent tail when background > audio
    if max_duration and max_duration > 0:
        cmd += ["-t", f"{max_duration:.3f}"]

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        raise RuntimeError(
            f"mux_final_video failed:\n{result.stderr[-1200:]}"
        )
    return output_path


# ── Top-level assembler ──────────────────────────────────────────────────── #

def assemble_background_video(
    sections: list[ScriptSection],
    section_durations: list[float],
    images_per_section: dict[str, list[Path]],
    clips_per_section: dict[str, Path | None],
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    color_grade: str = "warm",
    on_progress: Callable[[str, float], None] | None = None,
) -> Path:
    """
    Full assembly pipeline:
      Per-section clips (Ken Burns / Pexels) → crossfade concat → colour grade.

    Parameters
    ----------
    sections             Parsed script sections (from parser.py).
    section_durations    Duration in seconds for each section.
    images_per_section   {section_id: [local image paths]} from Wikimedia.
    clips_per_section    {section_id: local clip path | None} from Pexels.
    output_path          Final output .mp4 (silent background video).
    width / height       Target resolution (default 1280×720).
    color_grade          'warm' | 'cool' | 'neutral'.
    on_progress          Optional callback(step_label, fraction_0_to_1).
    """
    n_sections   = len(sections)
    total_steps  = n_sections + 2   # +2: concat + grade
    done         = 0

    def _prog(label: str) -> None:
        nonlocal done
        done += 1
        if on_progress:
            on_progress(label, done / total_steps)

    with tempfile.TemporaryDirectory(prefix="bk2v_comp_") as tmp:
        work_dir = Path(tmp)
        section_clips: list[Path] = []

        for section, dur in zip(sections, section_durations):
            imgs     = images_per_section.get(section.section_id, [])
            fallback = clips_per_section.get(section.section_id)

            clip = build_section_clip(
                section=section,
                images=imgs,
                fallback_clip=fallback,
                section_duration=dur,
                work_dir=work_dir,
                width=width,
                height=height,
            )
            section_clips.append(clip)
            _prog(f"Built clip · {section.section_id}  ({dur:.0f}s)")

        # Concat with crossfade
        concat_out = work_dir / "concat.mp4"
        concat_with_crossfade(section_clips, concat_out)
        _prog("Crossfade concat")

        # Colour grade → final file
        apply_color_grade(concat_out, output_path, grade=color_grade)
        _prog("Colour grade")

    return output_path
