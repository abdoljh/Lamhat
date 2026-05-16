"""
Phase 3 — FFmpeg effects for still images and video clips.

Ken Burns effect: slow zoom-in/out + pan applied to a still image,
producing a video clip that looks like documentary footage.

All FFmpeg calls are subprocess-based to minimise Python memory usage
on Streamlit Cloud (1 GB RAM limit).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# Cycling effect order applied to successive images
_EFFECTS = ["zoom_in", "zoom_out", "pan_right", "pan_left"]


def get_effect(index: int) -> str:
    """Return a deterministic Ken Burns effect name for image index `index`."""
    return _EFFECTS[index % len(_EFFECTS)]


def probe_duration(path: Path) -> float:
    """Return the duration in seconds of a video or audio file via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}:\n{result.stderr[-500:]}")
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def ken_burns(
    image_path: Path,
    output_path: Path,
    duration: float,
    width: int = 1280,
    height: int = 720,
    effect: str = "zoom_in",
    fps: int = 25,
) -> Path:
    """
    Render a Ken Burns motion clip from a still image.

    The source image is scaled to 2× the output resolution before
    the zoompan filter runs, giving the filter room to zoom and pan
    without upscaling artefacts.

    effect values
    -------------
    zoom_in   — slow zoom into the centre
    zoom_out  — slow zoom out from centre
    pan_right — slow rightward pan at 1.3× zoom
    pan_left  — slow leftward pan at 1.3× zoom
    """
    n_frames  = max(1, int(duration * fps))
    scale_w   = width  * 2
    scale_h   = height * 2

    # ── Zoom / pan expressions ───────────────────────────────────────── #
    # Zoom step is sized so the total zoom change is always 0.5
    # (from 1.0 → 1.5 or from 1.5 → 1.0) regardless of clip duration.
    # This ensures the Ken Burns motion is clearly visible on both
    # 8-second and 60-second clips.
    zoom_step = f"{0.5 / n_frames:.6f}"
    # Pan step: traverse half the scaled frame width over the clip.
    pan_step  = max(1, int(scale_w * 0.5 / n_frames))

    if effect == "zoom_in":
        z = f"min(pzoom+{zoom_step},1.5)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif effect == "zoom_out":
        z = f"if(lte(on,1),1.5,max(1,pzoom-{zoom_step}))"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif effect == "pan_right":
        z = "1.3"
        x = f"if(lte(on,1),0,min(x+{pan_step},iw-iw/zoom))"
        y = "ih/2-(ih/zoom/2)"
    else:  # pan_left
        z = "1.3"
        x = f"if(lte(on,1),iw-iw/zoom,max(0,x-{pan_step}))"
        y = "ih/2-(ih/zoom/2)"

    vf = (
        # 1. Scale to 2× output (letterbox/pillarbox if needed → crop to fill)
        f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,"
        f"crop={scale_w}:{scale_h},"
        # 2. Ken Burns via zoompan
        f"zoompan=z='{z}':x='{x}':y='{y}'"
        f":d={n_frames}:s={width}x{height}:fps={fps},"
        # 3. Ensure correct pixel format for libx264
        "format=yuv420p"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(fps),
        "-i", str(image_path),
        "-vf", vf,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",   # fast intermediate encode
        "-crf", "26",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"Ken Burns failed on {image_path.name} ({effect}):\n"
            f"{result.stderr[-1200:]}"
        )
    return output_path


def trim_clip(
    input_path: Path,
    output_path: Path,
    duration: float,
    start: float = 0.0,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """
    Trim a video clip to `duration` seconds and rescale to width×height.
    Audio is stripped (background video only).
    """
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "26",
        "-an",                    # no audio
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"Trim failed on {input_path.name}:\n{result.stderr[-800:]}"
        )
    return output_path
