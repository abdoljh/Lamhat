"""
Kraken OCR engine for Arabic scanned PDFs.

Uses the OpenITI apt-20221130 Arabic printed-text model.
Extracted from the tested OCR-me / Upgrade streamlit_app.py solution.

Requires Python 3.12 and the following packages:
    torch>=2.4.0,<=2.10.0
    lightning @ ./lightning-compat
    kraken==7.0.1

On Python 3.14 or when Kraken is not installed, import will succeed but
load_model() will raise KrakenNotAvailableError.

Public API
----------
    KrakenNotAvailableError
    ensure_model() -> str
    load_model(model_path=None) -> model
    binarize_page(pil_img, threshold=0.5) -> PIL.Image
    ocr_page(model, img, *, text_direction, autocast, pad, bidi_key,
             no_legacy_polygons, temperature) -> tuple[str, list[float]]
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── Availability check ───────────────────────────────────────────────────── #

try:
    import kraken  # noqa: F401
    _KRAKEN_AVAILABLE = True
except ImportError:
    _KRAKEN_AVAILABLE = False


class KrakenNotAvailableError(RuntimeError):
    """Raised when kraken/torch are not installed (requires Python 3.12)."""
    def __init__(self) -> None:
        super().__init__(
            "Kraken OCR is not installed. "
            "It requires Python 3.12 — redeploy the Streamlit Cloud app with "
            "Python 3.12 selected in Advanced settings, then uncomment "
            "torch/lightning/kraken in requirements.txt."
        )

# ── Model constants ──────────────────────────────────────────────────────── #

_MODEL_URL = (
    "https://raw.githubusercontent.com/OpenITI/AOCP_print_models"
    "/refs/heads/main/transcription/apt-20221130.mlmodel"
)
_MODEL_PATH = os.path.expanduser("~/.kraken_models/apt-20221130.mlmodel")

_BIDI_TO_RPRED: dict[str, object] = {
    "auto": True,
    "R": "R",
    "L": "L",
    "off": False,
}


# ── Model loading ─────────────────────────────────────────────────────────── #

def ensure_model(model_path: Optional[str] = None) -> str:
    """Download the Arabic model if not cached. Returns the local model path."""
    path = model_path or _MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


def load_model(model_path: Optional[str] = None):
    """Load and return the Kraken model (downloads on first call)."""
    if not _KRAKEN_AVAILABLE:
        raise KrakenNotAvailableError()
    from kraken.lib import models as kraken_models
    path = ensure_model(model_path)
    return kraken_models.load_any(path)


# ── Binarization (scipy NLBin port) ──────────────────────────────────────── #

def binarize_page(pil_img: Image.Image, threshold: float = 0.5) -> Image.Image:
    """Non-linear binarization of a PIL Image. Returns a binary PIL Image."""
    try:
        from scipy.ndimage import (
            uniform_filter,
            maximum_filter,
            minimum_filter,
        )
        _use_scipy = True
    except ImportError:
        _use_scipy = False

    gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0

    if _use_scipy:
        # NLBin: estimate local background and threshold adaptively
        scale = max(gray.shape) / 600.0
        w = max(3, int(round(0.3 * scale)))
        # local average
        blur = uniform_filter(gray, w * 2 + 1)
        # local contrast range
        hi = maximum_filter(gray, w * 2 + 1)
        lo = minimum_filter(gray, w * 2 + 1)
        contrast = hi - lo
        bg = np.where(contrast > 0.05, blur, gray)
        binary = gray < bg * threshold
    else:
        # Simple global threshold fallback
        binary = gray < threshold

    out = Image.fromarray((binary.astype(np.uint8)) * 255, mode="L")
    return out


# ── OCR ───────────────────────────────────────────────────────────────────── #

def ocr_page(
    model,
    pil_img: Image.Image,
    *,
    text_direction: str = "horizontal-rl",
    autocast: bool = False,
    pad: int = 16,
    bidi_key: str = "auto",
    no_legacy_polygons: bool = False,
    temperature: float = 1.0,
) -> tuple[str, list[float]]:
    """Run Kraken OCR on a PIL Image.

    Parameters
    ----------
    model           Loaded Kraken model (from load_model()).
    pil_img         Input image (already binarized is ideal; RGB accepted).
    text_direction  ``"horizontal-rl"`` for Arabic RTL text.
    autocast        Enable fp16 autocast (faster on GPU, no accuracy change).
    pad             Line padding in pixels.
    bidi_key        One of ``"auto"``, ``"R"``, ``"L"``, ``"off"``.
    no_legacy_polygons  Force the new polygon extractor.
    temperature     Softmax temperature — only affects confidence scores.

    Returns
    -------
    (text, per_line_confidences)
    """
    from kraken import blla, rpred

    img_rgb = pil_img.convert("RGB")

    # Temperature affects confidence scores only (not greedy decoding).
    if hasattr(model, "temperature"):
        model.temperature = temperature

    bidi = _BIDI_TO_RPRED.get(bidi_key, True)

    seg = blla.segment(
        img_rgb,
        text_direction=text_direction,
        no_legacy_polygons=no_legacy_polygons,
    )
    predictions = rpred.rpred(
        model,
        img_rgb,
        seg,
        bidi_reorder=bidi,
        pad=pad,
        autocast=autocast,
    )

    lines: list[str] = []
    confs: list[float] = []
    for line in predictions:
        lines.append(line.prediction)
        if line.confidences:
            confs.append(float(sum(line.confidences) / len(line.confidences)))

    return "\n".join(lines), confs
