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
    binarize_page(pil_img, threshold=0.5) -> PIL.Image   (scipy fallback)
    ocr_page(model, img, *, text_direction, autocast, pad, bidi_key,
             no_legacy_polygons, temperature, threshold) -> tuple[str, list[float]]

Notes
-----
ocr_page() accepts the *original* (non-binarized) PIL Image.  It binarizes
internally with kraken's own nlbin (falling back to our scipy port), passes
the binarized image to blla.segment and the original RGB image to rpred —
exactly as the tested Upgrade/streamlit_app.py solution does.
"""

from __future__ import annotations

import inspect
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


def _sig_params(fn) -> set:
    """Return the set of parameter names accepted by fn."""
    try:
        return set(inspect.signature(fn).parameters)
    except Exception:
        return set()


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


# ── Binarization ─────────────────────────────────────────────────────────── #

def binarize_page(pil_img: Image.Image, threshold: float = 0.5) -> Image.Image:
    """Binarize using kraken's nlbin when available, scipy NLBin otherwise.

    Always returns a grayscale (mode 'L') PIL Image with pixels 0 or 255.
    """
    # Prefer kraken's own nlbin — it's what the pipeline was designed for.
    if _KRAKEN_AVAILABLE:
        try:
            from kraken import binarization as _kbin
            return _kbin.nlbin(pil_img, threshold=threshold)
        except Exception:
            pass

    # Scipy NLBin port fallback.
    try:
        from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
        gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
        scale = max(gray.shape) / 600.0
        w = max(3, int(round(0.3 * scale)))
        blur = uniform_filter(gray, w * 2 + 1)
        hi   = maximum_filter(gray,  w * 2 + 1)
        lo   = minimum_filter(gray,  w * 2 + 1)
        bg   = np.where((hi - lo) > 0.05, blur, gray)
        binary = gray < bg * threshold
        return Image.fromarray((binary.astype(np.uint8)) * 255, mode="L")
    except ImportError:
        pass

    # Simple global threshold last resort.
    gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    binary = gray < threshold
    return Image.fromarray((binary.astype(np.uint8)) * 255, mode="L")


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
    threshold: float = 0.5,
) -> tuple[str, list[float]]:
    """Run Kraken OCR on a PIL Image.

    Parameters
    ----------
    model           Loaded Kraken model (from load_model()).
    pil_img         Original page image (RGB or grayscale). Binarization is
                    handled internally — do NOT pre-binarize before calling.
    text_direction  ``"horizontal-rl"`` for Arabic RTL text.
    autocast        fp16 autocast hint passed to blla.segment if supported.
    pad             Line padding in pixels for rpred.
    bidi_key        One of ``"auto"``, ``"R"``, ``"L"``, ``"off"``.
    no_legacy_polygons  Force the new polygon extractor (if supported).
    temperature     Softmax temperature — only affects confidence scores.
    threshold       Binarization threshold (0–1) forwarded to binarize_page.

    Returns
    -------
    (text, per_line_confidences)
    """
    from kraken import blla, rpred

    # Original image for rpred (recognition network needs real pixel values).
    orig_rgb = pil_img.convert("RGB")

    # Binarized image for segmentation (as the Upgrade solution does).
    bw_img = binarize_page(pil_img, threshold=threshold)

    if hasattr(model, "temperature"):
        model.temperature = temperature

    bidi = _BIDI_TO_RPRED.get(bidi_key, True)

    # ── Segmentation ─────────────────────────────────────────────────────── #
    _seg_p = _sig_params(blla.segment)
    _seg_extra: dict = {}
    if "autocast" in _seg_p:
        _seg_extra["autocast"] = autocast
    if "no_legacy_polygons" in _seg_p and no_legacy_polygons:
        _seg_extra["no_legacy_polygons"] = True

    seg = blla.segment(
        bw_img,
        text_direction=text_direction,
        **_seg_extra,
    )

    # ── Recognition ──────────────────────────────────────────────────────── #
    _rpred_p = _sig_params(rpred.rpred)
    _bidi_kwarg = "bidi_reordering" if "bidi_reordering" in _rpred_p else "bidi_reorder"
    _rpred_extra: dict = {}
    if "autocast" in _rpred_p:
        _rpred_extra["autocast"] = autocast

    predictions = rpred.rpred(
        model,
        orig_rgb,       # original image — not binarized
        seg,
        **{_bidi_kwarg: bidi},
        pad=pad,
        **_rpred_extra,
    )

    lines: list[str] = []
    confs: list[float] = []
    for line in predictions:
        lines.append(line.prediction)
        line_confs = getattr(line, "confidences", None)
        if line_confs:
            confs.append(float(sum(line_confs) / len(line_confs)))

    return "\n".join(lines), confs
