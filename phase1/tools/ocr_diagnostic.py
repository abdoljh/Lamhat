#!/usr/bin/env python3
"""
OCR diagnostic — find the exact Tesseract configuration that captures
edge lines (e.g. نجدة فتحى صفوة) on scanned Arabic pages.

Run:
    python phase1/tools/ocr_diagnostic.py samples/pages_5_9.pdf

Prints the first 300 chars of page-1 output for each approach so the
user can see immediately which approach captures the top attribution line.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path


def _highlight(text: str, needle: str = "نجدة") -> str:
    return f"*** FOUND ***  {text}" if needle in text else text


def run(pdf_path: str) -> None:
    try:
        from pdf2image import convert_from_path
    except ImportError:
        sys.exit("pdf2image not installed.  Run: pip install pdf2image")
    try:
        import pytesseract
    except ImportError:
        sys.exit("pytesseract not installed.  Run: pip install pytesseract")
    try:
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError:
        sys.exit("Pillow / numpy not installed.")

    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    print(f"PDF: {pdf_path}\n")

    pages_pil = convert_from_path(pdf_path, dpi=300)
    page = pages_pil[0]

    # ── Approaches ──────────────────────────────────────────────────────── #
    approaches: list[tuple[str, object, str]] = []

    # 1. PIL Image directly (no numpy) — the simplest approach
    approaches.append(("PIL Image direct", page, ""))

    # 2. numpy from PIL (exact notebook approach)
    arr_rgb = np.array(page)
    approaches.append(("numpy RGB (notebook)", arr_rgb, ""))

    # 3. numpy + explicit --psm 3
    approaches.append(("numpy --psm 3", arr_rgb, "--psm 3"))

    # 4. numpy + --psm 6 (uniform block of text — sometimes better for dense pages)
    approaches.append(("numpy --psm 6", arr_rgb, "--psm 6"))

    # 5. Grayscale numpy
    arr_gray = np.array(page.convert("L"))
    approaches.append(("numpy grayscale", arr_gray, ""))

    # 6. Grayscale + contrast boost
    enhanced = ImageEnhance.Contrast(page.convert("L")).enhance(2.0)
    arr_contrast = np.array(enhanced)
    approaches.append(("grayscale + contrast 2x", arr_contrast, ""))

    # 7. Otsu binarisation via PIL
    gray_arr = np.array(page.convert("L"))
    thresh = np.percentile(gray_arr, 40)  # rough Otsu approximation
    binary_arr = ((gray_arr > thresh) * 255).astype(np.uint8)
    approaches.append(("PIL binarised (40th pct)", binary_arr, ""))

    # 8. PNG round-trip then numpy (how the pipeline works)
    buf = io.BytesIO()
    page.save(buf, format="PNG")
    reloaded = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    arr_reloaded = np.array(reloaded)
    approaches.append(("PNG round-trip numpy", arr_reloaded, ""))

    # 9. --oem 1 (LSTM only, sometimes better for Arabic)
    approaches.append(("numpy --oem 1", arr_rgb, "--oem 1"))

    # 10. --oem 0 (Legacy engine, Tesseract 4 only)
    approaches.append(("numpy --oem 0", arr_rgb, "--oem 0"))

    # ── Run each ────────────────────────────────────────────────────────── #
    results: list[tuple[str, str, bool]] = []
    for label, img, cfg in approaches:
        try:
            text = pytesseract.image_to_string(img, lang="ara", config=cfg)
            found = "نجدة" in text
            results.append((label, text, found))
        except Exception as exc:
            results.append((label, f"ERROR: {exc}", False))

    # ── Report ──────────────────────────────────────────────────────────── #
    print("=" * 70)
    print(f"{'Approach':<35} {'Found?':<8} First 250 chars of output")
    print("=" * 70)
    for label, text, found in results:
        marker = "✓ YES" if found else "✗ no"
        preview = text[:250].replace("\n", " ").strip()
        print(f"{label:<35} {marker:<8} {preview}")
        print()

    winners = [lbl for lbl, _, found in results if found]
    if winners:
        print("\n>>> WINNING approaches:", winners)
    else:
        print("\n>>> NO approach detected the line on this machine.")
        print("    → The issue is the local Tesseract tessdata.")
        print("    → Fix: enable ocr_correction=True in Phase1Config (uses Claude Haiku)")
        print("    → Or: install tessdata_best: https://github.com/tesseract-ocr/tessdata_best")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "samples/pages_5_9.pdf"
    run(pdf)
