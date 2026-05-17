#!/usr/bin/env python3
"""
Verify Amiri font discovery without rendering a full video.

Drop this next to the patched typography.py and run:
    python verify_font_discovery.py

Prints which discovery strategy succeeded, all five weight paths, and
exits 0 on success / 1 on failure.  Quick smoke-test for any new
environment (fresh Colab runtime, Streamlit Cloud cold start, local dev
machine) before committing to a full render.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s  %(message)s",
)

try:
    from phase3.typography import FONT_PATHS
except RuntimeError as exc:
    print(f"\n✗ Discovery failed:\n  {exc}", file=sys.stderr)
    sys.exit(1)

print("\nResolved Amiri weights:")
for weight in ("regular", "bold", "italic", "bold_italic", "quran"):
    path = FONT_PATHS.get(weight)
    marker = "✓" if path else "·"
    label = path or "(not resolved — _font() will fall back to regular)"
    print(f"  {marker} {weight:14s}  {label}")

# Smoke test: render a tiny pull quote so Pillow actually loads the font
try:
    from phase3.typography import render, TypographySpec
    out = Path("/tmp/lamahat_font_smoke.png")
    render(
        TypographySpec(
            template="pull_quote",
            text="اللغة العربية تختبر التشكيل",
            width=1280, height=720,
        ),
        out,
    )
    print(f"\n✓ End-to-end render succeeded → {out} ({out.stat().st_size} bytes)")
except Exception as exc:
    print(f"\n✗ End-to-end render failed: {exc}", file=sys.stderr)
    sys.exit(1)
