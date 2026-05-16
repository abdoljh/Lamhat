"""
Smoke test for the typography module.

Renders every template once and asserts:
- The output file exists and is non-empty
- The output PNG opens, has the right resolution, and contains some
  non-background pixels (i.e. text was actually rendered)

Run with:
    python test_typography.py
"""

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from typography import TypographySpec, render
from PIL import Image


def test_template(template, text, subtitle="", width=1280, height=720):
    """Render and validate one template at 720p (faster than 1080p)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "card.png"
        spec = TypographySpec(template=template, text=text,
                              subtitle=subtitle, width=width, height=height)
        result = render(spec, out)

        assert result.exists(), f"{template}: output not created"
        size = result.stat().st_size
        assert size > 1024, f"{template}: suspiciously small file ({size} B)"

        img = Image.open(result)
        assert img.size == (width, height), \
            f"{template}: wrong size {img.size}"

        # Check some non-background pixels exist (text was rendered)
        # Cream background is ~240, dark text is ~30 — find pixels < 80
        gray = img.convert("L")
        dark_pixels = sum(1 for p in gray.getdata() if p < 80)
        assert dark_pixels > 100, \
            f"{template}: too few dark pixels ({dark_pixels}) — no text rendered?"

        print(f"  ✓ {template:<16} {img.size}  {dark_pixels:>6} text pixels  "
              f"{size//1024:>4} KB")


def main():
    print("\nTypography smoke test")
    print("─" * 70)

    test_template("title_card",
                  text="مذكرات جعفر العسكري",
                  subtitle="رواية لم يكتبها — بل عاشها")
    test_template("section_mark",
                  text="من الموصل إلى الاستانة")
    test_template("chapter_heading",
                  text="الخاتمة")
    test_template("pull_quote",
                  text="أشرس الأعداء قد يكونون من داخل صفوفك",
                  subtitle="من المذكرات")
    test_template("name_reveal",
                  text="جعفر العسكري",
                  subtitle="١٨٨٧ — ١٩٣٦")
    test_template("date_stamp",
                  text="١٩٠٨",
                  subtitle="الانقلاب")

    # Edge cases — empty input should not crash, just produce a blank-ish card
    print()
    print("Edge cases:")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "empty.png"
        render(TypographySpec(template="pull_quote", text="",
                              subtitle="should not crash"), out)
        assert out.exists()
        print(f"  ✓ empty pull_quote text: did not crash")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "no_subtitle.png"
        render(TypographySpec(template="title_card",
                              text="مذكرات جعفر العسكري"), out)
        assert out.exists()
        print(f"  ✓ title_card without subtitle: rendered")

    print()
    print("All typography smoke tests passed ✓")


if __name__ == "__main__":
    main()
