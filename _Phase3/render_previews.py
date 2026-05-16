"""Render one example of each template at 1920x1080 for visual review."""

import sys
sys.path.insert(0, '/home/claude/phase3_v3')

from pathlib import Path
from typography import TypographySpec, render

OUT = Path('/home/claude/phase3_v3/preview')
OUT.mkdir(exist_ok=True, parents=True)

# Real content from Abdol's al-Askari plan v2
EXAMPLES = [
    ("title_card",
     TypographySpec(template="title_card",
                    text="مذكرات جعفر العسكري",
                    subtitle="رواية لم يكتبها — بل عاشها",
                    width=1920, height=1080)),

    ("section_mark",
     TypographySpec(template="section_mark",
                    text="من الموصل إلى الاستانة",
                    subtitle="رحلة التحديث والطموح",
                    width=1920, height=1080)),

    ("section_mark_short",
     TypographySpec(template="section_mark",
                    text="الصراع بين الولاء والحلم",
                    width=1920, height=1080)),

    ("pull_quote_short",
     TypographySpec(template="pull_quote",
                    text="كيف يصنع العسكري ثورة؟",
                    width=1920, height=1080)),

    ("pull_quote_medium",
     TypographySpec(template="pull_quote",
                    text="أشرس الأعداء قد يكونون من داخل صفوفك",
                    width=1920, height=1080)),

    ("pull_quote_long",
     TypographySpec(template="pull_quote",
                    text="فهم حاضرنا لا يتحقق إلا بفهم عميق لماضينا الذي عاشه أمثال جعفر العسكري",
                    subtitle="من المذكرات",
                    width=1920, height=1080)),

    ("pull_quote_very_long",
     TypographySpec(template="pull_quote",
                    text="تاريخ الشرق الأوسط الحديث مكتوب جزئياً من خلال قرارات ضباط مثل جعفر العسكري الذين عاشوا أهم لحظات التاريخ",
                    width=1920, height=1080)),

    ("name_reveal",
     TypographySpec(template="name_reveal",
                    text="جعفر العسكري",
                    subtitle="١٨٨٧ — ١٩٣٦",
                    width=1920, height=1080)),

    ("name_reveal_long",
     TypographySpec(template="name_reveal",
                    text="محمود شوكت باشا",
                    subtitle="القائد العثماني العراقي",
                    width=1920, height=1080)),

    ("date_stamp",
     TypographySpec(template="date_stamp",
                    text="١٩٠٨",
                    subtitle="الانقلاب",
                    width=1920, height=1080)),
]

for name, spec in EXAMPLES:
    path = OUT / f"{name}.png"
    render(spec, path)
    print(f"  ✓ {name:<24} → {path}")

print(f"\nAll rendered to: {OUT}")
