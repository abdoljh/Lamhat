"""
Phase 1 — ArabicTextNormalizer
Post-extraction text normalisation.

Applies:
  1. fix_article()  — word-level fallback for residual lam-alef article errors.
  2. Scanned-only:  arabic-reshaper + python-bidi (OCR output only).
  3. Noise cleaning — lone page numbers, zero-width chars, excessive whitespace.

Primary lam-alef fix (Hex-Placeholder Technique) — now in ingestor.py
──────────────────────────────────────────────────────────────────────
The main fix for lam-alef ligature inversions is applied at the span level
inside PDFIngestor._extract_rtl_text(), before span-level RTL sorting.
For each span whose x-coordinates indicate visual (left→right) character
order, _fix_lamalef_visual_span() is called:
  1. ل+alef-variant pairs → Presentation Form single code points (U+FEF5–FEFB)
  2. Reverse the span string
  3. NFKD-decompose back to logical ل+alef order
This correctly handles all occurrences within a word, including word-internal
lam-alef pairs (e.g. إعالمية → إعلامية) that the word-level rules below miss.

fix_article rules (fallback)
─────────────────────────────
Catches residual cases not covered by the span-level fix — mainly PDFs whose
fonts use non-visual encoding with incorrect ToUnicode table mappings:
  امل   instead of  الم   (plain alef + consonant + lam → swap 1 and 2)
  اآلن  instead of  الآن  (alef + madda-alef + lam → swap 1 and 2)
  ألدوات instead of الأدوات (hamza-alef + lam + consonant → insert plain alef)
  ألي    instead of لأي    (short hamza-alef + lam → swap)
"""

from __future__ import annotations

import re
import unicodedata
import logging
from typing import Literal

logger = logging.getLogger(__name__)

Source = Literal["digital", "scanned"]

# ── Arabic character constants ──────────────────────────────────────── #
_ALEF     = '\u0627'   # ا
_ALEF_HA  = '\u0623'   # أ
_ALEF_HB  = '\u0625'   # إ
_ALEF_MA  = '\u0622'   # آ
_LAM      = '\u0644'   # ل
_ALL_ALEF = {_ALEF, _ALEF_HA, _ALEF_HB, _ALEF_MA}

# Mappings: (dotless-base | symbol-dot) bigrams → standard Arabic letter.
# Some PDF fonts encode Arabic letters as a dotless glyph + a separate dot
# symbol glyph.  PyMuPDF extracts these as two code points.  The dot may
# precede OR follow the base glyph depending on the font vendor.
#
# Dotless Beh (U+066E ٮ) — base shared by ب ت ث ن ي:
#   ﮳ (FBB3 dot-below)       + ٮ  →  ب  U+0628
#   ﮵ (FBB5 two-dots-below)  + ٮ  →  ي  U+064A
#   ﮲ (FBB2 dot-above)       + ٮ  →  ن  U+0646
#   ﮴ (FBB4 two-dots-above)  + ٮ  →  ت  U+062A
#   ﮶ (FBB6 three-dots-above)+ ٮ  →  ث  U+062B
# Dotless Feh (U+06A1 ڡ) — base shared by ف ق:
#   ﮲ (FBB2 dot-above)       + ڡ  →  ف  U+0641
#   ﮴ (FBB4 two-dots-above)  + ڡ  →  ق  U+0642
_DECOMPOSED_MAP: list[tuple[str, str]] = [
    # ---- Dotless Feh pairs ----
    ('\uFBB2\u06A1', '\u0641'),   # ﮲ + ڡ → ف
    ('\u06A1\uFBB2', '\u0641'),   # ڡ + ﮲ → ف
    ('\uFBB4\u06A1', '\u0642'),   # ﮴ + ڡ → ق
    ('\u06A1\uFBB4', '\u0642'),   # ڡ + ﮴ → ق
    # ---- Dotless Beh pairs ----
    ('\uFBB3\u066E', '\u0628'),   # ﮳ + ٮ → ب
    ('\u066E\uFBB3', '\u0628'),   # ٮ + ﮳ → ب
    ('\uFBB5\u066E', '\u064A'),   # ﮵ + ٮ → ي
    ('\u066E\uFBB5', '\u064A'),   # ٮ + ﮵ → ي
    ('\uFBB2\u066E', '\u0646'),   # ﮲ + ٮ → ن
    ('\u066E\uFBB2', '\u0646'),   # ٮ + ﮲ → ن
    ('\uFBB4\u066E', '\u062A'),   # ﮴ + ٮ → ت
    ('\u066E\uFBB4', '\u062A'),   # ٮ + ﮴ → ت
    ('\uFBB6\u066E', '\u062B'),   # ﮶ + ٮ → ث
    ('\u066E\uFBB6', '\u062B'),   # ٮ + ﮶ → ث
    # ---- Haa (ح) family: jeem (ج) and khaa (خ) ----
    ('\uFBB3\u062D', '\u062C'),   # ﮳ + ح → ج
    ('\u062D\uFBB3', '\u062C'),   # ح + ﮳ → ج
    ('\uFBB2\u062D', '\u062E'),   # ﮲ + ح → خ
    ('\u062D\uFBB2', '\u062E'),   # ح + ﮲ → خ
    # ---- Ain (ع) family: ghain (غ) ----
    ('\uFBB2\u0639', '\u063A'),   # ﮲ + ع → غ
    ('\u0639\uFBB2', '\u063A'),   # ع + ﮲ → غ
    # ---- Tah (ط) family: zah (ظ) ----
    ('\uFBB2\u0637', '\u0638'),   # ﮲ + ط → ظ
    ('\u0637\uFBB2', '\u0638'),   # ط + ﮲ → ظ
]
# Symbol-dot code points to strip after bigram mapping (FBB2–FBB6)
_SYMBOL_DOTS = ''.join(chr(cp) for cp in range(0xFBB2, 0xFBB7))


def _fix_decomposed_arabic(text: str) -> str:
    """
    Reconstruct Arabic letters that a PDF font split into dotless-base + dot-symbol.

    Applied before NFC so the corrected code points can be canonically composed.
    Safe to call on both digital and scanned paths — a no-op when the special
    code points are absent.
    """
    for src, tgt in _DECOMPOSED_MAP:
        text = text.replace(src, tgt)
    # Lone dotless bases that had no paired dot: map to the most common
    # representative letter for that base shape.
    text = text.replace('\u06A1', '\u0641')   # ڡ → ف (lone dotless feh)
    text = text.replace('\u066E', '\u0628')   # ٮ → ب (lone dotless beh)
    # Strip any remaining symbol-dot characters (FBB2–FBB6)
    for ch in _SYMBOL_DOTS:
        text = text.replace(ch, '')
    return text


def fix_article(word: str) -> str:
    """
    Fix lam-alef article encoding errors from Arabic PDF font ToUnicode tables.

    Rule B — word starts with [hamza/madda-alef][lam]:
      len==2:                    swap  (standalone أل → لأ)
      len==3 + إ at pos 0:       leave alone  (إلى, إلا are prepositions)
      len==3 + أ/آ at pos 0:     swap  (ألي → لأي)
      len≥4 + alef after lam:    insert plain alef  (اإل → الإ)
      len≥4 + consonant after:   insert plain alef  (ألدوات → الأدوات)

    Rule A — word starts with [ا][non-lam][ل] at positions 0,1,2 → swap 1 and 2:
      Fixes: امل→الم, اآلن→الآن, اإلنترنت→الإنترنت
      Restricted to word-START only (positions 0-2) to avoid corrupting
      genuine Arabic roots like كامل, عامل that contain ا+م+ل internally.

    Standalone "ال" → "لا"  (negation/emphasis particle).

    Rule C — single-char connector/preposition prefix + article error:
      If none of the above rules apply and the word begins with one of
      و ب ل ك ف س, recursively apply fix_article to the remainder.
      Fixes: واإلعالمية→والإعالمية, بالأسباب stays correct, etc.

    Note: word-internal لا inversion (e.g. إعالمية vs إعلامية) is now
    primarily handled at span level in ingestor.py via the Hex-Placeholder
    Technique with per-character x-coordinate comparison.
    """
    if len(word) < 2:
        return word
    c = list(word)

    # Rule B
    if c[0] in (_ALEF_HA, _ALEF_HB, _ALEF_MA) and c[1] == _LAM:
        after_lam = c[2] if len(c) > 2 else None
        if len(word) == 2:
            c[0], c[1] = c[1], c[0]                      # standalone → swap
        elif len(word) == 3:
            if c[0] != _ALEF_HB:                          # إ = preposition → leave
                c[0], c[1] = c[1], c[0]                  # أ/آ → swap (ألي → لأي)
        else:
            c = [_ALEF, _LAM] + c[0:1] + c[2:]           # long → insert plain alef

    # Rule A — word-start ONLY (positions 0,1,2)
    if len(c) >= 3 and c[0] == _ALEF and c[1] != _LAM and c[2] == _LAM:
        c[1], c[2] = c[2], c[1]

    # Standalone ال → لا
    if len(c) == 2 and c[0] == _ALEF and c[1] == _LAM:
        c[0], c[1] = c[1], c[0]

    result = ''.join(c)

    # Rule C — single-char Arabic connector/preposition prefix + article pattern.
    # e.g. "واإلعالمية" → و + fix_article("اإلعالمية") → "والإعالمية"
    #      "بالأسباب"  → ب + fix_article("الأسباب")   → "بالأسباب"
    # Only recurses when no other rule already changed the word (avoids
    # double-application) and the word is long enough to contain a real article.
    _CONNECTORS = frozenset('\u0648\u0628\u0644\u0643\u0641\u0633')  # و ب ل ك ف س
    if result == word and len(word) >= 4 and word[0] in _CONNECTORS:
        inner = fix_article(word[1:])
        if inner != word[1:]:
            return word[0] + inner

    return result


class ArabicTextNormalizer:
    """
    Source-aware Arabic text normaliser.

    digital: fix_article per word + NFC + noise clean
    scanned: fix_article per word + NFC + reshape + bidi + noise clean
    """

    _NOISE_PATTERNS = [
        re.compile(r"^\s*\d+\s*$", re.MULTILINE),          # lone page numbers
        re.compile(r"[\u200b\u200c\u200d\ufeff]"),          # zero-width chars
        re.compile(r"[ \t]{3,}", re.MULTILINE),             # excessive spaces
        re.compile(r"\n{4,}", re.MULTILINE),                # excessive blank lines
    ]

    # Running header pattern: a short line (≤ 60 chars) that contains a
    # digit — typical of "Book Title  17" or "17  Chapter Name" headers
    # that Tesseract picks up at the top of each scanned page.
    _HEADER_PATTERN = re.compile(
        r"^[^\n]{1,60}\d[^\n]{0,30}\n",
        re.MULTILINE,
    )

    # Footnote pattern: lines starting with (n) or * followed by content,
    # OR a horizontal rule (─────) followed by footnote text.
    _FOOTNOTE_PATTERN = re.compile(
        r"(?:^[─━═\-]{4,}.*$\n?(?:^.+\n?)*)"   # separator + following lines
        r"|(?:^\s*[\(\[（【]\d+[\)\]）】][^\n]*\n?)",  # (1) ... footnote lines
        re.MULTILINE,
    )

    # Matches contiguous Arabic / Arabic-Indic characters in a line.
    _ARABIC_WORD_RE = re.compile(r'[\u0600-\u06FF]+')

    def normalize(self, text: str, source: Source = "digital") -> str:
        if not text or not text.strip():
            return ""

        if source == "scanned":
            text = self._strip_page_furniture(text)
            if not text.strip():
                return ""
            text = self._join_ocr_lines(text)
            # Tesseract frequently misreads Arabic comma ، (U+060C) as » (U+00BB).
            # Replace » with ، when surrounded by Arabic text (mid-sentence position).
            text = re.sub(
                r'(?<=[\u0600-\u06FF\u064B-\u065F\u0660-\u0669])»(?=\s)',
                '،', text,
            )
        else:
            # Digital PDFs with decomposed font encoding sometimes emit a space
            # between a word fragment and the following dotless-base glyph (ٮ/ڡ)
            # because the glyphs render at visually separated positions.
            # Remove those intra-word spaces before bigram mapping.
            # ح/ع/ط are NOT included — they are standard Arabic letters that
            # legitimately begin words.  FBB2–FBB6 are also excluded because
            # they can appear as the first glyph of a word in this encoding.
            # OCR word boundaries come from Tesseract's visual analysis and are
            # reliable, so this step is skipped for the scanned path.
            text = re.sub(
                r'([\u0600-\u06FF\uFBB2-\uFBB6]) +([\u066E\u06A1])',
                r'\1\2', text,
            )

        # Reconstruct letters decomposed by unusual PDF font encoding
        # (dotless base glyph + Arabic Symbol Dot → standard letter).
        # Applied before NFC because NFC cannot compose these combinations.
        text = _fix_decomposed_arabic(text)

        text = unicodedata.normalize("NFC", text)

        # Normalise non-standard Arabic letter variants to their MSA forms.
        # U+06BE Heh Doachashmee (ھ) → U+0647 Arabic Heh (ه)
        # U+06CC Farsi Yeh (ی)        → U+064A Arabic Yeh (ي)
        # U+0649 Alef Maqsura (ى)     kept as-is (valid MSA word-final form)
        # These variants appear in OCR output from Urdu/Persian-influenced fonts
        # and cause Mishkal and other Arabic NLP tools to silently drop the text.
        text = text.replace('\u06BE', '\u0647')   # ھ → ه
        text = text.replace('\u06CC', '\u064A')   # ی → ي

        # Apply article fix word-by-word
        text = " ".join(fix_article(w) for w in text.split(" "))

        # Fix word-internal double-alef-before-lam: اال → الا
        # This artifact arises when a lam-alef ligature's two code points share
        # the same PDF origin x, so x-DESC sort leaves them adjacent rather than
        # interleaving the ل between them (e.g. المجاالت → المجالات).
        # ا+ا+ل is not a valid sequence in any Arabic word, so this is safe.
        text = " ".join(
            w.replace('\u0627\u0627\u0644', '\u0627\u0644\u0627')
            for w in text.split(" ")
        )

        # NOTE: reshape + bidi (arabic-reshaper / python-bidi) are NOT applied
        # to scanned text.  OCR engines (Tesseract, EasyOCR, PaddleOCR) already
        # output clean logical-order Unicode (U+0600–U+06FF).  Applying reshape
        # + get_display() would convert that to visual Presentation Forms
        # (U+FE70–FEFF), which is correct for image/PDF rendering but wrong for
        # text files and downstream NLP/LLM processing.

        return self._clean(text).strip()

    @staticmethod
    def _detect_header_words(pages) -> frozenset:
        """
        Pre-pass: inspect the first non-empty line of each scanned page.
        Any Arabic word that appears in those first-lines across 2+ distinct
        pages is a "running header word".  This catches digit-free headers
        whose page number was garbled by OCR into Arabic letters.
        """
        _WORD = re.compile(r'[\u0600-\u06FF]+')
        word_pages: dict[str, set] = {}
        for idx, p in enumerate(pages):
            if getattr(p, "pdf_type", "scanned") != "scanned":
                continue
            for line in p.raw_text.splitlines():
                s = line.strip()
                if not s:
                    continue
                if len(s) <= 60:
                    for w in _WORD.findall(s):
                        word_pages.setdefault(w, set()).add(idx)
                break  # first non-empty line only
        return frozenset(w for w, pg in word_pages.items() if len(pg) >= 2)

    def normalize_pages(self, pages: list) -> list:
        # Pre-pass: collect running-header words across all scanned pages.
        # Words that appear in short (≤ 60 char) page-start lines on 2+
        # pages are almost certainly part of the book's running header
        # (title + page-number), even when OCR drops or corrupts the digits.
        self._hdr_words: frozenset[str] = self._detect_header_words(pages)
        for page in pages:
            source: Source = "scanned" if page.pdf_type == "scanned" else "digital"
            before = len(page.raw_text)
            page.raw_text = self.normalize(page.raw_text, source=source)
            after = len(page.raw_text)
            logger.debug("Page %d [%s] normalised: %d → %d chars",
                         page.page_number, source, before, after)
            if after == 0 and before > 0:
                logger.warning("Page %d [%s] became EMPTY after normalisation (was %d chars). "
                                "Check for encoding issues in the source PDF.",
                                page.page_number, source, before)
        empty = sum(1 for p in pages if not p.raw_text.strip())
        if empty:
            logger.info("normalize_pages: %d/%d pages empty after normalisation.",
                        empty, len(pages))
        return pages

    def _strip_page_furniture(self, text: str) -> str:
        """
        Remove running headers and footnotes from scanned OCR output.

        Running headers: short lines containing a digit that appear at the
        very start of a page block (e.g. "مذكرات جعفر العسكري  17").
        We strip only the FIRST such line in each page section to avoid
        removing legitimate short sentences in the body.

        Footnotes: blocks starting with (n) or a horizontal separator line.
        """
        lines = text.splitlines()
        cleaned = []
        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            stripped = line.strip()
            # Skip standalone page numbers
            if re.match(r'^\d+$', stripped):
                continue
            # Drop running headers at the very start of each scanned page.
            # A line qualifies if it is the first non-empty line AND is short
            # (≤ 60 chars) AND either:
            #   (a) contains a digit (ASCII or Arabic-Indic), or
            #   (b) contains a word that cross-page frequency analysis marked
            #       as a running-header word (set by normalize_pages pre-pass).
            is_page_start = i == 0 or all(not lines[j].strip() for j in range(i))
            if is_page_start and len(stripped) <= 60:
                hdr_words = getattr(self, '_hdr_words', frozenset())
                has_digit = bool(re.search(r'[0-9\u0660-\u0669]', stripped))
                has_hdr_word = bool(hdr_words) and any(
                    w in hdr_words
                    for w in self._ARABIC_WORD_RE.findall(stripped)
                )
                if has_digit or has_hdr_word:
                    continue
            # Skip footnote separator lines
            if re.match(r'^[─━═\-─]{4,}', stripped):
                continue
            # Skip footnote content lines: (1) or [1] at line start
            if re.match(r'^[\(\[（【]\d+[\)\]）】]', stripped):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    _SENT_END = re.compile(r'[.؟!]\s*$')
    # Soft word cap per OCR paragraph.  Arabic prose rarely uses sentence
    # terminals between every clause, so without a cap a whole page can collapse
    # into one paragraph — a blob too large for the chunker to split further.
    # 250 words ≈ 350 tokens, well under max_tokens=1500, so the chunker can
    # still merge several paragraphs into a single chunk efficiently.
    _MAX_PARA_WORDS = 250

    @staticmethod
    def _join_ocr_lines(text: str) -> str:
        """
        Join OCR output lines that belong to the same paragraph.

        Tesseract emits one visual line per output line.  Lines that don't end
        with a sentence terminal (. ؟ !) are joined with a space to their
        successor.  Blank lines, sentence-terminal lines, and paragraphs that
        exceed _MAX_PARA_WORDS start a new paragraph.
        """
        lines = text.splitlines()
        paragraphs: list[str] = []
        buffer = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    paragraphs.append(buffer)
                    buffer = ""
                continue
            if buffer:
                at_limit = len(buffer.split()) >= ArabicTextNormalizer._MAX_PARA_WORDS
                if ArabicTextNormalizer._SENT_END.search(buffer) or at_limit:
                    paragraphs.append(buffer)
                    buffer = stripped
                else:
                    buffer += " " + stripped
            else:
                buffer = stripped
        if buffer:
            paragraphs.append(buffer)
        return "\n\n".join(paragraphs)

    def _clean(self, text: str) -> str:
        for pat in self._NOISE_PATTERNS:
            if pat.pattern == r"\n{4,}":
                text = pat.sub("\n\n\n", text)
            elif pat.pattern == r"[ \t]{3,}":
                text = pat.sub("  ", text)
            else:
                text = pat.sub("", text)
        # Remove space before period/full-stop
        text = re.sub(r'\s+\.(?=\s|$)', '.', text)
        # Join tanwin (ً ٌ ٍ) separated from its alef/alef-maqsura by a space
        # e.g. "خصوصً ا" → "خصوصًا"
        text = re.sub(r'([\u064B\u064C\u064D])\s+([\u0627\u0649])', r'\1\2', text)
        return text
