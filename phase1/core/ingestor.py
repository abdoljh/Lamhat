"""
Phase 1 — PDFIngestor
Detects whether a PDF is digitally-born or scanned, then routes to the
appropriate extraction backend.

Arabic RTL extraction — span-level spatial sort with gap-based word joining
────────────────────────────────────────────────────────────────────────────
The PDF stores Arabic text with glyphs in visual left-to-right order.
The lam-alef article ligature is split across MULTIPLE SPANS by PyMuPDF,
with the article alef in one span and the lam+rest in a separate span.
These must be joined WITHOUT a space to form complete words.

Algorithm:
  1. Collect spans from rawdict with BOTH left-x (for RTL sort) and
     right-x (for gap detection).
  2. Group spans into visual lines by y-coordinate.
  3. Sort spans descending left-x → RTL reading order.
  4. Join consecutive spans: insert a space only when the visual gap
     between the right edge of span[i] and the left edge of span[i+1]
     exceeds WORD_GAP_PT. Otherwise join directly (same word).
  5. Merge diacritic-only spans to their adjacent word span.
  6. Fix comma positions and duplicate punctuation.
  7. Reconstruct paragraphs with heading detection.

Lam-alef ligature fix — Hex-Placeholder Technique
──────────────────────────────────────────────────
Arabic PDF generators often preserve the obligatory lam-alef ligatures
(ل+ا, ل+أ, ل+إ, ل+آ) in their *logical* order even inside a visual-order
glyph run.  When the span characters are reversed to recover logical reading
order, these pairs flip (لا → ال), corrupting the article ال and word-internal
alef vowels (e.g. إعلامية extracted as إعالمية).

Fix — applied per-span whenever x-coordinates confirm visual (left→right) order:
  1. Replace each ل+alef-variant pair with its single Presentation Form code
     point (U+FEF5–FEFB) so the pair is treated as ONE character during reversal.
  2. Reverse the span character string.
  3. NFKD-decompose to restore the standard ل+alef sequence in correct
     logical order (e.g. U+FEFB → U+0644 U+0627 = ل + ا).
  4. The caller finishes with NFKC to re-compose any canonical forms.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

PDFType = Literal["digital", "scanned", "mixed"]

_DIGITAL_CHARS_THRESHOLD   = 100
_LINE_TOL_PT               = 4.0   # y-tolerance for grouping spans onto same line
_WORD_GAP_PT               = 3.0   # minimum gap (pts) between spans to insert a space
_DIAC_SPAN_EPSILON         = 5.0   # pt: demote single-char diacritic spans in RTL span sort
_CORRUPT_LATIN1_THRESHOLD  = 0.05  # Latin-1 (U+0080–00FF) fraction above which a
                                   # "digital" page is rerouted to OCR.  These chars
                                   # signal a PDF font with a custom glyph encoding
                                   # that maps Arabic glyphs to extended-Latin code
                                   # points — yielding garbage from any text extractor.
# Decomposed-encoding markers: Arabic Symbol Dots (U+FBB2–FBB6) and the two
# dotless base glyphs (U+066E ٮ, U+06A1 ڡ).  Their presence in digitally-extracted
# text means the font encodes letters as dotless-base + separate dot glyph —
# a class of PDF that OCR handles far better than any text-extraction algorithm.
_DECOMPOSED_MARKERS = re.compile(r'[\uFBB2-\uFBB6\u066E\u06A1]')
# U+063C–063F: Arabic Extended letters (ؼ ؽ ؾ ؿ) that virtually never appear in
# standard Modern Arabic text.  Their presence signals an internal Arabic block
# remapping (a different corruption from Latin-1; detected in e.g. Sample 6).
_CORRUPT_ARABIC_EXT        = frozenset(range(0x063C, 0x0640))
_SENT_TERMINAL = re.compile(r'[.؟!]\s*$')
_IS_HEADING    = re.compile(r'^(?![\u064B-\u065F\u0670])(?!.*[.،؛؟!]).{4,55}$')
# Arabic diacritics (combining marks) codepoint set – used in multiple places
_DIACRITIC_CP = (
    set(range(0x0610, 0x061B)) | set(range(0x064B, 0x0653)) | {0x0670}
)

# Arabic letters that never connect to the FOLLOWING letter (non-joining).
# After one of these, the next joining letter always starts a new cluster.
_NON_JOINING_ARABIC = frozenset([
    0x0622, 0x0623, 0x0624, 0x0625, 0x0627, 0x0629,   # آ أ ؤ إ ا ة
    0x062F, 0x0630, 0x0631, 0x0632, 0x0648, 0x0649,   # د ذ ر ز و ى
    0x0671, 0x06BE, 0x06C1,                             # ٱ ھ ہ
])

# Arabic Presentation Forms-B (FE70–FEFF) that are INITIAL forms.
# An Initial-form letter always starts a new connected cluster (word-initial
# position).  Detecting [non-joining][non-joining][initial] at span-start
# positions 0–2 reveals an omitted inter-word space.
_ARABIC_PF_INITIAL = frozenset([
    0xFE8B,  # Yeh with Hamza Above — Initial
    0xFE91,  # Ba — Initial
    0xFE97,  # Ta — Initial
    0xFE9B,  # Tha — Initial
    0xFE9F,  # Jeem — Initial
    0xFEA3,  # Hah — Initial
    0xFEA7,  # Khah — Initial
    0xFEB3,  # Seen — Initial
    0xFEB7,  # Sheen — Initial
    0xFEBB,  # Sad — Initial
    0xFEBF,  # Dad — Initial
    0xFEC3,  # Tah — Initial
    0xFEC7,  # Dhah — Initial
    0xFECB,  # Ain — Initial  ← key case: هو + عملیة
    0xFECF,  # Ghain — Initial
    0xFED3,  # Fa — Initial
    0xFED7,  # Qaf — Initial
    0xFEDB,  # Kaf — Initial
    0xFEDF,  # Lam — Initial
    0xFEE3,  # Meem — Initial
    0xFEE7,  # Nun — Initial
    0xFEEB,  # Heh — Initial
    0xFEF3,  # Yeh — Initial
])

# Lam-alef obligatory ligature pairs → Unicode Presentation Form placeholders.
# Kept as module-level reference; the active fix uses per-character x comparison.
_LAM_ALEF_PF: list[tuple[str, str]] = [
    ('\u0644\u0622', '\uFEF5'),   # ل + آ  (madda above)
    ('\u0644\u0623', '\uFEF7'),   # ل + أ  (hamza above)
    ('\u0644\u0625', '\uFEF9'),   # ل + إ  (hamza below)
    ('\u0644\u0627', '\uFEFB'),   # ل + ا  (plain alef)
]


def _fix_lamalef_visual_span(
    chars: list[str],
    x_origins: list[float],
) -> str:
    """
    Correct lam-alef ligature pairs in a visual-order (ascending-x) Arabic span.

    In an ascending-x (left→right) visual stream, PDF generators sometimes
    preserve obligatory lam-alef ligatures (ل+ا, ل+أ, ل+إ, ل+آ) in their
    *logical* order: lam is stored BEFORE the alef-variant even though the alef
    has a higher screen-x and would normally appear first in an ascending-x run.

    Distinguishing true preservation from a plain ل+ا adjacency:
      • True preservation  →  x(lam) > x(alef-variant)
        (lam is more to the right on screen but stored first = ligature preserved)
      • Plain adjacency    →  x(lam) ≤ x(alef-variant)
        (lam is naturally to the left of the alef; plain reversal handles it)

    For true-preservation pairs:
      1. Replace with a single Presentation Form code point (U+FEF5–FEFB) so
         the pair survives reversal as ONE character.
      2. Reverse the entire char list (visual → logical order).
      3. NFKD-decompose to restore the standard ل+alef sequence in correct
         logical order.

    Plain-adjacency pairs are handled correctly by step 2 alone.
    """
    _ALEF_TO_PF = {
        '\u0627': '\uFEFB',   # ا
        '\u0622': '\uFEF5',   # آ
        '\u0623': '\uFEF7',   # أ
        '\u0625': '\uFEF9',   # إ
    }
    _LAM = '\u0644'

    n = len(chars)
    result: list[str | None] = list(chars)

    i = 0
    while i < n - 1:
        if result[i] == _LAM:
            nxt = result[i + 1]
            if nxt is not None and nxt in _ALEF_TO_PF:
                # Apply FEFB only when lam has HIGHER screen-x than the alef
                # (true preservation: lam stored first despite being more rightward)
                if x_origins[i] > x_origins[i + 1]:
                    result[i]     = _ALEF_TO_PF[nxt]
                    result[i + 1] = None   # consumed into the placeholder
                    i += 2
                    continue
        i += 1

    filtered = [c for c in result if c is not None]
    return unicodedata.normalize('NFKD', ''.join(filtered[::-1]))


@dataclass
class RawPage:
    page_number:  int
    pdf_type:     PDFType
    raw_text:     str
    raw_text_pre: str = ""
    image_bytes:  bytes | None = field(default=None, repr=False)


@dataclass
class IngestionResult:
    source_path: str
    pdf_type:    PDFType
    total_pages: int
    pages:       list[RawPage]
    metadata:    dict


class PDFIngestor:
    def __init__(self, dpi: int = 200):
        self.dpi = dpi

    def ingest(self, pdf_path: str | Path) -> IngestionResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc   = fitz.open(str(pdf_path))
        meta  = self._extract_metadata(doc)
        pages: list[RawPage] = []

        # Pre-render all pages with pdf2image (poppler) so that scanned pages
        # are rendered using the full MediaBox rather than PyMuPDF's CropBox.
        # CropBox rendering clips page-edge text (e.g. attribution headers).
        _rendered_images: list = []
        try:
            from pdf2image import convert_from_path as _pdf2img  # noqa: PLC0415
            _rendered_images = _pdf2img(str(pdf_path), dpi=self.dpi)
            logger.info("Scanned pages rendered via pdf2image (poppler) at %d DPI.", self.dpi)
        except ImportError:
            logger.info("pdf2image not installed — using PyMuPDF rendering for scanned pages.")
        except Exception as _err:
            logger.warning("pdf2image failed (%s) — falling back to PyMuPDF rendering.", _err)

        try:
            for i, page in enumerate(doc):
                page_num   = i + 1
                probe_text = page.get_text("text").strip()
                is_digital = len(probe_text) >= _DIGITAL_CHARS_THRESHOLD

                # Even if the page has plenty of text, route to OCR when the font
                # encoding is corrupted.  Two distinct corruption types are detected:
                #   1. Latin-1 Supplement chars (U+0080–00FF) map Arabic glyphs to
                #      extended-Latin code points.
                #   2. Arabic Extended chars (U+063C–003F) appear as letter substitutes
                #      due to an internal Arabic block remapping in the font's CMap.
                if is_digital and self._is_corrupted(page):
                    logger.warning(
                        "Page %d of '%s' has corrupted font encoding — routing to OCR.",
                        page_num, pdf_path.name,
                    )
                    is_digital = False

                if is_digital:
                    text = self._extract_rtl_text(page)
                    # Auto-detect decomposed font encoding (dotless-base + symbol-dot
                    # pairs) and re-route to OCR.  These PDFs produce far cleaner
                    # output from Tesseract than from any text-extraction algorithm.
                    # The check is O(n) on page text — negligible vs. rendering cost.
                    if _DECOMPOSED_MARKERS.search(text):
                        logger.info(
                            "Page %d of '%s': decomposed Arabic font encoding detected"
                            " — re-routing to OCR.",
                            page_num, pdf_path.name,
                        )
                        is_digital = False   # fall through to image render below

                if is_digital:
                    pages.append(RawPage(
                        page_number  = page_num,
                        pdf_type     = "digital",
                        raw_text     = text,
                        raw_text_pre = text,
                    ))
                else:
                    if i < len(_rendered_images):
                        # poppler render — full page, no CropBox clipping
                        import io as _io
                        _buf = _io.BytesIO()
                        _rendered_images[i].save(_buf, format="PNG")
                        img_bytes = _buf.getvalue()
                    else:
                        # PyMuPDF fallback — expand CropBox to MediaBox so that
                        # text at page-edge margins is not silently clipped.
                        # (pdf2image uses poppler which always renders the full
                        # MediaBox; we replicate that here for the fallback path.)
                        page.set_cropbox(page.mediabox)
                        mat       = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                        pix       = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                        img_bytes = pix.tobytes("png")
                    pages.append(RawPage(
                        page_number  = page_num,
                        pdf_type     = "scanned",
                        raw_text     = "",
                        raw_text_pre = "",
                        image_bytes  = img_bytes,
                    ))
        finally:
            doc.close()
        digital = sum(1 for p in pages if p.pdf_type == "digital")
        scanned = sum(1 for p in pages if p.pdf_type == "scanned")
        overall_type: PDFType = (
            "scanned" if digital == 0 else
            "digital" if scanned == 0 else "mixed"
        )
        logger.info("Ingested '%s' — %d pages (%d digital, %d scanned)",
                    pdf_path.name, len(pages), digital, scanned)
        return IngestionResult(
            source_path=str(pdf_path),
            pdf_type=overall_type,
            total_pages=len(pages),
            pages=pages,
            metadata=meta,
        )

    # ------------------------------------------------------------------ #
    #  Corruption detection                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_corrupted(page: fitz.Page) -> bool:
        """
        Return True when the page's font encoding is corrupted and reliable
        text extraction is impossible.

        Two corruption types are detected:

        Type 1 — Latin-1 remapping (Samples 4 & 5):
            A custom CMap maps Arabic glyphs to Latin-1 Supplement code points
            (U+0080–00FF).  Detected when ≥ 5 % of visible chars fall in that
            range.  Clean Arabic PDFs score 0 %; medical/technical PDFs that
            legitimately use ASCII abbreviations still score 0 % because ASCII
            stays in U+0021–007E.

        Type 2 — Internal Arabic block remapping (Sample 6):
            A custom CMap maps some Arabic letters to rare Extended Arabic code
            points (U+063C–063F: ؼ ؽ ؾ ؿ) that virtually never occur in
            standard Modern Arabic text.  Even a single occurrence is conclusive
            because these characters have no role in MSA.
        """
        raw = page.get_text(
            "rawdict",
            flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP,
        )
        n_total = n_latin1 = 0
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for ch in span.get("chars", []):
                        cp = ch.get("c", 0)
                        if isinstance(cp, str):
                            cp = ord(cp) if cp else 0
                        if cp <= 0x20:
                            continue
                        n_total += 1
                        if 0x0080 <= cp <= 0x00FF:
                            n_latin1 += 1
                        elif cp in _CORRUPT_ARABIC_EXT:
                            return True   # Type 2: one occurrence is enough
        # Type 1: threshold-based
        return (n_latin1 / n_total) >= _CORRUPT_LATIN1_THRESHOLD if n_total > 0 else False

    # ------------------------------------------------------------------ #
    #  RTL text extraction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_rtl_text(page: fitz.Page) -> str:
        raw = page.get_text(
            "rawdict",
            flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP,
        )

        # Each entry: (x_left, x_right, y, text)
        # x_left  = leftmost char origin  → used for RTL sort (descending)
        # x_right = rightmost char bbox right edge → used for gap detection
        block_texts: list[str] = []

        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            span_entries: list[tuple[float, float, float, str]] = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    char_data = span.get("chars", [])
                    if not char_data:
                        continue

                    span_chars: list[str] = []
                    x_origins: list[float] = []
                    x_rights:  list[float] = []
                    y_origins: list[float] = []

                    for ch in char_data:
                        c = ch.get("c", 0)
                        if isinstance(c, int):
                            if c < 0x20:   # skip control chars; keep space (0x20) as word boundary
                                continue
                            ch_str = chr(c)
                        else:
                            ch_str = str(c)
                            if not ch_str:
                                continue
                            if len(ch_str) == 1 and ord(ch_str) < 0x20:
                                continue

                        ox, oy = ch["origin"]
                        # bbox: (x0, y0, x1, y1) — x1 is right edge of glyph
                        bbox = ch.get("bbox", (ox, oy, ox, oy))
                        span_chars.append(ch_str)
                        x_origins.append(ox)
                        x_rights.append(bbox[2])
                        y_origins.append(oy)

                    if not span_chars:
                        continue

                    # Sort chars by x DESC to recover logical Unicode order.
                    #
                    # For spans containing diacritics we use a nearest-host
                    # assignment: each diacritic is placed immediately after the
                    # base letter whose x-origin is closest (min absolute distance).
                    # This avoids the fundamental conflict that arises with a fixed
                    # epsilon — e.g. خصوصًا needs epsilon > 5.9 pt while واحدًا
                    # needs epsilon < 5.0 pt for the same kind of diacritic.
                    # Arabic standard block + both Presentation Form blocks
                    # (FB50–FDFF = Arabic Presentation Forms-A,
                    #  FE70–FEFF = Arabic Presentation Forms-B).
                    # NFKC normalisation later decomposes all presentation
                    # forms to their base Arabic equivalents.
                    has_arabic = any(
                        '\u0600' <= c <= '\u06FF'
                        or '\uFB50' <= c <= '\uFDFF'
                        or '\uFE70' <= c <= '\uFEFF'
                        for c in span_chars
                    )
                    if has_arabic and len(span_chars) > 1:
                        base_idx = [k for k in range(len(span_chars))
                                    if ord(span_chars[k]) not in _DIACRITIC_CP]
                        diac_idx = [k for k in range(len(span_chars))
                                    if ord(span_chars[k]) in _DIACRITIC_CP]

                        if not base_idx or not diac_idx:
                            # All base or all diacritics: simple x-DESC sort
                            order = sorted(
                                range(len(span_chars)),
                                key=lambda k: (-round(x_origins[k], 1), k),
                            )
                        else:
                            # Assign each diacritic to its nearest base letter.
                            # Diacritic sort key: (same rounded-x bucket as host,
                            # secondary=1 so it follows host's secondary=0).
                            diac_set = set(diac_idx)
                            host_rx: dict[int, float] = {}
                            for d in diac_idx:
                                nb = min(base_idx,
                                         key=lambda b: abs(x_origins[b] - x_origins[d]))
                                host_rx[d] = round(x_origins[nb], 1)

                            def _char_sort_key(k: int) -> tuple:
                                if k in diac_set:
                                    return (-host_rx[k], 1, k)
                                return (-round(x_origins[k], 1), 0, k)

                            order = sorted(range(len(span_chars)), key=_char_sort_key)

                        # ── Intra-span word-break detection ────────────────────
                        # Some Arabic PDFs omit the word-space glyph between two
                        # adjacent words.  Detect by checking the first three chars
                        # (positions 0–2 in x-DESC sorted order): if two consecutive
                        # non-joining Arabic base letters are immediately followed by
                        # an Arabic Initial Presentation Form, the PDF likely merged
                        # a short word (e.g. the pronoun هو) with the next word.
                        # Restricting the check to span-start positions 0–2 prevents
                        # false positives for intra-word sequences like وأنواعه.
                        if (len(order) >= 3
                                and ord(span_chars[order[0]]) in _NON_JOINING_ARABIC
                                and ord(span_chars[order[1]]) in _NON_JOINING_ARABIC
                                and ord(span_chars[order[2]]) in _ARABIC_PF_INITIAL):
                            raw_chars = ("".join(span_chars[k] for k in order[:2])
                                         + ' '
                                         + "".join(span_chars[k] for k in order[2:]))
                        else:
                            raw_chars = "".join(span_chars[k] for k in order)
                    else:
                        # Non-Arabic spans (punctuation, spaces, digits): sort by
                        # x-DESC to respect RTL embedding order.  Without this, a
                        # span like [space(x=170), period(x=174)] stays as " ." and
                        # the period ends up BEFORE the next Arabic word instead of
                        # after the preceding one.
                        order = sorted(range(len(span_chars)),
                                       key=lambda k: -x_origins[k])
                        raw_chars = "".join(span_chars[k] for k in order)

                    span_text = unicodedata.normalize("NFKC", raw_chars)
                    if not span_text.strip():
                        continue

                    x_left  = min(x_origins)
                    x_right = max(x_rights)
                    # Use y of first non-diacritic char so that spans whose first
                    # PDF char is a diacritic (with a shifted baseline) still get
                    # grouped onto the correct text line.
                    non_diac_ys = [
                        y_origins[k] for k in range(len(span_chars))
                        if ord(span_chars[k]) not in _DIACRITIC_CP
                    ]
                    y_rep = non_diac_ys[0] if non_diac_ys else y_origins[0]
                    span_entries.append((x_left, x_right, y_rep, span_text))

            if span_entries:
                block_text = PDFIngestor._span_entries_to_text(span_entries)
                if block_text.strip():
                    block_texts.append(block_text)

        return "\n\n".join(block_texts)

    @staticmethod
    def _span_entries_to_text(span_entries: list[tuple[float, float, float, str]]) -> str:
        """Convert a list of (x_left, x_right, y, text) span entries into paragraph text."""
        if not span_entries:
            return ""

        # ── Group spans into visual lines by y-coordinate ──────────────
        span_entries.sort(key=lambda e: e[2])   # sort by y
        lines: list[list[tuple[float, float, str]]] = []
        current_line: list[tuple[float, float, str]] = []
        current_y = span_entries[0][2]

        for x_l, x_r, y, text in span_entries:
            if abs(y - current_y) > _LINE_TOL_PT:
                if current_line:
                    lines.append(current_line)
                current_line = [(x_l, x_r, text)]
                current_y    = y
            else:
                current_line.append((x_l, x_r, text))
        if current_line:
            lines.append(current_line)

        # ── Per line: sort spans RTL, merge diacritics, join with gap ──
        _ALEF_CHARS = {'\u0627', '\u0622', '\u0623', '\u0625'}

        def is_diacritic_only(s: str) -> bool:
            """Catches pure diacritics AND alef+diacritics (tanwin-fath marker اًّ)."""
            s = s.strip()
            if not s:
                return False
            if all(ord(c) in _DIACRITIC_CP for c in s):
                return True
            # Standalone alef followed only by diacritics = tanwin-fath marker
            if s[0] in _ALEF_CHARS and len(s) >= 2 and all(ord(c) in _DIACRITIC_CP for c in s[1:]):
                return True
            return False

        # Punctuation chars that must not have a space inserted BEFORE them
        # (i.e. they attach directly to the preceding word in Arabic).
        _ATTACH_BEFORE = re.compile(r'^[.،؛؟!:\u0640]+')

        def fix_comma(line: str) -> str:
            # In RTL text, ، follows the word to its RIGHT in visual space
            # (the word that PRECEDES it in reading order).
            # Pattern: X ، Y → X Y،   (comma trails Y, which is read first)
            line = re.sub(r'(\S+)\s+،\s*(\S+)', r'\1 \2،', line)
            # Leading ، with no left-side word: ، X → X،
            line = re.sub(r'^،\s*(\S+)', r'\1،', line)
            return line

        def clean_punct(line: str) -> str:
            line = re.sub(r'،،+', '،', line)
            line = re.sub(r'\.{2,}', '.', line)
            return line

        visual_lines: list[str] = []
        _pending_prefix = ""   # right-fragment carried from previous line's mid-line split

        for line_spans in lines:
            # Sort descending by left-x → RTL reading order.
            # Diacritic-only spans are demoted by _DIAC_SPAN_EPSILON so they
            # sort just AFTER the word span whose x range they overlap, rather
            # than accidentally preceding it due to a tiny x difference.
            line_spans.sort(
                key=lambda t: t[0] - (_DIAC_SPAN_EPSILON if is_diacritic_only(t[2]) else 0.0),
                reverse=True,
            )

            # Merge diacritic-only spans into adjacent word spans
            # (x_left, x_right, text)
            merged: list[tuple[float, float, str]] = []
            pending_diac = ""

            for x_l, x_r, t in line_spans:
                if is_diacritic_only(t):
                    if merged:
                        prev = merged[-1]
                        merged[-1] = (prev[0], prev[1], prev[2] + t)
                    else:
                        pending_diac += t
                else:
                    _ALEF_JOIN = '\u0627\u0622\u0623\u0625\u0671'
                    if pending_diac and pending_diac[0] in _ALEF_JOIN:
                        merged.append((x_l, x_r, t + pending_diac))
                    else:
                        merged.append((x_l, x_r, pending_diac + t))
                    pending_diac = ""

            if pending_diac and merged:
                prev = merged[-1]
                merged[-1] = (prev[0], prev[1], prev[2] + pending_diac)

            if not merged:
                continue

            # Join spans: insert space only when visual gap exceeds threshold.
            # When spans are adjacent (gap < threshold), also check if the
            # RIGHT-side span (sorted first = higher x) is a diacritic/tanwin-alef.
            # If so, it belongs AFTER the left-side span (append, not prepend).
            # e.g. اًّ (high x) + ضروري (low x) → gap=0 → join as ضروريًّا not اًّضروري
            parts: list[str] = []
            skip_next = False
            for i in range(len(merged)):
                if skip_next:
                    skip_next = False
                    continue
                x_l, x_r, text = merged[i]
                if i + 1 < len(merged):
                    next_x_l, next_x_r, next_text = merged[i + 1]
                    gap = x_l - next_x_r
                    if gap < _WORD_GAP_PT and is_diacritic_only(text):
                        # This span (rightmost) is diacritic — append to next word
                        parts.append(next_text + text)
                        skip_next = True
                        continue
                    elif gap >= _WORD_GAP_PT and not _ATTACH_BEFORE.match(next_text):
                        parts.append(text)
                        parts.append(" ")
                        continue
                parts.append(text)

            line_text = clean_punct(fix_comma("".join(parts).strip()))
            if _pending_prefix:
                line_text = _pending_prefix + " " + line_text if line_text else _pending_prefix
                _pending_prefix = ""
            if not line_text:
                continue
            # Split on mid-line sentence terminals (period/؟/! followed by a
            # space and an Arabic letter) so paragraph reconstruction can detect
            # sentence boundaries that fall inside a single PDF visual line.
            # The right-fragment of the LAST split is carried as a prefix to the
            # NEXT visual line — this avoids the fragment being mistaken for a
            # standalone heading when it is really a sentence opener that
            # continues on the following line.
            sub = re.split(r'(?<=[.؟!])\s+(?=[\u0600-\u06FF])', line_text)
            for part in sub[:-1]:
                if part.strip():
                    visual_lines.append(part)
            last = sub[-1].strip()
            if len(sub) > 1 and last:
                _pending_prefix = last   # carry forward; join with next line
            elif last:
                visual_lines.append(last)

        if _pending_prefix:              # flush any remaining prefix
            visual_lines.append(_pending_prefix)

        # ── Paragraph reconstruction with heading detection ─────────────
        if not visual_lines:
            return ""

        _DIAC_ONLY_LINE = re.compile(
            r'^[\u0600-\u0615\u064B-\u065F\u0670\u0627\u0622\u0623\u0625\s]+$'
        )

        def is_heading(s: str) -> bool:
            s = s.strip()
            if _DIAC_ONLY_LINE.match(s):
                return False
            return bool(_IS_HEADING.match(s)) and len(s) <= 55

        paragraphs: list[str] = []
        buffer = visual_lines[0]

        for line in visual_lines[1:]:
            if is_heading(buffer) and paragraphs:
                paragraphs.extend([buffer, ""])
                buffer = line
            elif is_heading(line) and buffer.strip():
                paragraphs.extend([buffer, ""])
                buffer = line
            elif _SENT_TERMINAL.search(buffer):
                paragraphs.append(buffer)
                buffer = line
            else:
                buffer = buffer + " " + line

        paragraphs.append(buffer)
        while paragraphs and not paragraphs[0].strip():
            paragraphs.pop(0)
        while paragraphs and not paragraphs[-1].strip():
            paragraphs.pop()
        return "\n".join(paragraphs)

    @staticmethod
    def _extract_metadata(doc: fitz.Document) -> dict:
        raw = doc.metadata or {}
        return {
            "title":   raw.get("title",   ""),
            "author":  raw.get("author",  ""),
            "subject": raw.get("subject", ""),
            "creator": raw.get("creator", ""),
            "pages":   doc.page_count,
        }
