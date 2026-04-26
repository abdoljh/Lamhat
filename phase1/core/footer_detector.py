"""
Phase 1 — FooterDetector
Classifies and strips footer/header elements from scanned Arabic OCR pages.

Detected element types
──────────────────────
PAGE_NUMBER    — standalone or embedded digit lines (Arabic-Indic, Eastern, Western)
FOOTNOTE       — lines starting with parenthesised numbers, asterisks, daggers;
                 handles RTL-reversed parentheses )١( common in raw Arabic OCR
RUNNING_HEADER — short top-of-page lines without sentence punctuation, with an
                 embedded number or known header keyword
SEPARATOR      — horizontal rule lines (dashes, underscores, equals signs)
FOOTER_TEXT    — generic short footer lines not matching other categories

Algorithm
─────────
For each page:
  • Top 15 % of lines  → check for RUNNING_HEADER
  • Bottom 15 % of lines → check for PAGE_NUMBER, FOOTNOTE, SEPARATOR
  • Multi-line footnote continuations are linked to their marker lines.
  • Cross-page continuation: a footnote line ending with ``=`` or `` -`` sets a
    flag so the next page's top lines beginning with ``=`` or ``- `` are stripped
    as continuation.

Adapted from output/ph1-nb/footer_detector_v3.py (tested on Al-Askari Memoirs,
pages_5_7.pdf).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class FooterType(Enum):
    PAGE_NUMBER    = "page_number"
    FOOTNOTE       = "footnote"
    RUNNING_HEADER = "running_header"
    FOOTER_TEXT    = "footer_text"
    SEPARATOR      = "separator"
    UNKNOWN        = "unknown"


@dataclass
class DetectedFooter:
    text:          str
    footer_type:   FooterType
    confidence:    float
    page_num:      int
    line_index:    int
    original_line: str
    is_stripped:   bool = False


class FooterDetector:
    """
    Detect and strip running headers, page numbers, footnotes, and separator
    lines from a single page of Arabic OCR output.

    Usage::

        detector = FooterDetector()
        footers  = detector.analyze_page(page_text, page_num)
        cleaned  = detector.strip_footers(page_text, footers)
        detector.reset()          # before the next page
    """

    def __init__(
        self,
        page_height_ratio: float = 0.15,
        min_footer_lines:  int   = 1,
    ):
        self.page_height_ratio  = page_height_ratio
        self.min_footer_lines   = min_footer_lines
        self.detected_footers: List[DetectedFooter] = []
        # Cross-page footnote continuation: set True when the last processed
        # page ended with a footnote line ending in ``=`` or `` -`` (Arabic
        # typographic convention for footnotes that overflow to the next page).
        # Consumed (reset to False) at the start of the next analyze_page call.
        self._continuation_pending: bool = False

    # ------------------------------------------------------------------ #
    #  Bidi cleaning                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_bidi_marks(text: str) -> str:
        """Strip invisible bidirectional-control characters before pattern matching."""
        _BIDI = '‎‏‪‫‬‭‮‍‌'
        return ''.join(c for c in text if c not in _BIDI)

    # ------------------------------------------------------------------ #
    #  Inline number extraction                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_inline_numbers(text: str) -> List[Tuple[str, float]]:
        """
        Return (number_string, confidence) pairs for digits embedded in a line.
        Covers Arabic-Indic (U+0660-0669), Eastern Arabic-Indic (U+06F0-06F9),
        and Western ASCII digits.
        """
        numbers: List[Tuple[str, float]] = []
        sep = r'[\s,\-–—]*'
        for pat, conf in [
            (rf'{sep}([٠-٩]{{1,3}}){sep}', 0.85),
            (rf'{sep}([۰-۹]{{1,3}}){sep}', 0.85),
            (rf'{sep}([0-9]{{1,3}}){sep}',            0.80),
        ]:
            for m in re.finditer(pat, text):
                numbers.append((m.group(1), conf))
        return numbers

    # ------------------------------------------------------------------ #
    #  Individual classifiers                                              #
    # ------------------------------------------------------------------ #

    def _is_page_number(self, text: str) -> Tuple[bool, float]:
        s = text.strip()
        if not s:
            return False, 0.0
        # Pure digit lines
        if re.fullmatch(r'[٠-٩]+', s):           return True, 0.95
        if re.fullmatch(r'[۰-۹]+', s):           return True, 0.95
        if re.fullmatch(r'[0-9]+', s):                     return True, 0.90
        # Digits with decorative surroundings
        if re.fullmatch(r'[-–—\s]*[٠-٩۰-۹0-9]+[-–—\s]*', s):
            return True, 0.85
        # Arabic word for "page" + number
        if re.search(r'صفحة?\s*[٠-٩۰-۹0-9]+', s):
            return True, 0.90
        return False, 0.0

    def _is_footnote(self, text: str) -> Tuple[bool, float]:
        """
        Detect footnote marker lines.

        Handles RTL-reversed parentheses: in raw Arabic OCR the visual "("
        may be encoded as U+0029 ) and vice versa — yielding )١( instead of (١).
        Bidi marks are cleaned first.
        """
        cleaned = self._clean_bidi_marks(text)
        s = cleaned.strip()
        if not s:
            return False, 0.0
        # Standard: (١), [٢], {٣}, (؟)  or Western equivalents
        # ؟ (U+061F Arabic question mark) appears in some footnotes as a "sic" marker.
        if re.match(r'^[\(\[\{]\s*[٠-٩۰-۹0-9؟\?]\s*[\)\]\}]', s):
            return True, 0.95
        # RTL-reversed: )١( or )؟(
        if re.match(r'^[\)\]\}]\s*[٠-٩۰-۹0-9؟\?]\s*[\(\[\{]', s):
            return True, 0.90
        # Asterisk, dagger, or similar typographic markers
        if re.match(r'^[*†‡§¶#\+\-—]', s):
            return True, 0.85
        # Arabic letter + closing paren  e.g.  أ)
        if re.match(r'^[ء-ي]\)', s):
            return True, 0.70
        # Short line with cross-reference keywords (انظر، راجع، هامش)
        if len(s) < 50 and any(kw in s for kw in [
            'انظر',   # انظر
            'راجع',   # راجع
            'هامش',   # هامش
        ]):
            return True, 0.60
        return False, 0.0

    def _is_separator(self, text: str) -> Tuple[bool, float]:
        s = text.strip()
        if not s:
            return False, 0.0
        if re.fullmatch(r'[-_*=—–]+', s):
            return True, 0.90
        if re.fullmatch(r'[-_*=—–\s]*[٠-٩۰-۹0-9]+[-_*=—–\s]*', s):
            return True, 0.75
        return False, 0.0

    def _is_running_header(self, text: str) -> Tuple[bool, float]:
        """
        Detect running headers in the top region of a page.

        Stricter than the cross-page frequency approach in normalizer:
          • Must be < 60 chars
          • Must NOT end with sentence punctuation (., ،, :, ;)
          • Must have an embedded page number OR a known chapter/title keyword
            OR be very short (< 30 chars, handled as likely title fragment)
        """
        s = text.strip()
        if not s or len(s) >= 60:
            return False, 0.0
        if any(c in s for c in '.،:؛'):
            return False, 0.0
        has_title = bool(re.search(
            r'مقدمة'
            r'|فصل'
            r'|كتاب'
            r'|ذكريات'
            r'|مذكرات'
            r'|مدكرات',   # common Tesseract misspelling of مذكرات
            s,
        ))
        has_number = bool(re.search(r'[٠-٩۰-۹0-9]', s))
        # Only strip a top-of-page line when it clearly looks like a header:
        # has an embedded page number OR a known book/chapter keyword.
        # Short-but-pure-text lines (attributions, section titles) must NOT be
        # stripped here — use the cross-page frequency detector for those.
        if has_title or has_number:
            return True, 0.80
        return False, 0.0

    # ------------------------------------------------------------------ #
    #  Multi-line footnote continuation                                    #
    # ------------------------------------------------------------------ #

    def _link_footnote_continuations(
        self,
        lines: List[str],
        footers: List[DetectedFooter],
        page_num: int,
    ) -> List[DetectedFooter]:
        """
        Attach continuation lines that follow a footnote marker.

        Scans up to 15 lines after each footnote marker (increased from 5 to
        handle long footnotes).  Stops at a blank line or a new footnote marker.
        """
        fn_markers = [f for f in footers if f.footer_type == FooterType.FOOTNOTE]
        already = {f.line_index for f in footers}
        for fn in fn_markers:
            idx = fn.line_index + 1
            while idx < len(lines) and idx < fn.line_index + 15:
                line = lines[idx].strip()
                if not line:
                    break
                if idx in already:
                    idx += 1
                    continue
                cleaned = self._clean_bidi_marks(line)
                is_new = (
                    re.match(r'^[\(\[\{]\s*[٠-٩۰-۹0-9]\s*[\)\]\}]', cleaned) or
                    re.match(r'^[\)\]\}]\s*[٠-٩۰-۹0-9]\s*[\(\[\{]', cleaned)
                )
                if is_new:
                    break
                if len(line) < 120 or not line.endswith('.'):
                    cont = DetectedFooter(
                        text=line,
                        footer_type=FooterType.FOOTNOTE,
                        confidence=0.60,
                        page_num=page_num,
                        line_index=idx,
                        original_line=lines[idx],
                    )
                    footers.append(cont)
                    already.add(idx)
                    idx += 1
                else:
                    break
        return footers

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze_page(self, page_text: str, page_num: int) -> List[DetectedFooter]:
        """
        Classify all footer/header elements on one page.

        Results are appended to ``self.detected_footers`` and also returned.
        Call ``reset()`` between pages (within a document).
        """
        lines       = page_text.split('\n')
        total       = len(lines)
        if total == 0:
            self._continuation_pending = False
            return []
        # Ensure at least 1 line is examined at each end even on short pages.
        header_end   = max(1, int(total * self.page_height_ratio))
        footer_start = min(total - 1, int(total * (1 - self.page_height_ratio)))
        footers: List[DetectedFooter] = []

        # ── Cross-page footnote continuation from previous page ─────────
        # When the previous page ended with a footnote line containing the
        # Arabic page-break marker (line ending in ``=`` or `` -``), the
        # continuation starts at the top of this page with ``=`` or ``- ``.
        cross_page_indices: set = set()
        if self._continuation_pending:
            self._continuation_pending = False
            for idx in range(min(total, 12)):
                line = lines[idx]
                if not line.strip():
                    continue
                s = self._clean_bidi_marks(line).strip()
                is_cross = (
                    s.startswith('=') or
                    (s.startswith('-') and len(s) > 1 and s[1] in ' \t')
                )
                if is_cross:
                    cross = DetectedFooter(
                        text=s, footer_type=FooterType.FOOTNOTE,
                        confidence=0.80, page_num=page_num,
                        line_index=idx, original_line=line,
                    )
                    footers.append(cross)
                    cross_page_indices.add(idx)
                    # Link continuation lines immediately following
                    c_idx = idx + 1
                    while c_idx < len(lines) and c_idx < idx + 15:
                        c_line = lines[c_idx].strip()
                        if not c_line:
                            break
                        c_cleaned = self._clean_bidi_marks(c_line)
                        is_new = (
                            re.match(r'^[\(\[\{]\s*[٠-٩۰-۹0-9]\s*[\)\]\}]', c_cleaned) or
                            re.match(r'^[\)\]\}]\s*[٠-٩۰-۹0-9]\s*[\(\[\{]', c_cleaned)
                        )
                        if is_new:
                            break
                        cont = DetectedFooter(
                            text=c_line, footer_type=FooterType.FOOTNOTE,
                            confidence=0.70, page_num=page_num,
                            line_index=c_idx, original_line=lines[c_idx],
                        )
                        footers.append(cont)
                        cross_page_indices.add(c_idx)
                        c_idx += 1
                    break  # only one cross-page block per page

        # ── Bottom region: footnotes → page numbers → separators ────────
        # Footnote is checked first: a line like (١) is a footnote marker,
        # NOT a page number, even though it contains an Arabic digit.
        for idx in range(footer_start, total):
            line = lines[idx]
            if not line.strip() or idx in cross_page_indices:
                continue
            for check, ftype in [
                (self._is_footnote,    FooterType.FOOTNOTE),
                (self._is_page_number, FooterType.PAGE_NUMBER),
                (self._is_separator,   FooterType.SEPARATOR),
            ]:
                detected, conf = check(line)
                if detected:
                    footers.append(DetectedFooter(
                        text=line.strip(), footer_type=ftype,
                        confidence=conf, page_num=page_num,
                        line_index=idx, original_line=line,
                    ))
                    break

        # ── Top region: running headers ──────────────────────────────────
        for idx in range(header_end):
            line = lines[idx]
            if not line.strip() or idx in cross_page_indices:
                continue
            detected, conf = self._is_running_header(line)
            if detected:
                footers.append(DetectedFooter(
                    text=line.strip(), footer_type=FooterType.RUNNING_HEADER,
                    confidence=conf, page_num=page_num,
                    line_index=idx, original_line=line,
                ))
                # Embedded page number inside a header line (e.g. "مذكرات ,5")
                for num_text, num_conf in self._extract_inline_numbers(line):
                    footers.append(DetectedFooter(
                        text=num_text, footer_type=FooterType.PAGE_NUMBER,
                        confidence=num_conf, page_num=page_num,
                        line_index=idx, original_line=line,
                    ))

        # ── Full-page scan for mid-page footnote markers ────────────────
        # Footnotes inserted mid-page by Tesseract layout analysis (outside the
        # bottom 15% region) are caught here.  Only the strict parenthesised-
        # digit / question-mark patterns are used — avoiding false positives on
        # body text that happens to contain a digit.
        already_flagged = {f.line_index for f in footers} | cross_page_indices
        for idx, line in enumerate(lines):
            if idx in already_flagged or not line.strip():
                continue
            cleaned = self._clean_bidi_marks(line)
            s = cleaned.strip()
            if (re.match(r'^[\(\[\{]\s*[٠-٩۰-۹0-9؟\?]\s*[\)\]\}]', s) or
                    re.match(r'^[\)\]\}]\s*[٠-٩۰-۹0-9؟\?]\s*[\(\[\{]', s)):
                fn = DetectedFooter(
                    text=line.strip(), footer_type=FooterType.FOOTNOTE,
                    confidence=0.88, page_num=page_num,
                    line_index=idx, original_line=line,
                )
                footers.append(fn)
                already_flagged.add(idx)

        footers = self._link_footnote_continuations(lines, footers, page_num)

        # ── Set cross-page flag if a footnote line ends with = or `` -`` ─
        # This signals that the footnote continues on the next page.
        for f in footers:
            if f.footer_type == FooterType.FOOTNOTE:
                tail = f.text.rstrip()
                if tail.endswith('=') or tail.endswith(' -'):
                    self._continuation_pending = True
                    break

        self.detected_footers.extend(footers)
        return footers

    def strip_footers(
        self,
        page_text: str,
        footers: List[DetectedFooter],
        preserve_types: Optional[List[FooterType]] = None,
    ) -> str:
        """Remove footer lines from ``page_text``, skipping any ``preserve_types``."""
        if preserve_types is None:
            preserve_types = []
        lines = page_text.split('\n')
        remove = {
            f.line_index for f in footers
            if f.footer_type not in preserve_types
        }
        for f in footers:
            if f.footer_type not in preserve_types:
                f.is_stripped = True
        return '\n'.join(line for i, line in enumerate(lines) if i not in remove)

    def get_footer_report(self) -> str:
        """Human-readable summary of all detected footer elements."""
        if not self.detected_footers:
            return "No footers detected."
        lines = ["=== FOOTER DETECTION REPORT ===", ""]
        cur_page = 0
        for f in self.detected_footers:
            if f.page_num != cur_page:
                cur_page = f.page_num
                lines.append(f"\n--- Page {cur_page} ---")
            status = "STRIPPED" if f.is_stripped else "PRESERVED"
            lines.append(
                f"  [{f.footer_type.value}] (conf: {f.confidence:.2f}) {status}: {f.text[:60]}"
            )
        return '\n'.join(lines)

    def reset(self) -> None:
        """
        Clear per-page accumulated detections.

        Note: ``_continuation_pending`` is intentionally NOT cleared here —
        it must carry over from page N to page N+1 within the same document.
        Create a fresh ``FooterDetector()`` instance between documents.
        """
        self.detected_footers.clear()
