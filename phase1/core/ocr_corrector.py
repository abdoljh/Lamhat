"""
Phase 1 — OCRTextCorrector
Passes raw Tesseract output through Claude Haiku to fix OCR errors and
join line-wrapped text into clean, flowing Arabic paragraphs.

The corrector runs page-by-page after OCR and before the normalizer.
raw_text_pre always retains the original Tesseract output for auditing;
raw_text is replaced with the LLM-corrected version.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .ingestor import RawPage

logger = logging.getLogger(__name__)

_CORRECTION_PROMPT = """\
أنت خبير في تحقيق النصوص التاريخية. مهمتك تصحيح نص OCR من كتاب عربي مطبوع.

طبِّق هذه القواعد:
1. دمج الأسطر: ألغِ الفواصل الاصطناعية داخل الفقرة الواحدة وحوِّل النص إلى فقرات متدفقة ومتصلة.
2. إصلاح أخطاء OCR: صحِّح الكلمات المشوَّهة آليًّا دون تغيير الأسلوب أو إضافة كلمات. فضِّل دائماً الكلمة الأقرب شكلاً وحرفاً إلى النص المشوَّه على الكلمة الأكثر انسجاماً سياقياً (مثال: "بئات" ← "بنات" لأن ئ≈ن بصرياً، وليس "بهاء"). أخطاء OCR الشائعة في العربية: ئ/ى/ي↔ن، ة↔ه، و↔ر، ق↔ف.
3. الأسماء التاريخية والجغرافية: احتفظ بها كما وردت.
4. الجمل المفتوحة: لا تضف نقطة نهاية إذا انتهى النص بجملة غير مكتملة.
5. العناوين والأسماء الرئيسية: أبقِها في أسطر مستقلة.
6. الشمولية: لا تحذف أي محتوى.

أخرِج النص المصحح مباشرةً فقط، بدون أي عنوان أو تعليق في البداية.

النص الخام:
{text}
"""


class OCRTextCorrector:
    """
    Corrects raw Tesseract output for scanned Arabic pages using Claude Haiku.

    One API call is made per page.  Failures are fail-open: if the API call
    raises an exception the original Tesseract text is kept unchanged.

    Cost estimate: ~$0.001 per page (Haiku pricing, ~2 k tokens in+out).
    A 255-page book costs roughly $0.25 in correction calls.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        on_progress: Callable[[str, float], None] | None = None,
    ):
        self.api_key     = api_key
        self.model       = model
        self.on_progress = on_progress or (lambda s, p: None)
        self._client     = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def correct_pages(self, pages: list[RawPage]) -> list[RawPage]:
        """
        Correct OCR text for scanned pages in-place.
        Digital pages pass through unchanged.
        Returns the same list.
        """
        scanned = [p for p in pages if p.pdf_type == "scanned" and p.raw_text.strip()]
        if not scanned:
            logger.info("No scanned pages with text — OCR correction skipped.")
            return pages

        logger.info(
            "LLM OCR correction: %d scanned page(s) via %s",
            len(scanned), self.model,
        )
        self._ensure_client()

        for idx, page in enumerate(scanned):
            self.on_progress(
                f"LLM OCR correction — page {page.page_number} / {scanned[-1].page_number} …",
                idx / len(scanned),
            )
            corrected = self._correct_page(page.raw_text, page.page_number)
            if corrected.strip():
                page.raw_text = corrected
            else:
                logger.warning(
                    "Page %d: LLM returned empty text — keeping original Tesseract output.",
                    page.page_number,
                )

        return pages

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # noqa: PLC0415
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def _correct_page(self, raw_text: str, page_num: int) -> str:
        try:
            response = self._client.messages.create(
                model      = self.model,
                max_tokens = 2048,
                messages   = [
                    {
                        "role":    "user",
                        "content": _CORRECTION_PROMPT.format(text=raw_text),
                    }
                ],
            )
            text = response.content[0].text.strip()
            # Strip any echoed prompt label the model may prepend
            # (e.g. "النص المصحح:" or "# النص المصحح:")
            lines = text.splitlines()
            if lines and "النص المصحح" in lines[0]:
                text = "\n".join(lines[1:]).lstrip()
            return text
        except Exception as exc:
            logger.warning("Page %d: LLM OCR correction API call failed: %s", page_num, exc)
            return raw_text  # fail-open
