"""
Phase 1 — OCREngine
Three backends: Tesseract (default), EasyOCR, PaddleOCR.

Tesseract is the default and the only backend that fits Streamlit Cloud's
1 GB RAM limit (no PyTorch required).
EasyOCR is a good alternative for local runs with more RAM (~450 MB model).
PaddleOCR gives the best Arabic accuracy but requires ~1 GB and Python ≤ 3.12.
"""

from __future__ import annotations

import io
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ingestor import RawPage

logger = logging.getLogger(__name__)


class OCRBackend(Enum):
    EASYOCR   = auto()
    PADDLEOCR = auto()
    TESSERACT = auto()


class OCREngine:
    """
    Fills in `raw_text` for pages that were flagged as scanned by PDFIngestor.

    Backend comparison for Arabic PDFs:

    Tesseract  — Default. Lightweight; fits Streamlit Cloud's 1 GB RAM limit.
                 Requires: pytesseract + system tesseract-ocr with arabic language pack.

    EasyOCR    — Good accuracy without extra setup. Handles Arabic RTL natively.
                 Requires: easyocr (~450 MB download on first run).  Local use only.

    PaddleOCR  — Best Arabic accuracy. Uses a dedicated PP-OCRv3-ar model.
                 Requires: paddlepaddle + paddleocr (~1 GB), Python ≤ 3.12.  Local only.

    Usage::

        engine = OCREngine(backend=OCRBackend.TESSERACT)
        pages  = engine.process_pages(ingestion_result.pages)
    """

    def __init__(
        self,
        backend: OCRBackend = OCRBackend.TESSERACT,
        gpu: bool = False,
        dpi: int = 300,
    ):
        self.backend = backend
        self.gpu = gpu
        self.dpi = dpi
        self._reader = None   # lazy-loaded

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_pages(self, pages: list[RawPage]) -> list[RawPage]:
        """
        Returns the same list with `raw_text` populated for scanned pages.
        Digital pages are passed through unchanged.
        """
        scanned = [p for p in pages if p.pdf_type == "scanned"]
        if not scanned:
            logger.info("No scanned pages — OCR skipped.")
            return pages

        logger.info("Running OCR on %d scanned page(s) via %s …", len(scanned), self.backend.name)
        self._lazy_init()

        ocr_fn = {
            OCRBackend.EASYOCR:   self._easyocr_page,
            OCRBackend.PADDLEOCR: self._paddleocr_page,
            OCRBackend.TESSERACT: self._tesseract_page,
        }[self.backend]

        for page in scanned:
            if page.image_bytes:
                page.raw_text = ocr_fn(page.image_bytes)
                logger.debug("Page %d OCR'd — %d chars", page.page_number, len(page.raw_text))
            else:
                logger.warning("Page %d marked scanned but has no image bytes — skipping.", page.page_number)

        return pages

    # ------------------------------------------------------------------ #
    #  Lazy initialisation                                                 #
    # ------------------------------------------------------------------ #

    def _lazy_init(self):
        if self._reader is not None:
            return

        if self.backend == OCRBackend.EASYOCR:
            try:
                import easyocr  # noqa: PLC0415
                # 'ar' = Arabic; also load 'en' so mixed pages work
                self._reader = easyocr.Reader(["ar", "en"], gpu=self.gpu)
                logger.info("EasyOCR reader initialised (gpu=%s).", self.gpu)
            except ImportError as exc:
                raise ImportError(
                    "EasyOCR not installed. Run: pip install easyocr"
                ) from exc

        elif self.backend == OCRBackend.PADDLEOCR:
            try:
                from paddleocr import PaddleOCR  # noqa: PLC0415
                # lang='ar'  → Arabic PP-OCRv3 recognition model
                # use_angle_cls=True  → auto-correct rotated text blocks
                # show_log=False  → suppress PaddlePaddle's verbose output
                self._reader = PaddleOCR(
                    use_angle_cls=True,
                    lang="ar",
                    use_gpu=self.gpu,
                    show_log=False,
                )
                logger.info("PaddleOCR reader initialised (gpu=%s).", self.gpu)
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR not installed. Run:\n"
                    "  pip install paddlepaddle paddleocr\n"
                    "For GPU: pip install paddlepaddle-gpu paddleocr"
                ) from exc

        else:  # TESSERACT
            try:
                import pytesseract  # noqa: PLC0415 (just verify it's available)
                self._reader = pytesseract
                logger.info("Tesseract reader initialised.")
            except ImportError as exc:
                raise ImportError(
                    "pytesseract not installed. Run: pip install pytesseract"
                ) from exc

    # ------------------------------------------------------------------ #
    #  Backend implementations                                            #
    # ------------------------------------------------------------------ #

    def _easyocr_page(self, image_bytes: bytes) -> str:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        results = self._reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(results)

    def _paddleocr_page(self, image_bytes: bytes) -> str:
        """
        Run PaddleOCR on a single page image.

        PaddleOCR result structure (per page):
            result[0]  — list of detected text blocks, each:
                [[tl, tr, br, bl],   ← quadrilateral bounding box (4 × [x, y])
                 [text, confidence]]

        Blocks are returned in approximate top-to-bottom, right-to-left order
        for Arabic text (PaddleOCR sorts by the top-left y of each box).
        We extract only the text strings, filter low-confidence hits, and join
        with newlines so downstream normalisation sees a clean line-per-block
        structure.
        """
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        result = self._reader.ocr(arr, cls=True)

        if not result or result[0] is None:
            return ""

        lines: list[str] = []
        for block in result[0]:
            # block = [bbox, [text, confidence]]
            text, conf = block[1]
            if conf >= 0.3 and text.strip():   # discard very low-confidence noise
                lines.append(text.strip())

        return "\n".join(lines)

    def _tesseract_page(self, image_bytes: bytes) -> str:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        # Pass a numpy array (not a PIL Image) so that pytesseract writes the
        # temp file without embedded DPI metadata.  PIL Images carry the
        # pdf2image render DPI (300 dpi) into the PNG pHYs chunk; Tesseract
        # then applies stricter layout-analysis thresholds and drops isolated
        # lines at page edges (attribution headers, last body lines).  A
        # numpy array has no DPI metadata, so Tesseract uses its 70-dpi
        # default and correctly segments the full page — matching the
        # behaviour of the reference notebook (optimized_approach.txt).
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        try:
            return self._reader.image_to_string(arr, lang="ara", config=f"--psm 3 --dpi {self.dpi}")
        except Exception as exc:
            logger.warning("Tesseract OCR failed on page: %s", exc)
            return ""
