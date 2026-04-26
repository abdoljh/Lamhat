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
import re
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
    #  tessdata_best upgrade                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _upgrade_ara_tessdata() -> None:
        """
        Replace the standard apt-installed ara.traineddata with the
        tessdata_best model from the Tesseract project.  The 'best' model
        is trained on more data and captures edge-of-page lines that the
        standard model misses.

        Runs silently once per environment; logs a message either way.
        Fails gracefully — the standard model is kept if the download fails.
        """
        import os
        import subprocess
        import urllib.request
        from pathlib import Path

        _URL = (
            "https://github.com/tesseract-ocr/tessdata_best"
            "/raw/main/ara.traineddata"
        )
        _MARKER = "tessdata_best"   # written into first line of downloaded file

        # Locate tessdata directory from tesseract binary
        try:
            out = subprocess.run(
                ["tesseract", "--print-parameters"],
                capture_output=True, text=True, timeout=10,
            )
            # tesseract --list-langs also reveals the tessdata path
            out2 = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True, text=True, timeout=10,
            )
            # Extract path from stderr: "TESSDATA_PREFIX = /usr/share/..."
            tessdata_dir: str | None = None
            for line in (out.stderr + out2.stderr).splitlines():
                if "TESSDATA_PREFIX" in line or "tessdata" in line.lower():
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        candidate = parts[1].strip().rstrip("/")
                        if Path(candidate).is_dir():
                            tessdata_dir = candidate
                            break
                    elif "tessdata" in line:
                        for tok in line.split():
                            if "tessdata" in tok and Path(tok.rstrip(":")).is_dir():
                                tessdata_dir = tok.rstrip(":")
                                break
        except Exception:
            tessdata_dir = None

        # Fallback search
        if not tessdata_dir:
            for candidate in [
                os.environ.get("TESSDATA_PREFIX", ""),
                "/usr/share/tesseract-ocr/5/tessdata",
                "/usr/share/tesseract-ocr/4.00/tessdata",
                "/usr/local/share/tessdata",
            ]:
                if candidate and Path(candidate).is_dir():
                    tessdata_dir = candidate
                    break

        if not tessdata_dir:
            logger.debug("tessdata dir not found — skipping tessdata_best upgrade.")
            return

        target = Path(tessdata_dir) / "ara.traineddata"

        # Check if already upgraded (first 64 bytes contain our marker URL)
        if target.exists():
            try:
                with open(target, "rb") as fh:
                    header = fh.read(200).decode("latin-1", errors="ignore")
                if _MARKER in header:
                    logger.debug("tessdata_best Arabic model already installed.")
                    return
            except Exception:
                pass  # unreadable header — proceed with download

        # Download
        logger.info("Downloading tessdata_best Arabic model → %s …", target)
        try:
            tmp = target.with_suffix(".tmp")
            urllib.request.urlretrieve(_URL, tmp)
            tmp.replace(target)
            logger.info("tessdata_best Arabic model installed successfully.")
        except Exception as exc:
            logger.warning(
                "tessdata_best download failed (%s) — using standard model.", exc
            )
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

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
                self._upgrade_ara_tessdata()
                logger.info("Tesseract reader initialised.")
            except ImportError as exc:
                raise ImportError(
                    "pytesseract not installed. Run: pip install pytesseract"
                ) from exc

    # ------------------------------------------------------------------ #
    #  Backend implementations + OCR text quality helpers                #
    # ------------------------------------------------------------------ #

    _ARA_WORD_RE = re.compile(r'[؀-ۿ]{3,}')

    @classmethod
    def _is_valid_rescue(cls, text: str) -> bool:
        """
        True if rescue OCR text is substantive Arabic content.

        Requires at least 2 Arabic sequences of ≥ 3 chars.  This rejects
        garbled fragments like ``سي لق ضبن نه`` (only one 3-char Arabic run)
        while accepting real attribution lines like ``نجدة فتحى صفوة``.
        """
        return len(cls._ARA_WORD_RE.findall(text)) >= 2

    @classmethod
    def _filter_ocr_garbage(cls, text: str) -> str:
        """
        Remove lines that contain no Arabic word of ≥ 3 chars.

        PSM 4 sometimes picks up decorative page elements (rules, ornaments,
        stray punctuation) as single-character or digit-only lines.  Any line
        lacking at least one Arabic sequence of ≥ 3 chars is discarded.
        Empty lines (paragraph breaks) are always preserved.
        """
        return '\n'.join(
            line for line in text.split('\n')
            if not line.strip() or cls._ARA_WORD_RE.search(line)
        )

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

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        try:
            # Full-page pass: PSM 4 (single column, variable font sizes) fits
            # Arabic book pages and preserves lines near the page edges better
            # than PSM 3 (auto-layout) whose region-scoring can drop isolated
            # short lines at the top/bottom margins.
            body = self._reader.image_to_string(arr, lang="ara", config="--psm 4")
            body = self._filter_ocr_garbage(body)

            h = arr.shape[0]

            # ── Header rescue pass ────────────────────────────────────────
            # Crop the top 8 % of the image (one text-line height at 300 Dpi)
            # and run a second pass with PSM 7 (single text line).
            # PSM 7 is more reliable than PSM 11 for a narrow one-line strip:
            # it produces a single clean string rather than fragmented tokens.
            # The result is only accepted when it contains ≥ 2 Arabic words of
            # ≥ 3 chars — this blocks garbled noise such as "سي لق ضبن نه"
            # (only one qualifying run) while admitting "نجدة فتحى صفوة".
            top_crop = arr[: max(1, int(h * 0.08)), :]
            header = self._reader.image_to_string(
                top_crop, lang="ara", config="--psm 7"
            ).strip()
            if self._is_valid_rescue(header) and header not in body:
                body = header + "\n" + body

            # ── Footer rescue pass ────────────────────────────────────────
            # PSM 11 (sparse text) on the bottom 12 % catches partial last
            # lines that PSM 4 sometimes clips at the lower margin.
            # Garbage-filter and quality-gate applied before appending.
            bottom_crop = arr[int(h * 0.88):, :]
            footer = self._reader.image_to_string(
                bottom_crop, lang="ara", config="--psm 11"
            ).strip()
            footer = self._filter_ocr_garbage(footer)
            if self._is_valid_rescue(footer) and footer not in body:
                body = body + "\n" + footer

            return body
        except Exception as exc:
            logger.warning("Tesseract OCR failed on page: %s", exc)
            return ""
