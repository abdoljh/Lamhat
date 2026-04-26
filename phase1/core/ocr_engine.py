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
                 Downloads tessdata_best Arabic model on first run (~14 MB).

    EasyOCR    — Good accuracy without extra setup. Handles Arabic RTL natively.
                 Requires: easyocr (~450 MB download on first run).  Local use only.

    PaddleOCR  — Best Arabic accuracy. Uses a dedicated PP-OCRv3-ar model.
                 Requires: paddlepaddle + paddleocr (~1 GB), Python ≤ 3.12.  Local only.

    Usage::

        engine = OCREngine(backend=OCRBackend.TESSERACT)
        pages  = engine.process_pages(ingestion_result.pages)
    """

    # White padding (px) added around the page image before OCR.
    # Prevents Tesseract's layout analysis from clipping text at the physical
    # page edges, which it treats as image boundaries.
    _BORDER_PX: int = 100

    # Top-of-page zone (mm) treated as a separate header strip.
    # Processed with PSM 6 + confidence filtering instead of PSM 4.
    _HEADER_MM: float = 20.0

    # Minimum Tesseract word-level confidence (0–100) to keep a word from the
    # header strip.  Decorative elements / noise score < 65; real text ≥ 65.
    _HEADER_CONF: int = 65

    # Compiled regex for Arabic word detection (used by _filter_ocr_garbage).
    _ARA_WORD_RE = re.compile(r'[؀-ۿ]{3,}')

    def __init__(
        self,
        backend: OCRBackend = OCRBackend.TESSERACT,
        gpu: bool = False,
        dpi: int = 400,
    ):
        self.backend = backend
        self.gpu = gpu
        self.dpi = dpi
        self._reader      = None   # lazy-loaded
        self._tessdata_dir = ""    # path to ~/.tessdata_custom if download succeeded
        self._lang         = "ara"

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
    #  tessdata_best download                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _download_tessdata_best() -> str:
        """
        Download the tessdata_best Arabic model to ``~/.tessdata_custom/``.

        Returns the directory path so callers can pass ``--tessdata-dir``
        to Tesseract.  Returns ``""`` if the download fails (falls back to
        the system-installed standard model).

        The user home directory is always writable, unlike the system
        tessdata prefix which requires root on most Linux installations.
        Cached after the first successful download.
        """
        import urllib.request
        from pathlib import Path

        _URL = (
            "https://github.com/tesseract-ocr/tessdata_best"
            "/raw/main/ara.traineddata"
        )
        cache = Path.home() / ".tessdata_custom"
        dest  = cache / "ara.traineddata"

        if dest.exists():
            logger.debug("tessdata_best Arabic model already at %s.", dest)
            return str(cache)

        try:
            cache.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            logger.info("Downloading tessdata_best Arabic model → %s …", dest)
            urllib.request.urlretrieve(_URL, tmp)
            tmp.replace(dest)
            logger.info("tessdata_best Arabic model installed at %s.", dest)
            return str(cache)
        except Exception as exc:
            logger.warning(
                "tessdata_best download failed (%s) — using system model.", exc
            )
            try:
                dest.with_suffix(".tmp").unlink(missing_ok=True)
            except Exception:
                pass
            return ""

    # ------------------------------------------------------------------ #
    #  Lazy initialisation                                                 #
    # ------------------------------------------------------------------ #

    def _lazy_init(self) -> None:
        if self._reader is not None:
            return

        if self.backend == OCRBackend.EASYOCR:
            try:
                import easyocr  # noqa: PLC0415
                self._reader = easyocr.Reader(["ar", "en"], gpu=self.gpu)
                logger.info("EasyOCR reader initialised (gpu=%s).", self.gpu)
            except ImportError as exc:
                raise ImportError(
                    "EasyOCR not installed. Run: pip install easyocr"
                ) from exc

        elif self.backend == OCRBackend.PADDLEOCR:
            try:
                from paddleocr import PaddleOCR  # noqa: PLC0415
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
                import pytesseract  # noqa: PLC0415
                self._reader       = pytesseract
                self._tessdata_dir = self._download_tessdata_best()
                self._lang         = "ara"
                logger.info(
                    "Tesseract reader initialised (tessdata=%s).",
                    self._tessdata_dir or "system",
                )
            except ImportError as exc:
                raise ImportError(
                    "pytesseract not installed. Run: pip install pytesseract"
                ) from exc

    # ------------------------------------------------------------------ #
    #  Tesseract helpers                                                   #
    # ------------------------------------------------------------------ #

    def _tess_config(self, psm: int) -> str:
        """
        Build a Tesseract config string: OEM 1 (LSTM only) + given PSM +
        ``--tessdata-dir`` when tessdata_best was downloaded successfully.
        """
        parts = ["--oem 1", f"--psm {psm}"]
        if self._tessdata_dir:
            parts.append(f"--tessdata-dir {self._tessdata_dir}")
        return " ".join(parts)

    def _header_ocr(self, strip) -> str:
        """
        OCR a narrow header strip using ``image_to_data`` and return only
        the words whose Tesseract confidence ≥ _HEADER_CONF.

        This separates real header text (high confidence) from decorative
        noise and faint artefacts (low confidence) even when both appear in
        the same strip — something a simple ratio or count filter cannot do.
        Words are grouped by (block, paragraph, line) so RTL line order is
        preserved.
        """
        import pytesseract  # noqa: PLC0415
        try:
            data = pytesseract.image_to_data(
                strip,
                lang=self._lang,
                config=self._tess_config(psm=6),
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            return ""

        line_words: dict[tuple, list[str]] = {}
        for word, conf, block, par, line in zip(
            data["text"],  data["conf"],
            data["block_num"], data["par_num"], data["line_num"],
        ):
            if word.strip() and int(conf) >= self._HEADER_CONF:
                line_words.setdefault((block, par, line), []).append(word)
        return "\n".join(" ".join(words) for words in line_words.values())

    @classmethod
    def _filter_ocr_garbage(cls, text: str) -> str:
        """
        Remove lines that are not predominantly Arabic.

        A non-empty line is kept only when it contains at least one Arabic
        sequence of ≥ 3 chars AND Arabic chars form > 75 % of its
        non-whitespace content.  Empty lines (paragraph separators) are
        always preserved.
        """
        clean = []
        for line in text.split('\n'):
            s = line.strip()
            if not s:
                clean.append(line)
                continue
            if cls._ARA_WORD_RE.search(s) and cls._arabic_ratio(s) > 0.75:
                clean.append(line)
        return '\n'.join(clean)

    @classmethod
    def _arabic_ratio(cls, text: str) -> float:
        """Fraction of non-whitespace chars in the Arabic Unicode block (U+0600–U+06FF)."""
        non_ws = text.replace(' ', '').replace('\n', '').replace('\t', '')
        if not non_ws:
            return 0.0
        arabic = sum(1 for c in non_ws if '؀' <= c <= 'ۿ')
        return arabic / len(non_ws)

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
        for Arabic text.  We extract only the text strings, filter low-confidence
        hits, and join with newlines.
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
            text, conf = block[1]
            if conf >= 0.3 and text.strip():
                lines.append(text.strip())
        return "\n".join(lines)

    def _tesseract_page(self, image_bytes: bytes) -> str:
        """
        Two-pass Tesseract OCR for a single scanned Arabic page.

        Pass 1 — Header strip (top 20 mm of content + 100 px border):
            PSM 6 + ``image_to_data`` with confidence filtering ≥ 65.
            Isolates real header text (attribution lines, running titles)
            from decorative noise without brittle ratio heuristics.

        Pass 2 — Body (everything below the header strip):
            PSM 4 (single column, variable font sizes) + ``image_to_string``.
            Applied to the cropped sub-image so layout analysis is not
            confused by the attribution region.

        Preprocessing applied to the full image before either pass:
            • Grayscale conversion (removes colour noise).
            • 2× contrast enhancement (makes faint near-margin text readable).
            • 100 px white border (prevents edge-text clipping).

        Both passes use OEM 1 (LSTM neural network only) and the
        tessdata_best Arabic model when available (downloaded once to
        ``~/.tessdata_custom/`` and referenced via ``--tessdata-dir``).
        """
        from PIL import Image, ImageEnhance, ImageOps  # noqa: PLC0415

        # Preprocess: grayscale → 2× contrast → white border
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        padded = ImageOps.expand(img, border=self._BORDER_PX, fill=255)

        try:
            # Height (px) of the header strip: border + top 20 mm of content
            strip_h = self._BORDER_PX + int(self._HEADER_MM / 25.4 * self.dpi)
            header_strip = padded.crop((0, 0, padded.width, strip_h))
            header = self._header_ocr(header_strip)

            body_img = padded.crop((0, strip_h, padded.width, padded.height))
            body = self._reader.image_to_string(
                body_img, lang=self._lang, config=self._tess_config(psm=4)
            )
            body = self._filter_ocr_garbage(body)

            return (header + "\n" + body) if header else body

        except Exception as exc:
            logger.warning("Tesseract OCR failed on page: %s", exc)
            return ""
