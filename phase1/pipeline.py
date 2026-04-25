"""
Phase 1 — Pipeline
Top-level orchestrator. Captures a raw-text snapshot after extraction
(before normalisation/diacritization) so both versions are saved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .core.ingestor      import PDFIngestor
from .core.ocr_engine    import OCREngine, OCRBackend
from .core.ocr_corrector import OCRTextCorrector
from .core.page_stitcher import PageStitcher
from .core.normalizer    import ArabicTextNormalizer
from .core.chunker       import SemanticChunker, Chunk
from .core.output_writer import OutputWriter
from .core.summarizer    import BookSummarizer

logger = logging.getLogger(__name__)


@dataclass
class Phase1Config:
    ocr_gpu:        bool   = False
    ocr_backend:    str    = "tesseract"
    ocr_dpi:        int    = 300
    max_tokens:     int    = 1500
    overlap_tokens: int    = 200
    output_dir:     str    = "output"
    # "auto"    — let the ingestor decide (default)
    # "digital" — force PyMuPDF extraction for all pages
    # "ocr"     — force OCR for all pages regardless of content
    pdf_mode:       str    = "auto"
    # LLM OCR correction (scanned pages only)
    # Sends raw Tesseract output through Claude Haiku to fix OCR errors and
    # join line-wrapped text.  Requires anthropic_api_key.  ~$0.001/page.
    # LLM OCR correction and page stitching are OFF by default to keep Phase 1
    # focused on raw text extraction without spending API tokens.  Enable them
    # explicitly when higher-quality flowing text is needed.
    ocr_correction: bool   = False
    # Cross-page boundary stitching (scanned pages only)
    # Strips running headers/footers and joins sentences split across page
    # breaks.  Runs after OCR correction.  ~$0.0001/boundary.
    page_stitching: bool   = False
    # LLM summarization
    anthropic_api_key: str = ""
    script_genre:   str    = "non-fiction"   # hint for Scriptwriter tone
    # Optional book metadata injected into the formal presentation section
    book_author:    str    = ""   # e.g. "تحقيق وتقديم نجدة فتحي صفوة"
    book_pages:     int    = 0    # actual page count (0 = omit from script)
    book_structure: str    = ""   # e.g. "مقدمة و١٦ فصلاً وملاحق"


@dataclass
class Phase1Result:
    source_path:       str
    pdf_type:          str
    total_pages:       int
    chunks:            list[Chunk]
    json_path:         Path
    txt_path:          Path
    raw_txt_path:      Path
    script_path:       Path | None = None
    script_diac_path:  Path | None = None
    script_meta_path:  Path | None = None
    elapsed_sec:       float = 0.0
    warnings:          list[str] = field(default_factory=list)


class Phase1Pipeline:

    def __init__(
        self,
        config: Phase1Config | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ):
        self.cfg         = config or Phase1Config()
        self.on_progress = on_progress or (lambda s, p: None)

    def run(self, pdf_path: str | Path) -> Phase1Result:
        t0       = time.perf_counter()
        warnings = []

        # ── Step 1: Ingest ───────────────────────────────────────────── #
        self._progress("Ingesting PDF …", 0.0)
        ingestor  = PDFIngestor(dpi=self.cfg.ocr_dpi)
        ingestion = ingestor.ingest(pdf_path)
        logger.info("PDF type: %s (%d pages)", ingestion.pdf_type, ingestion.total_pages)

        # ── Mode override ─────────────────────────────────────────────── #
        if self.cfg.pdf_mode == "ocr":
            for p in ingestion.pages:
                p.pdf_type  = "scanned"
                p.raw_text  = ""       # discard digital extraction; OCR will fill it
            ingestion.pdf_type = "scanned"
            logger.info("pdf_mode=ocr — all pages forced to OCR.")
        elif self.cfg.pdf_mode == "digital":
            for p in ingestion.pages:
                p.pdf_type = "digital"
            ingestion.pdf_type = "digital"
            logger.info("pdf_mode=digital — all pages forced to digital extraction.")

        # ── Step 2: OCR (scanned pages only) ─────────────────────────── #
        self._progress("Running OCR on scanned pages …", 0.18)
        has_scanned = any(p.pdf_type == "scanned" for p in ingestion.pages)
        if has_scanned:
            _backend_map = {
                "easyocr":   OCRBackend.EASYOCR,
                "paddleocr": OCRBackend.PADDLEOCR,
                "tesseract": OCRBackend.TESSERACT,
            }
            if self.cfg.ocr_backend not in _backend_map:
                warnings.append(
                    f"Unknown OCR backend '{self.cfg.ocr_backend}' — defaulting to Tesseract."
                )
                logger.warning(warnings[-1])
            backend = _backend_map.get(self.cfg.ocr_backend, OCRBackend.TESSERACT)
            ocr = OCREngine(backend=backend, gpu=self.cfg.ocr_gpu, dpi=self.cfg.ocr_dpi)
            try:
                ingestion.pages = ocr.process_pages(ingestion.pages)
            except ImportError as exc:
                warnings.append(f"OCR skipped — library missing: {exc}")
                logger.warning(warnings[-1])

        # ── Snapshot: capture raw text BEFORE any text processing ─────── #
        # For scanned pages the OCR output is the raw baseline;
        # for digital pages it was already stored in raw_text_pre by ingestor.
        # Here we ensure scanned pages also get their pre-norm snapshot.
        for page in ingestion.pages:
            if page.pdf_type == "scanned" and not page.raw_text_pre:
                page.raw_text_pre = page.raw_text   # OCR output = raw baseline

        # ── Step 2b: LLM OCR correction (optional, scanned pages only) ── #
        # Sends raw Tesseract output through Claude Haiku to fix OCR errors
        # and join line-wrapped text into flowing paragraphs.  Runs only when
        # an API key is present and ocr_correction is enabled.
        if self.cfg.ocr_correction and self.cfg.anthropic_api_key and has_scanned:
            self._progress("LLM OCR correction …", 0.28)
            try:
                corrector = OCRTextCorrector(
                    api_key     = self.cfg.anthropic_api_key,
                    on_progress = self.on_progress,
                )
                ingestion.pages = corrector.correct_pages(ingestion.pages)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"LLM OCR correction failed: {exc}")
                logger.exception("LLM OCR correction failed")

        # ── Step 2c: Cross-page boundary stitching (scanned pages only) ─ #
        # Strips running headers/footers and joins sentences split across
        # page breaks.  Runs after OCR correction so the LLM sees clean text.
        if self.cfg.page_stitching and self.cfg.anthropic_api_key and has_scanned:
            self._progress("Stitching page boundaries …", 0.32)
            try:
                stitcher = PageStitcher(
                    api_key     = self.cfg.anthropic_api_key,
                    on_progress = self.on_progress,
                )
                ingestion.pages = stitcher.stitch_pages(ingestion.pages)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Page stitching failed: {exc}")
                logger.exception("Page stitching failed")

        # ── Step 3: Normalise ─────────────────────────────────────────── #
        self._progress("Normalising Arabic text …", 0.35)
        normalizer = ArabicTextNormalizer()
        ingestion.pages = normalizer.normalize_pages(ingestion.pages)

        # ── Step 4: Chunk ─────────────────────────────────────────────── #
        self._progress("Chunking text …", 0.50)
        chunker = SemanticChunker(
            max_tokens     = self.cfg.max_tokens,
            overlap_tokens = self.cfg.overlap_tokens,
        )
        chunks = chunker.chunk_pages(ingestion.pages)

        # ── Step 5: Write extraction output ──────────────────────────── #
        self._progress("Writing extraction output …", 0.62)
        writer = OutputWriter(output_dir=self.cfg.output_dir)
        json_path, txt_path, raw_txt_path = writer.write(ingestion, chunks)

        # ── Step 6: Summarize → Script → Diacritize ───────────────────── #
        script_path = script_diac_path = script_meta_path = None
        if self.cfg.anthropic_api_key and chunks:
            try:
                self._progress("Summarising book (Reader + Consolidator) …", 0.70)
                summarizer = BookSummarizer(
                    api_key        = self.cfg.anthropic_api_key,
                    genre          = self.cfg.script_genre,
                    output_dir     = self.cfg.output_dir,
                    book_author    = self.cfg.book_author,
                    book_pages     = self.cfg.book_pages,
                    book_structure = self.cfg.book_structure,
                )
                self._progress("Writing script (Scriptwriter) …", 0.82)
                script_path, script_diac_path, script_meta_path = summarizer.run(
                    chunks       = chunks,
                    title        = ingestion.metadata.get("title", ""),
                    on_progress  = self._progress,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Summarization failed: {exc}")
                logger.exception("Summarization failed")
        elif not self.cfg.anthropic_api_key:
            warnings.append("No Anthropic API key — script generation skipped.")
            logger.info(warnings[-1])

        elapsed = time.perf_counter() - t0
        self._progress("Done ✓", 1.0)
        logger.info("Phase 1 complete in %.1fs — %d chunks.", elapsed, len(chunks))

        return Phase1Result(
            source_path       = str(pdf_path),
            pdf_type          = ingestion.pdf_type,
            total_pages       = ingestion.total_pages,
            chunks            = chunks,
            json_path         = json_path,
            txt_path          = txt_path,
            raw_txt_path      = raw_txt_path,
            script_path       = script_path,
            script_diac_path  = script_diac_path,
            script_meta_path  = script_meta_path,
            elapsed_sec       = elapsed,
            warnings          = warnings,
        )

    def _progress(self, step: str, pct: float):
        logger.debug("[%.0f%%] %s", pct * 100, step)
        self.on_progress(step, pct)
