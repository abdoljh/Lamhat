"""
Phase 1 — Pipeline
Split into two independently runnable stages:

  Phase1aPipeline  — Steps 1–3: Ingest → OCR → LLM correction → Normalise
    Saves:
      *_phase1a_corrected.txt   LLM-corrected OCR text, per page (pre-normalisation)
      *_phase1a_normalized.txt  After Arabic normalisation, per page (human-readable)
      *_phase1a.json            Structured page data consumed by Phase 1b

  Phase1bPipeline  — Steps 4–6: Chunk → Write outputs → Summarise
    Accepts:  Phase1aResult (in-memory)  OR  path to *_phase1a.json

  Phase1Pipeline   — Combined 1a + 1b in a single call (backward-compatible).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .core.ingestor      import PDFIngestor, RawPage, IngestionResult
from .core.ocr_engine    import OCREngine, OCRBackend
from .core.ocr_corrector import OCRTextCorrector
from .core.page_stitcher import PageStitcher
from .core.normalizer    import ArabicTextNormalizer
from .core.chunker       import SemanticChunker, Chunk
from .core.output_writer import OutputWriter
from .core.summarizer    import BookSummarizer

logger = logging.getLogger(__name__)

_PAGE_SEP = "\n" + "═" * 60 + "\n"


# ──────────────────────────────────────────────────────────────────────────── #
#  Shared configuration                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class Phase1Config:
    ocr_gpu:        bool   = False
    ocr_backend:    str    = "tesseract"
    ocr_dpi:        int    = 400
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
    ocr_correction: bool   = False
    # Cross-page boundary stitching (scanned pages only)
    # Strips running headers/footers and joins sentences split across page
    # breaks.  Runs after OCR correction.  ~$0.0001/boundary.
    page_stitching: bool   = False
    # LLM summarization + OCR correction
    anthropic_api_key: str = ""
    script_genre:   str    = "non-fiction"   # hint for Scriptwriter tone
    # Optional book metadata injected into the formal presentation section
    book_author:    str    = ""   # e.g. "تحقيق وتقديم نجدة فتحي صفوة"
    book_pages:     int    = 0    # actual page count (0 = omit from script)
    book_structure: str    = ""   # e.g. "مقدمة و١٦ فصلاً وملاحق"
    # Diacritisation (Mishkal) of the final script.
    # When False, the diacritized script file is not written and _script_diacritized.txt
    # is omitted.  Set to False when you don't need Mishkal output to save ~2 s per run.
    diacritize:     bool   = True


# ──────────────────────────────────────────────────────────────────────────── #
#  Result types                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class Phase1aResult:
    """Output of Phase1aPipeline (steps 1–3)."""
    source_path:          str
    pdf_type:             str
    total_pages:          int
    metadata:             dict
    # Serialisable page records (no image bytes).
    # raw_text     = normalised text (ready for chunking)
    # raw_text_pre = original OCR output (for audit / raw snapshot)
    pages:                list[dict]
    corrected_txt_path:   Path
    normalized_txt_path:  Path
    normalized_json_path: Path
    elapsed_sec:          float = 0.0
    warnings:             list[str] = field(default_factory=list)


@dataclass
class Phase1Result:
    """Output of Phase1bPipeline (steps 4–6) or the combined Phase1Pipeline."""
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


# ──────────────────────────────────────────────────────────────────────────── #
#  Phase 1a — Ingest → OCR → LLM correction → Normalise                        #
# ──────────────────────────────────────────────────────────────────────────── #

class Phase1aPipeline:
    """
    Runs the extraction half of Phase 1 and saves three files:

      *_phase1a_corrected.txt   — OCR output after LLM correction, before normalisation
      *_phase1a_normalized.txt  — text after Arabic normalisation (human-readable, per page)
      *_phase1a.json            — structured page data for Phase1bPipeline to load

    Use this when you want to inspect or iterate on OCR / normalisation quality
    independently of the chunking and summarisation steps.
    """

    def __init__(
        self,
        config: Phase1Config | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ):
        self.cfg         = config or Phase1Config()
        self.on_progress = on_progress or (lambda s, p: None)

    def run(self, pdf_path: str | Path) -> Phase1aResult:
        t0       = time.perf_counter()
        warnings: list[str] = []
        pdf_path = Path(pdf_path)
        stem     = pdf_path.stem
        out_dir  = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Ingest ───────────────────────────────────────────── #
        self._progress("Ingesting PDF …", 0.0)
        ingestor  = PDFIngestor(dpi=self.cfg.ocr_dpi)
        ingestion = ingestor.ingest(pdf_path)
        logger.info("PDF type: %s (%d pages)", ingestion.pdf_type, ingestion.total_pages)

        # ── Mode override ─────────────────────────────────────────────── #
        if self.cfg.pdf_mode == "ocr":
            for p in ingestion.pages:
                p.pdf_type = "scanned"
                p.raw_text = ""
            ingestion.pdf_type = "scanned"
            logger.info("pdf_mode=ocr — all pages forced to OCR.")
        elif self.cfg.pdf_mode == "digital":
            for p in ingestion.pages:
                p.pdf_type = "digital"
            ingestion.pdf_type = "digital"
            logger.info("pdf_mode=digital — all pages forced to digital extraction.")

        # ── Step 2: OCR (scanned pages only) ─────────────────────────── #
        self._progress("Running OCR on scanned pages …", 0.10)
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
            backend = _backend_map.get(self.cfg.ocr_backend, OCRBackend.TESSERACT)
            ocr = OCREngine(backend=backend, gpu=self.cfg.ocr_gpu, dpi=self.cfg.ocr_dpi)
            try:
                ingestion.pages = ocr.process_pages(ingestion.pages)
            except ImportError as exc:
                warnings.append(f"OCR skipped — library missing: {exc}")
                logger.warning(warnings[-1])

        # Snapshot: capture Tesseract output BEFORE any LLM processing.
        for page in ingestion.pages:
            if page.pdf_type == "scanned" and not page.raw_text_pre:
                page.raw_text_pre = page.raw_text

        # ── Step 2b: LLM OCR correction (optional) ───────────────────── #
        if self.cfg.ocr_correction and self.cfg.anthropic_api_key and has_scanned:
            self._progress("LLM OCR correction …", 0.30)
            try:
                corrector = OCRTextCorrector(
                    api_key     = self.cfg.anthropic_api_key,
                    on_progress = self.on_progress,
                )
                ingestion.pages = corrector.correct_pages(ingestion.pages)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"LLM OCR correction failed: {exc}")
                logger.exception("LLM OCR correction failed")

        # ── Step 2c: Cross-page boundary stitching (optional) ────────── #
        if self.cfg.page_stitching and self.cfg.anthropic_api_key and has_scanned:
            self._progress("Stitching page boundaries …", 0.42)
            try:
                stitcher = PageStitcher(
                    api_key     = self.cfg.anthropic_api_key,
                    on_progress = self.on_progress,
                )
                ingestion.pages = stitcher.stitch_pages(ingestion.pages)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Page stitching failed: {exc}")
                logger.exception("Page stitching failed")

        # Save corrected text (post-OCR-correction, pre-normalisation).
        corrected_txt_path = out_dir / f"{stem}_phase1a_corrected.txt"
        self._write_corrected(ingestion, corrected_txt_path)
        logger.info("Phase 1a corrected text → %s", corrected_txt_path)

        # ── Step 3: Normalise ─────────────────────────────────────────── #
        self._progress("Normalising Arabic text …", 0.60)
        normalizer = ArabicTextNormalizer()
        ingestion.pages = normalizer.normalize_pages(ingestion.pages)

        # Save normalised text (human-readable) and structured JSON.
        normalized_txt_path  = out_dir / f"{stem}_phase1a_normalized.txt"
        normalized_json_path = out_dir / f"{stem}_phase1a.json"
        pages_data = _serialize_pages(ingestion)
        self._write_normalized_txt(ingestion, normalized_txt_path)
        self._write_normalized_json(ingestion, pages_data, normalized_json_path)
        logger.info(
            "Phase 1a output → corrected: %s | normalized: %s | json: %s",
            corrected_txt_path, normalized_txt_path, normalized_json_path,
        )

        elapsed = time.perf_counter() - t0
        self._progress("Phase 1a done ✓", 1.0)
        logger.info("Phase 1a complete in %.1fs — %d pages.", elapsed, ingestion.total_pages)

        return Phase1aResult(
            source_path          = str(pdf_path),
            pdf_type             = ingestion.pdf_type,
            total_pages          = ingestion.total_pages,
            metadata             = ingestion.metadata,
            pages                = pages_data,
            corrected_txt_path   = corrected_txt_path,
            normalized_txt_path  = normalized_txt_path,
            normalized_json_path = normalized_json_path,
            elapsed_sec          = elapsed,
            warnings             = warnings,
        )

    # ── File writers ──────────────────────────────────────────────────── #

    @staticmethod
    def _write_corrected(ingestion: IngestionResult, path: Path) -> None:
        src   = Path(ingestion.source_path).name
        lines = [
            f"# Phase 1a OCR-CORRECTED — {src}",
            f"# PDF Type   : {ingestion.pdf_type}",
            f"# Total pages: {ingestion.total_pages}",
            "# NOTE: text after LLM OCR correction, BEFORE Arabic normalisation.",
            "# When ocr_correction=False this is identical to the raw Tesseract output.",
            "",
        ]
        for page in ingestion.pages:
            lines.append(f"[Page {page.page_number:03d} | {page.pdf_type}]")
            lines.append(page.raw_text or "(empty)")
            lines.append(_PAGE_SEP)
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _write_normalized_txt(ingestion: IngestionResult, path: Path) -> None:
        blocks = [p.raw_text.strip() for p in ingestion.pages if p.raw_text.strip()]
        path.write_text("\n\n".join(blocks), encoding="utf-8")

    @staticmethod
    def _write_normalized_json(
        ingestion:  IngestionResult,
        pages_data: list[dict],
        path:       Path,
    ) -> None:
        payload = {
            "source":      ingestion.source_path,
            "pdf_type":    ingestion.pdf_type,
            "total_pages": ingestion.total_pages,
            "metadata":    ingestion.metadata,
            "pages":       pages_data,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _progress(self, step: str, pct: float) -> None:
        logger.debug("[%.0f%%] %s", pct * 100, step)
        self.on_progress(step, pct)


# ──────────────────────────────────────────────────────────────────────────── #
#  Phase 1b — Chunk → Write outputs → Summarise                                #
# ──────────────────────────────────────────────────────────────────────────── #

class Phase1bPipeline:
    """
    Runs the summarisation half of Phase 1.

    Accepts either:
      • a Phase1aResult object returned by Phase1aPipeline.run()
      • a path (str or Path) to a *_phase1a.json file written by Phase1aPipeline

    This lets you run Phase 1a once on a short sample, inspect the normalised
    text, and then run Phase 1b repeatedly (with different chunking/summarisation
    settings) without repeating OCR and normalisation.
    """

    def __init__(
        self,
        config: Phase1Config | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ):
        self.cfg         = config or Phase1Config()
        self.on_progress = on_progress or (lambda s, p: None)

    def run(self, source: Phase1aResult | str | Path) -> Phase1Result:
        t0       = time.perf_counter()
        warnings: list[str] = []

        # Accept a file path as well as an in-memory result.
        if isinstance(source, (str, Path)):
            source = _load_phase1a_json(Path(source))

        warnings.extend(source.warnings)
        ingestion = _reconstruct_ingestion(source)

        # ── Step 4: Chunk ─────────────────────────────────────────────── #
        self._progress("Chunking text …", 0.10)
        chunker = SemanticChunker(
            max_tokens     = self.cfg.max_tokens,
            overlap_tokens = self.cfg.overlap_tokens,
        )
        chunks = chunker.chunk_pages(ingestion.pages)

        # ── Step 5: Write extraction output ──────────────────────────── #
        self._progress("Writing extraction output …", 0.40)
        writer = OutputWriter(output_dir=self.cfg.output_dir)
        json_path, txt_path, raw_txt_path = writer.write(ingestion, chunks)

        # ── Step 6: Summarize → Script → Diacritize ───────────────────── #
        script_path = script_diac_path = script_meta_path = None
        if self.cfg.anthropic_api_key and chunks:
            try:
                self._progress("Summarising book (Reader + Consolidator) …", 0.55)
                summarizer = BookSummarizer(
                    api_key        = self.cfg.anthropic_api_key,
                    genre          = self.cfg.script_genre,
                    output_dir     = self.cfg.output_dir,
                    book_author    = self.cfg.book_author,
                    book_pages     = self.cfg.book_pages,
                    book_structure = self.cfg.book_structure,
                    diacritize     = self.cfg.diacritize,
                )
                self._progress("Writing script (Scriptwriter) …", 0.80)
                script_path, script_diac_path, script_meta_path = summarizer.run(
                    chunks      = chunks,
                    title       = ingestion.metadata.get("title", ""),
                    on_progress = self._progress,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Summarization failed: {exc}")
                logger.exception("Summarization failed")
        elif not self.cfg.anthropic_api_key:
            warnings.append("No Anthropic API key — script generation skipped.")
            logger.info(warnings[-1])

        elapsed = time.perf_counter() - t0
        self._progress("Phase 1b done ✓", 1.0)
        logger.info("Phase 1b complete in %.1fs — %d chunks.", elapsed, len(chunks))

        return Phase1Result(
            source_path      = source.source_path,
            pdf_type         = source.pdf_type,
            total_pages      = source.total_pages,
            chunks           = chunks,
            json_path        = json_path,
            txt_path         = txt_path,
            raw_txt_path     = raw_txt_path,
            script_path      = script_path,
            script_diac_path = script_diac_path,
            script_meta_path = script_meta_path,
            elapsed_sec      = elapsed,
            warnings         = warnings,
        )

    def _progress(self, step: str, pct: float) -> None:
        logger.debug("[%.0f%%] %s", pct * 100, step)
        self.on_progress(step, pct)


# ──────────────────────────────────────────────────────────────────────────── #
#  Combined pipeline (backward-compatible)                                      #
# ──────────────────────────────────────────────────────────────────────────── #

class Phase1Pipeline:
    """
    Runs Phase1aPipeline then Phase1bPipeline in one call.
    Kept for backward compatibility — the Streamlit UI uses this class directly.
    Also writes the Phase 1a intermediate files so they are available for
    inspection even when the full pipeline is used.
    """

    def __init__(
        self,
        config: Phase1Config | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ):
        self.cfg         = config or Phase1Config()
        self.on_progress = on_progress or (lambda s, p: None)

    def run(self, pdf_path: str | Path) -> Phase1Result:
        def _a_progress(step: str, pct: float) -> None:
            self.on_progress(step, pct * 0.50)

        def _b_progress(step: str, pct: float) -> None:
            self.on_progress(step, 0.50 + pct * 0.50)

        result_a = Phase1aPipeline(config=self.cfg, on_progress=_a_progress).run(pdf_path)
        return Phase1bPipeline(config=self.cfg, on_progress=_b_progress).run(result_a)


# ──────────────────────────────────────────────────────────────────────────── #
#  Module-level helpers                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

def _serialize_pages(ingestion: IngestionResult) -> list[dict]:
    """Convert RawPage objects to JSON-serialisable dicts (no image bytes)."""
    return [
        {
            "page_number":  p.page_number,
            "pdf_type":     p.pdf_type,
            "raw_text":     p.raw_text,       # normalised — used by Phase 1b
            "raw_text_pre": p.raw_text_pre,   # original OCR — for raw snapshot
        }
        for p in ingestion.pages
    ]


def _load_phase1a_json(json_path: Path) -> Phase1aResult:
    """Reconstruct a Phase1aResult from a *_phase1a.json file."""
    data    = json.loads(json_path.read_text(encoding="utf-8"))
    out_dir = json_path.parent
    stem    = json_path.stem[: -len("_phase1a")]   # "pages_5_9_phase1a" → "pages_5_9"
    return Phase1aResult(
        source_path          = data["source"],
        pdf_type             = data["pdf_type"],
        total_pages          = data["total_pages"],
        metadata             = data["metadata"],
        pages                = data["pages"],
        corrected_txt_path   = out_dir / f"{stem}_phase1a_corrected.txt",
        normalized_txt_path  = out_dir / f"{stem}_phase1a_normalized.txt",
        normalized_json_path = json_path,
    )


def _reconstruct_ingestion(result: Phase1aResult) -> IngestionResult:
    """Rebuild an IngestionResult from serialised page dicts."""
    pages = [
        RawPage(
            page_number  = p["page_number"],
            pdf_type     = p["pdf_type"],
            raw_text     = p["raw_text"],
            raw_text_pre = p.get("raw_text_pre", ""),
        )
        for p in result.pages
    ]
    return IngestionResult(
        source_path = result.source_path,
        pdf_type    = result.pdf_type,
        total_pages = result.total_pages,
        pages       = pages,
        metadata    = result.metadata,
    )
