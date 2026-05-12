"""
Phase 1 — Pipeline
Split into two independently runnable stages:

  Phase1aPipeline  — PDF preprocessing: strip margins → export images → OCR
    Four modes (matching OCR-me):
      "single_book"  Strip headers/footers → export clean page images → ZIP
                     + optional footer images (PDF + ZIP)
                     + optional photograph extraction (ZIP)
                     + optional Kraken OCR
      "raw_export"   Export original pages (no stripping) → ZIP
      "batch"        Alias for single_book; caller iterates over PDFs
      "visual"       Streamlit-only per-page wizard; pipeline unchanged

    Saves:
      *_phase1a_pages.zip            Clean (or raw) page images for offline OCR
      *_phase1a_footers.pdf          Assembled footer regions, one per page (optional)
      *_phase1a_footers_imgs.zip     Footer images, one PNG per page (optional)
      *_phase1a_photos.zip           Extracted photographs + captions (optional)
      *_phase1a_corrected.txt        Raw Kraken OCR text, per page
      *_phase1a_normalized.txt       After Arabic normalisation, per page
      *_phase1a.json                 Structured page data consumed by Phase 1b

    When ocr_backend="none": only the ZIPs are produced; text files are empty
    stubs so the user can run OCR externally and upload text to Phase 1b.

  Phase1bPipeline  — Steps 4-6: Chunk → Write outputs → Summarise
    Accepts:  Phase1aResult (in-memory)  OR  path to *_phase1a.json

  Phase1Pipeline   — Combined 1a + 1b in a single call (backward-compatible).
"""

from __future__ import annotations

import json
import logging
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .core.normalizer    import ArabicTextNormalizer
from .core.chunker       import SemanticChunker, Chunk
from .core.output_writer import OutputWriter
from .core.summarizer    import BookSummarizer
from .core.ingestor      import RawPage, IngestionResult

logger = logging.getLogger(__name__)

_PAGE_SEP = "\n" + "═" * 60 + "\n"


# ──────────────────────────────────────────────────────────────────────────── #
#  Shared configuration                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class Phase1Config:
    # ── Phase 1a: PDF preprocessing ──────────────────────────────────────── #

    # Processing mode — matches OCR-me's four modes.
    # "single_book"  strip margins + export + optional footers/photos + OCR
    # "raw_export"   export original pages (no stripping), no OCR
    # "batch"        same as single_book (caller iterates over multiple PDFs)
    # "visual"       Streamlit per-page wizard; pipeline skips visual-only steps
    mode: str = "single_book"

    # Strip running headers/footers before exporting page images.
    strip_margins: bool = True
    # DPI used for header/footer margin detection (lower = faster).
    hf_dpi: int = 300
    # DPI for page image export (higher = better OCR quality).
    export_dpi: int = 400

    # Extract footnote regions into a labeled PDF and image ZIP.
    include_footers: bool = True
    # Extract photographic regions (and captions) into a ZIP.
    include_photos: bool = False
    # Split page images ZIP into parts no larger than this (MB). 0 = no split.
    zip_split_mb: float = 250.0

    # OCR backend for in-app recognition.
    # "kraken"  — offline Kraken model (Arabic, best quality on Streamlit Cloud)
    # "none"    — skip OCR; export images only for offline/external OCR
    ocr_backend: str = "kraken"

    # Kraken-specific parameters
    kraken_bidi: str = "auto"             # "auto" | "R" | "L" | "off"
    kraken_threshold: float = 0.5        # NLBin binarization threshold (0–1)
    kraken_pad: int = 16                 # line padding in pixels
    kraken_autocast: bool = False        # fp16 autocast (GPU only)
    kraken_text_direction: str = "horizontal-rl"
    kraken_no_legacy_polygons: bool = False

    # ── Phase 1b: chunking & summarisation ───────────────────────────────── #
    max_tokens: int = 1500
    overlap_tokens: int = 200
    output_dir: str = "output"

    # LLM summarization + script generation
    anthropic_api_key: str = ""
    script_genre: str = "non-fiction"
    book_author: str = ""
    book_pages: int = 0
    book_structure: str = ""
    diacritize: bool = True
    scriptwriter_model: str = "claude-haiku-4-5-20251001"

    # ── Legacy fields (kept for backward compatibility) ───────────────────── #
    ocr_gpu: bool = False
    ocr_dpi: int = 400
    pdf_mode: str = "auto"
    ocr_correction: bool = False
    page_stitching: bool = False


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
    # Serialisable page records.
    # raw_text     = normalised text (ready for chunking, empty if no OCR)
    # raw_text_pre = raw Kraken output (for audit / raw snapshot)
    pages:                list[dict]
    corrected_txt_path:   Path
    normalized_txt_path:  Path
    normalized_json_path: Path
    # Page images ZIP (one or more parts when zip_split_mb is active)
    pages_zip_paths:      list[Path] = field(default_factory=list)
    # Footer outputs (optional — only when include_footers=True and footers found)
    footers_pdf_path:     Optional[Path] = None
    footers_zip_path:     Optional[Path] = None
    n_footer_pages:       int = 0
    # Photo outputs (optional — only when include_photos=True and photos found)
    photos_zip_path:      Optional[Path] = None
    n_photos:             int = 0
    elapsed_sec:          float = 0.0
    warnings:             list[str] = field(default_factory=list)

    # Backward-compat: single-zip callers can still access .pages_zip_path
    @property
    def pages_zip_path(self) -> Optional[Path]:
        return self.pages_zip_paths[0] if self.pages_zip_paths else None


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
    script_path:       Optional[Path] = None
    script_diac_path:  Optional[Path] = None
    script_meta_path:  Optional[Path] = None
    elapsed_sec:       float = 0.0
    warnings:          list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────── #
#  Phase 1a — Strip margins → Export images → OCR → Normalise                  #
# ──────────────────────────────────────────────────────────────────────────── #

class Phase1aPipeline:
    """
    Phase 1a pipeline implementing all four OCR-me modes:

      single_book  Strip headers/footers → export page images → extract
                   footers (optional) → extract photos (optional) → Kraken OCR
      raw_export   Export original pages (no stripping), no OCR
      batch        Same as single_book; caller iterates over multiple PDFs
      visual       Streamlit per-page wizard; pipeline produces same outputs
                   as single_book (visual crop control is a UI concern)

    Outputs:
      *_phase1a_pages[_part_NNN].zip   Page images (may be split into parts)
      *_phase1a_footers.pdf            Footer regions assembled into a PDF
      *_phase1a_footers_imgs.zip       Footer images, one PNG per page
      *_phase1a_photos.zip             Extracted photographs + captions
      *_phase1a_corrected.txt          Raw Kraken OCR text
      *_phase1a_normalized.txt         Text after Arabic normalisation
      *_phase1a.json                   Structured data consumed by Phase1bPipeline
    """

    def __init__(
        self,
        config: Phase1Config | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ):
        self.cfg         = config or Phase1Config()
        self.on_progress = on_progress or (lambda s, p: None)

    def run(self, pdf_path: str | Path) -> Phase1aResult:
        t0        = time.perf_counter()
        warnings: list[str] = []
        pdf_path  = Path(pdf_path)
        stem      = pdf_path.stem
        out_dir   = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_export = self.cfg.mode == "raw_export"

        # ── Step 1: Strip headers/footers (skipped for raw_export) ────────── #
        self._progress("Detecting and stripping page margins …", 0.05)
        if self.cfg.strip_margins and not raw_export:
            try:
                from .core.header_footer import strip_pdf, Params as HFParams
                stripped_path = out_dir / f"{stem}_stripped.pdf"
                hf_params = HFParams(dpi=self.cfg.hf_dpi)
                strip_pdf(pdf_path, stripped_path, p=hf_params, mode="cropbox")
                work_pdf = stripped_path
                logger.info("Margins stripped → %s", stripped_path)
            except Exception as exc:
                warnings.append(f"Margin stripping failed ({exc}) — using original PDF.")
                logger.warning(warnings[-1])
                work_pdf = pdf_path
        else:
            work_pdf = pdf_path

        # ── Step 2: Export pages as images ───────────────────────────────── #
        self._progress("Exporting page images …", 0.15)
        try:
            from .core.page_export import export_pages_as_images
            pages_dir = out_dir / f"{stem}_pages"
            pages_dir.mkdir(exist_ok=True)
            img_paths = export_pages_as_images(
                work_pdf, pages_dir, dpi=self.cfg.export_dpi, fmt="png"
            )
            total_pages = len(img_paths)
            logger.info("Exported %d page images to %s", total_pages, pages_dir)
        except Exception as exc:
            warnings.append(f"Page image export failed: {exc}")
            logger.exception("Page image export failed")
            img_paths   = []
            total_pages = 0

        # ── Step 3: Bundle images into ZIP (with optional split) ──────────── #
        self._progress("Creating page images ZIP …", 0.25)
        pages_zip_paths: list[Path] = []
        if img_paths:
            pages_zip_paths = _bundle_to_zips(
                img_paths, out_dir, stem, self.cfg.zip_split_mb
            )
            logger.info(
                "Page images → %d ZIP part(s): %s",
                len(pages_zip_paths), [p.name for p in pages_zip_paths],
            )
        else:
            warnings.append("No page images produced — ZIP not created.")

        # ── Step 4: Extract footers (Single Book / Visual mode only) ─────── #
        footers_pdf_path: Optional[Path] = None
        footers_zip_path: Optional[Path] = None
        n_footer_pages   = 0

        if self.cfg.include_footers and not raw_export and img_paths:
            self._progress("Extracting footnote regions …", 0.30)
            try:
                from .core.page_export import extract_footers_pdf
                from .core.header_footer import Params as HFParams
                foot_pdf  = out_dir / f"{stem}_phase1a_footers.pdf"
                foot_dir  = out_dir / f"{stem}_footer_imgs"
                foot_zip  = out_dir / f"{stem}_phase1a_footers_imgs.zip"
                hf_params = HFParams(dpi=self.cfg.hf_dpi)
                foot_pages = extract_footers_pdf(
                    pdf_path, foot_pdf,
                    p=hf_params,
                    img_dir=str(foot_dir),
                    images_dpi=self.cfg.export_dpi,
                    zip_path=str(foot_zip),
                )
                n_footer_pages = len(foot_pages)
                if n_footer_pages:
                    if foot_pdf.exists():
                        footers_pdf_path = foot_pdf
                    if foot_zip.exists():
                        footers_zip_path = foot_zip
                    logger.info(
                        "Footers extracted from %d page(s) → %s / %s",
                        n_footer_pages, foot_pdf.name, foot_zip.name,
                    )
                else:
                    warnings.append("No footnote regions detected in this document.")
                    logger.info("No footnote regions found.")
            except Exception as exc:
                warnings.append(f"Footer extraction failed: {exc}")
                logger.exception("Footer extraction failed")

        # ── Step 5: Extract photographs (optional) ────────────────────────── #
        photos_zip_path: Optional[Path] = None
        n_photos         = 0

        if self.cfg.include_photos and not raw_export and img_paths:
            self._progress("Extracting photographs …", 0.40)
            try:
                from .core.image_extract import extract_images
                photo_dir = out_dir / f"{stem}_photos"
                photo_zip = out_dir / f"{stem}_phase1a_photos.zip"
                photo_results = extract_images(
                    pdf_path, str(photo_dir),
                    dpi=self.cfg.export_dpi,
                    with_captions=True,
                    use_body_crop=True,
                    zip_path=str(photo_zip),
                )
                n_photos = sum(len(r.photos) for r in photo_results)
                if n_photos and photo_zip.exists():
                    photos_zip_path = photo_zip
                    logger.info(
                        "Photographs extracted: %d image(s) → %s",
                        n_photos, photo_zip.name,
                    )
                else:
                    warnings.append("No photographs detected in this document.")
                    logger.info("No photographs found.")
            except Exception as exc:
                warnings.append(f"Photograph extraction failed: {exc}")
                logger.exception("Photograph extraction failed")

        # ── Step 6: OCR (Kraken, in-app) — skip for raw_export ───────────── #
        pages_data: list[dict] = []

        if self.cfg.ocr_backend == "kraken" and img_paths and not raw_export:
            self._progress("Loading Kraken OCR model …", 0.45)
            try:
                from .core.kraken_engine import (
                    load_model, ocr_page, KrakenNotAvailableError,
                )
                from PIL import Image as PILImage
                model = load_model()
                for i, img_path in enumerate(img_paths):
                    pct = 0.45 + 0.40 * (i / total_pages)
                    self._progress(
                        f"Kraken OCR — page {i + 1}/{total_pages} …", pct
                    )
                    img     = PILImage.open(img_path)
                    text, _ = ocr_page(
                        model, img,
                        threshold          = self.cfg.kraken_threshold,
                        text_direction     = self.cfg.kraken_text_direction,
                        autocast           = self.cfg.kraken_autocast,
                        pad                = self.cfg.kraken_pad,
                        bidi_key           = self.cfg.kraken_bidi,
                        no_legacy_polygons = self.cfg.kraken_no_legacy_polygons,
                    )
                    pages_data.append({
                        "page_number":  i + 1,
                        "pdf_type":     "scanned",
                        "raw_text":     text,
                        "raw_text_pre": text,
                    })
                logger.info("Kraken OCR complete — %d pages", len(pages_data))
            except KrakenNotAvailableError as exc:
                warnings.append(str(exc))
                logger.warning("Kraken not available: %s", exc)
                pages_data = _empty_pages(total_pages)
            except Exception as exc:
                warnings.append(f"Kraken OCR failed: {exc}")
                logger.exception("Kraken OCR failed")
                pages_data = _empty_pages(total_pages)

        else:
            pages_data = _empty_pages(total_pages)
            if self.cfg.ocr_backend not in ("none", "kraken") and not raw_export:
                warnings.append(
                    f"Unknown OCR backend '{self.cfg.ocr_backend}' — "
                    "no OCR performed."
                )

        # ── Step 7: Normalise text ────────────────────────────────────────── #
        has_text = any(d["raw_text"].strip() for d in pages_data)
        if has_text:
            self._progress("Normalising Arabic text …", 0.87)
            try:
                normalizer = ArabicTextNormalizer()
                raw_pages  = [
                    RawPage(
                        page_number  = d["page_number"],
                        pdf_type     = d["pdf_type"],
                        raw_text     = d["raw_text"],
                        raw_text_pre = d["raw_text_pre"],
                    )
                    for d in pages_data
                ]
                raw_pages  = normalizer.normalize_pages(raw_pages)
                pages_data = [
                    {
                        "page_number":  p.page_number,
                        "pdf_type":     p.pdf_type,
                        "raw_text":     p.raw_text,
                        "raw_text_pre": p.raw_text_pre,
                    }
                    for p in raw_pages
                ]
            except Exception as exc:
                warnings.append(f"Arabic normalisation failed: {exc}")
                logger.exception("Arabic normalisation failed")

        # ── Step 8: Save output files ─────────────────────────────────────── #
        self._progress("Saving output files …", 0.92)

        corrected_txt_path   = out_dir / f"{stem}_phase1a_corrected.txt"
        normalized_txt_path  = out_dir / f"{stem}_phase1a_normalized.txt"
        normalized_json_path = out_dir / f"{stem}_phase1a.json"

        _write_corrected(pdf_path, pages_data, self.cfg.ocr_backend, corrected_txt_path)
        _write_normalized_txt(pages_data, normalized_txt_path)
        _write_normalized_json(
            str(pdf_path), "scanned", total_pages,
            {"title": stem}, pages_data, normalized_json_path,
        )
        logger.info(
            "Phase 1a output → zips: %s | footers: %s | photos: %s",
            [p.name for p in pages_zip_paths], footers_pdf_path, photos_zip_path,
        )

        elapsed = time.perf_counter() - t0
        self._progress("Phase 1a done ✓", 1.0)
        logger.info("Phase 1a complete in %.1fs — %d pages.", elapsed, total_pages)

        return Phase1aResult(
            source_path          = str(pdf_path),
            pdf_type             = "scanned",
            total_pages          = total_pages,
            metadata             = {"title": stem},
            pages                = pages_data,
            corrected_txt_path   = corrected_txt_path,
            normalized_txt_path  = normalized_txt_path,
            normalized_json_path = normalized_json_path,
            pages_zip_paths      = pages_zip_paths,
            footers_pdf_path     = footers_pdf_path,
            footers_zip_path     = footers_zip_path,
            n_footer_pages       = n_footer_pages,
            photos_zip_path      = photos_zip_path,
            n_photos             = n_photos,
            elapsed_sec          = elapsed,
            warnings             = warnings,
        )

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

        if isinstance(source, (str, Path)):
            source = _load_phase1a_json(Path(source))

        warnings.extend(source.warnings)
        ingestion = _reconstruct_ingestion(source)

        # ── Step 4: Chunk ─────────────────────────────────────────────────── #
        self._progress("Chunking text …", 0.10)
        chunker = SemanticChunker(
            max_tokens     = self.cfg.max_tokens,
            overlap_tokens = self.cfg.overlap_tokens,
        )
        chunks = chunker.chunk_pages(ingestion.pages)

        # ── Step 5: Write extraction output ───────────────────────────────── #
        self._progress("Writing extraction output …", 0.40)
        writer = OutputWriter(output_dir=self.cfg.output_dir)
        json_path, txt_path, raw_txt_path = writer.write(ingestion, chunks)

        # ── Step 6: Summarize → Script → Diacritize ────────────────────────── #
        script_path = script_diac_path = script_meta_path = None
        if self.cfg.anthropic_api_key and chunks:
            try:
                self._progress("Summarising book (Reader + Consolidator) …", 0.55)
                summarizer = BookSummarizer(
                    api_key            = self.cfg.anthropic_api_key,
                    genre              = self.cfg.script_genre,
                    output_dir         = self.cfg.output_dir,
                    book_author        = self.cfg.book_author,
                    book_pages         = self.cfg.book_pages,
                    book_structure     = self.cfg.book_structure,
                    diacritize         = self.cfg.diacritize,
                    scriptwriter_model = self.cfg.scriptwriter_model,
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
    Kept for backward compatibility.
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

def _bundle_to_zips(
    img_paths: list[Path],
    out_dir: Path,
    stem: str,
    split_mb: float,
) -> list[Path]:
    """Bundle a list of image files into one or more ZIPs.

    When split_mb > 0 the output is split so no part exceeds split_mb MB.
    Single-part output: ``{stem}_phase1a_pages.zip``.
    Multi-part output:  ``{stem}_phase1a_pages_part_001.zip``, …
    """
    max_bytes = int(split_mb * 1024 * 1024) if split_mb > 0 else 0

    parts: list[list[Path]] = [[]]
    cur_size = 0
    for p in img_paths:
        size = p.stat().st_size
        if max_bytes and cur_size and cur_size + size > max_bytes:
            parts.append([])
            cur_size = 0
        parts[-1].append(p)
        cur_size += size

    zip_paths: list[Path] = []
    multi = len(parts) > 1
    for j, part in enumerate(parts, 1):
        if not part:
            continue
        name = (
            f"{stem}_phase1a_pages_part_{j:03d}.zip" if multi
            else f"{stem}_phase1a_pages.zip"
        )
        zp = out_dir / name
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in part:
                zf.write(p, arcname=p.name)
        zip_paths.append(zp)
    return zip_paths


def _empty_pages(total: int) -> list[dict]:
    return [
        {"page_number": i + 1, "pdf_type": "scanned", "raw_text": "", "raw_text_pre": ""}
        for i in range(total)
    ]


def _write_corrected(
    pdf_path: Path,
    pages_data: list[dict],
    ocr_backend: str,
    path: Path,
) -> None:
    lines = [
        f"# Phase 1a OCR — {pdf_path.name}",
        f"# OCR backend  : {ocr_backend}",
        f"# Total pages  : {len(pages_data)}",
        "# NOTE: raw OCR text before Arabic normalisation.",
        "",
    ]
    for d in pages_data:
        lines.append(f"[Page {d['page_number']:03d} | {d['pdf_type']}]")
        lines.append(d["raw_text_pre"] or "(empty)")
        lines.append(_PAGE_SEP)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_normalized_txt(pages_data: list[dict], path: Path) -> None:
    blocks = [d["raw_text"].strip() for d in pages_data if d["raw_text"].strip()]
    path.write_text("\n\n".join(blocks), encoding="utf-8")


def _write_normalized_json(
    source: str,
    pdf_type: str,
    total_pages: int,
    metadata: dict,
    pages_data: list[dict],
    path: Path,
) -> None:
    payload = {
        "source":      source,
        "pdf_type":    pdf_type,
        "total_pages": total_pages,
        "metadata":    metadata,
        "pages":       pages_data,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_phase1a_json(json_path: Path) -> Phase1aResult:
    """Reconstruct a Phase1aResult from a *_phase1a.json file."""
    data    = json.loads(json_path.read_text(encoding="utf-8"))
    out_dir = json_path.parent
    stem    = json_path.stem[: -len("_phase1a")]
    return Phase1aResult(
        source_path          = data["source"],
        pdf_type             = data["pdf_type"],
        total_pages          = data["total_pages"],
        metadata             = data["metadata"],
        pages                = data["pages"],
        corrected_txt_path   = out_dir / f"{stem}_phase1a_corrected.txt",
        normalized_txt_path  = out_dir / f"{stem}_phase1a_normalized.txt",
        normalized_json_path = json_path,
        pages_zip_paths      = [],
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
