"""
Phase 1 — OutputWriter
Writes three files per run:

  *_phase1_raw.txt   — text straight from ingestor/OCR, before any processing.
                       Use this to debug extraction issues in isolation.

  *_phase1.json      — fully processed chunks with metadata (machine-readable).

  *_phase1.txt       — processed chunks, human-readable with separators.

Having the raw snapshot means you can diff raw vs processed to verify each
processing step (normalisation, diacritization) independently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chunker import Chunk
    from .ingestor import IngestionResult

logger = logging.getLogger(__name__)

_PAGE_SEP = "\n" + "═" * 60 + "\n"


class OutputWriter:
    """
    Writes Phase 1 output to disk.

    Returns (json_path, txt_path, raw_txt_path).
    """

    def __init__(self, output_dir: str | Path = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        ingestion: IngestionResult,
        chunks: list[Chunk],
        stem: str | None = None,
    ) -> tuple[Path, Path, Path]:
        base         = stem or Path(ingestion.source_path).stem
        raw_txt_path = self.output_dir / f"{base}_phase1_raw.txt"
        json_path    = self.output_dir / f"{base}_phase1.json"
        txt_path     = self.output_dir / f"{base}_phase1.txt"

        self._write_raw_txt(ingestion, raw_txt_path)
        self._write_json(ingestion, chunks, json_path)
        self._write_txt(ingestion, chunks, txt_path)

        logger.info("Phase 1 output → raw: %s | json: %s | txt: %s",
                    raw_txt_path, json_path, txt_path)
        return json_path, txt_path, raw_txt_path

    # ------------------------------------------------------------------ #
    #  Raw snapshot (pre-processing)                                       #
    # ------------------------------------------------------------------ #

    def _write_raw_txt(self, ingestion: IngestionResult, path: Path) -> None:
        """
        Writes the text exactly as it came out of PyMuPDF / OCR,
        before normalisation, diacritization, or any post-processing.
        One section per page, clearly labelled.
        """
        src  = Path(ingestion.source_path).name
        meta = ingestion.metadata
        lines = [
            f"# Phase 1 RAW EXTRACT — {src}",
            f"# PDF Type   : {ingestion.pdf_type}",
            f"# Total pages: {ingestion.total_pages}",
            f"# Title      : {meta.get('title', 'N/A')}",
            f"# Author     : {meta.get('author', 'N/A')}",
            f"# NOTE       : This is the text BEFORE normalisation or diacritization.",
            f"#              Compare with *_phase1.txt to audit processing quality.",
            "",
        ]

        for page in ingestion.pages:
            lines.append(
                f"[Page {page.page_number:03d} | {page.pdf_type}]"
            )
            lines.append(page.raw_text_pre or "(empty)")
            lines.append(_PAGE_SEP)

        path.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------ #
    #  JSON (processed)                                                    #
    # ------------------------------------------------------------------ #

    def _write_json(
        self, ingestion: IngestionResult, chunks: list[Chunk], path: Path
    ) -> None:
        payload = {
            "source":      ingestion.source_path,
            "pdf_type":    ingestion.pdf_type,
            "total_pages": ingestion.total_pages,
            "metadata":    ingestion.metadata,
            "chunk_count": len(chunks),
            # Per-page raw snapshot embedded in JSON for programmatic diffing
            "pages_raw": [
                {
                    "page_number": p.page_number,
                    "pdf_type":    p.pdf_type,
                    "raw_pre":     p.raw_text_pre,
                    "raw_post":    p.raw_text,
                }
                for p in ingestion.pages
            ],
            "chunks": [
                {
                    "chunk_id":   c.chunk_id,
                    "chapter":    c.chapter,
                    "page_start": c.page_start,
                    "page_end":   c.page_end,
                    "word_count": c.word_count,
                    "token_est":  c.token_est,
                    "text":       c.text,
                }
                for c in chunks
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    #  Plain text (processed)                                              #
    # ------------------------------------------------------------------ #

    def _write_txt(
        self, ingestion: IngestionResult, chunks: list[Chunk], path: Path
    ) -> None:
        parts = [c.text.strip() for c in chunks if c.text.strip()]
        path.write_text("\n\n".join(parts), encoding="utf-8")
