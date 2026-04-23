"""
Phase 1 — PageStitcher
For each consecutive pair of scanned pages, uses Claude Haiku to:
  1. Detect and strip running headers/footers from the top of each page.
  2. Join a sentence split across a page boundary.

Runs after OCR correction, before the normalizer.
Cost: ~$0.0001 per boundary (~$0.002 for a 20-page document).
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .ingestor import RawPage

logger = logging.getLogger(__name__)

_BOUNDARY_PROMPT = """\
نهاية الصفحة {prev_num}:
{prev_tail}

بداية الصفحة {next_num}:
{next_head}

أجب بـ JSON فقط (لا تضف أي نص خارجه):
{{"header": "نص الترويسة إن وجدت (عنوان كتاب أو فصل أو رقم صفحة) وإلا null", "join": true إذا كانت الجملة منقطعة بين الصفحتين وتستكمل في الصفحة التالية}}"""


class PageStitcher:
    """
    Stitches scanned-page boundaries: strips running headers/footers and
    joins sentences split across page breaks.

    One API call per consecutive page-pair.  Failures are fail-open: the
    original page texts are kept unchanged if the API call raises.

    Cost: ~$0.0001 per boundary (Haiku, ~80 tokens in+out).
    A 255-page book costs roughly $0.025 in stitching calls.
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

    def stitch_pages(self, pages: list[RawPage]) -> list[RawPage]:
        """
        Process boundaries between consecutive scanned pages in-place.
        Digital pages are skipped.
        Returns the same list.
        """
        scanned = [p for p in pages if p.pdf_type == "scanned" and p.raw_text.strip()]
        if len(scanned) < 2:
            logger.info("Fewer than 2 scanned pages — boundary stitching skipped.")
            return pages

        logger.info("Page stitching: %d boundary(ies) to process.", len(scanned) - 1)
        self._ensure_client()

        for idx in range(len(scanned) - 1):
            page_n  = scanned[idx]
            page_n1 = scanned[idx + 1]
            self.on_progress(
                f"Stitching boundary {page_n.page_number}→{page_n1.page_number} …",
                idx / (len(scanned) - 1),
            )
            self._stitch_boundary(page_n, page_n1)

        return pages

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # noqa: PLC0415
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def _stitch_boundary(self, page_n: RawPage, page_n1: RawPage) -> None:
        prev_lines = [l for l in page_n.raw_text.splitlines() if l.strip()]
        next_lines = [l for l in page_n1.raw_text.splitlines() if l.strip()]
        if not prev_lines or not next_lines:
            return

        prev_tail = "\n".join(prev_lines[-2:])
        next_head = "\n".join(next_lines[:3])

        prompt = _BOUNDARY_PROMPT.format(
            prev_num=page_n.page_number,
            next_num=page_n1.page_number,
            prev_tail=prev_tail,
            next_head=next_head,
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            m   = re.search(r"\{.*\}", raw, re.DOTALL)
            data: dict = json.loads(m.group()) if m else {}
        except Exception as exc:
            logger.warning(
                "Page boundary %d→%d: LLM call failed — %s",
                page_n.page_number, page_n1.page_number, exc,
            )
            return

        # 1. Strip running header/footer from the top of page N+1
        header = data.get("header")
        if header and isinstance(header, str):
            lines = page_n1.raw_text.splitlines()
            for i, line in enumerate(lines):
                if line.strip() and (
                    header.strip() in line or line.strip() in header
                ):
                    lines[i] = ""
                    logger.debug(
                        "Stripped header %r from page %d.",
                        header, page_n1.page_number,
                    )
                    break
            page_n1.raw_text = "\n".join(lines)

        # 2. Join a sentence split across the boundary
        if data.get("join"):
            next_lines_mut = page_n1.raw_text.lstrip().splitlines()
            first_idx = next(
                (i for i, l in enumerate(next_lines_mut) if l.strip()), None
            )
            if first_idx is not None:
                continuation          = next_lines_mut[first_idx].lstrip()
                next_lines_mut[first_idx] = ""
                page_n.raw_text       = page_n.raw_text.rstrip() + " " + continuation
                page_n1.raw_text      = "\n".join(next_lines_mut)
                logger.debug(
                    "Joined sentence across page %d→%d.",
                    page_n.page_number, page_n1.page_number,
                )
