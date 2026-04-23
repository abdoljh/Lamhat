"""
Phase 1 — SemanticChunker
Splits the full book text into overlapping chunks suitable for LLM context
windows, respecting chapter/section boundaries where they exist.

Strategy (in priority order)
──────────────────────────────
1. Chapter boundaries   — detected from PDF outline OR heading patterns.
2. Paragraph boundaries — double newline as natural paragraph break.
3. Arabic sentence ends — '.' / '؟' / '!' / '،' as sentence terminators.
4. Hard token limit     — never exceed max_tokens regardless of above.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Arabic + Latin sentence terminators
_SENT_ENDS = re.compile(r"(?<=[.؟!])\s+")

# Heuristic heading pattern: short lines (< 60 chars) that contain no
# sentence-ending punctuation — typical for Arabic chapter titles.
_HEADING_PATTERN = re.compile(
    r"^(?!.*[.،؛؟!])(.{4,60})\s*$",
    re.MULTILINE,
)


@dataclass
class Chunk:
    chunk_id:    int
    chapter:     str           # detected chapter title, or "unknown"
    page_start:  int
    page_end:    int
    text:        str
    word_count:  int = field(init=False)
    token_est:   int = field(init=False)   # rough estimate: words × 1.4 for Arabic

    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.token_est  = int(self.word_count * 1.4)


class SemanticChunker:
    """
    Produces a list of Chunk objects from normalised, diacritized page text.

    Args:
        max_tokens:   Hard ceiling on estimated tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks for context.
        min_chunk_words: Discard chunks shorter than this (headers, blank pages).
    """

    def __init__(
        self,
        max_tokens:      int = 1500,
        overlap_tokens:  int = 200,
        min_chunk_words: int = 8,
    ):
        self.max_tokens      = max_tokens
        self.overlap_tokens  = overlap_tokens
        self.min_chunk_words = min_chunk_words

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def chunk_pages(self, pages) -> list[Chunk]:
        """
        Accept a list of RawPage objects and return a flat list of Chunks.
        """
        # 1. Merge all pages into a single stream, tracking page boundaries
        non_empty = [p for p in pages if p.raw_text.strip()]
        if not non_empty:
            logger.warning("chunk_pages: all %d pages are empty — returning no chunks.", len(pages))
            return []
        if len(non_empty) < len(pages):
            logger.info("chunk_pages: %d/%d pages non-empty (skipping empty pages).",
                        len(non_empty), len(pages))
        full_text, page_map = self._merge_pages(non_empty)
        logger.debug("chunk_pages: merged text length = %d chars.", len(full_text))

        # 2. Split on chapter headings first
        sections = self._split_by_chapters(full_text)
        logger.debug("chunk_pages: detected %d section(s).", len(sections))

        chunks: list[Chunk] = []
        chunk_id = 0
        pending_stub = ""   # buffered text (heading + short body) prepended to next real chunk

        for chapter_title, section_text in sections:
            # 3. Split each section into token-safe pieces
            pieces = self._split_to_token_limit(section_text)
            for piece in pieces:
                piece_text = piece.strip()
                if len(piece_text.split()) < self.min_chunk_words:
                    # Buffer heading + body so it is prepended to next real chunk
                    pending_stub = (pending_stub + "\n\n" + piece_text).strip() if pending_stub else piece_text
                    logger.debug("Buffering stub (%d words) in section '%s': %r",
                                 len(piece_text.split()), chapter_title, piece_text[:60])
                    continue
                text = (pending_stub + "\n\n" + piece_text).strip() if pending_stub else piece_text
                pending_stub = ""
                page_s, page_e = self._estimate_pages(piece, page_map)
                chunks.append(Chunk(
                    chunk_id   = chunk_id,
                    chapter    = chapter_title,
                    page_start = page_s,
                    page_end   = page_e,
                    text       = text,
                ))
                chunk_id += 1

        # Flush any trailing stub into the last chunk (or as its own chunk)
        if pending_stub:
            if chunks:
                last = chunks[-1]
                chunks[-1] = Chunk(
                    chunk_id   = last.chunk_id,
                    chapter    = last.chapter,
                    page_start = last.page_start,
                    page_end   = last.page_end,
                    text       = last.text + "\n\n" + pending_stub,
                )
            else:
                page_s, page_e = self._estimate_pages(pending_stub, page_map)
                chunks.append(Chunk(
                    chunk_id   = 0,
                    chapter    = "intro",
                    page_start = page_s,
                    page_end   = page_e,
                    text       = pending_stub,
                ))

        logger.info(
            "Chunked into %d pieces (max_tokens=%d, overlap=%d).",
            len(chunks), self.max_tokens, self.overlap_tokens,
        )
        if len(chunks) == 0:
            logger.warning(
                "chunk_pages produced 0 chunks from %d pages (%d chars). "
                "min_chunk_words=%d. Check normalisation output.",
                len(non_empty), len(full_text), self.min_chunk_words,
            )
        return chunks

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_pages(pages) -> tuple[str, dict[int, tuple[int, int]]]:
        """
        Returns full text and a char-offset → (page_start, page_end) map.
        """
        parts = []
        offset = 0
        page_map: dict[int, tuple[int, int]] = {}  # char_offset → (pnum, pnum)

        for page in pages:
            start = offset
            parts.append(page.raw_text)
            offset += len(page.raw_text) + 1   # +1 for the \n separator
            page_map[start] = (page.page_number, page.page_number)

        return "\n".join(p.raw_text for p in pages), page_map

    def _split_by_chapters(self, text: str) -> list[tuple[str, str]]:
        """
        Returns list of (chapter_title, chapter_text).
        Falls back to [("full_book", text)] if no headings detected.

        Consecutive headings with no body text between them (e.g. a chapter
        title followed immediately by a subtitle) are accumulated and prepended
        to the next section that does have body content.  This prevents title
        and subtitle lines from being silently discarded.
        """
        matches = list(_HEADING_PATTERN.finditer(text))
        if not matches:
            return [("full_book", text)]

        sections: list[tuple[str, str]] = []
        pending: list[str] = []   # heading-only lines awaiting a body

        pre = text[:matches[0].start()].strip()
        if pre:
            sections.append(("intro", pre))

        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if section_text:
                if pending:
                    # Preserve correct order: pending headings → current heading → body
                    section_text = "\n\n".join(pending + [title]) + "\n\n" + section_text
                    pending = []
                else:
                    section_text = title + "\n\n" + section_text
                sections.append((title, section_text))
            else:
                pending.append(title)

        # Flush any trailing headings with no body after them
        if pending:
            extra = "\n\n".join(pending)
            if sections:
                last_title, last_text = sections[-1]
                sections[-1] = (last_title, last_text + "\n\n" + extra)
            else:
                sections.append(("full_book", extra))

        return sections if sections else [("full_book", text)]

    def _split_to_token_limit(self, text: str) -> list[str]:
        """
        Recursive character splitting with overlap, Arabic-aware.
        Separators tried in order: paragraph → sentence → character.
        """
        # Approximate tokens
        if int(len(text.split()) * 1.4) <= self.max_tokens:
            return [text]

        separators = ["\n\n", "\n", ".", "،", " "]
        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                return self._merge_with_overlap(parts, sep)

        # Last resort: hard character split
        size = int(self.max_tokens / 1.4 * 5)   # rough chars
        return [text[i:i + size] for i in range(0, len(text), size)]

    def _merge_with_overlap(self, parts: list[str], sep: str) -> list[str]:
        chunks, current, overlap_buf = [], "", ""
        target_words = int(self.max_tokens / 1.4)
        overlap_words = int(self.overlap_tokens / 1.4)

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate.split()) <= target_words:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    # Build overlap from tail of current chunk
                    words = current.split()
                    overlap_buf = " ".join(words[-overlap_words:])
                current = (overlap_buf + " " + part).strip() if overlap_buf else part

        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _estimate_pages(chunk_text: str, page_map: dict) -> tuple[int, int]:
        """Best-effort page attribution; returns (1, 1) when mapping fails."""
        if not page_map:
            return 1, 1
        pages = list(page_map.values())
        return pages[0][0], pages[-1][1]
