"""
Phase 3 sources — Claude vision scorer with three-dimensional rubric.

Each candidate is scored 0-3 on three axes:

  subject:    Does the image show the requested subject or scene?
  quality:    Is it sharp, well-composed, undamaged?
  cinematic:  Does it carry emotional or documentary impact?

Total score: 0-9.  Threshold for "kept" is total >= 4 AND subject >= 1.
Within kept candidates, ranking is by total score (higher = better).

This is qualitatively better than the v1 binary keep/drop because it
gives the orchestrator a ranking signal: when several candidates pass
threshold, we can pick the *best* image, not just any acceptable one.

Cost
----
~$0.005 per image (Haiku vision pricing).  Typical video: 30 image
shots × 3-4 candidates each = ~100 calls = ~$0.50.  Caching makes
re-renders free.

Images are always resized to ≤800 px wide before encoding to base64,
as required by the Anthropic API (oversized → 400 error).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .base import ImageCandidate

log = logging.getLogger(__name__)


MIN_KEEP_TOTAL = 4         # minimum total score to keep an image
MIN_KEEP_SUBJECT = 1       # subject must also clear this bar


_SYSTEM = (
    "You score images for a documentary video on their fit, quality, "
    "and cinematic weight.  Return only valid JSON — no markdown, no "
    "explanation."
)

_USER_TMPL = """\
This image is a candidate visual for a documentary about:
  Book: {book_title}
  Subject: {character_name}
  Specific shot query: "{query}"

Score the image on three dimensions, 0-3 each:

1. subject (0-3): does this image show the requested subject or scene?
   0 = completely unrelated
   1 = vaguely related (e.g. wrong era / wrong region)
   2 = related (right era and region, wrong specific subject)
   3 = exact match for what was requested

2. quality (0-3): is the image sharp, well-composed, undamaged?
   0 = unusable (heavily damaged, cropped wrong, illegible, watermarked)
   1 = poor (blurry, low contrast, awkward composition)
   2 = acceptable (sharp enough, reasonable composition)
   3 = excellent (sharp, well-framed, visually rich)

3. cinematic (0-3): does this image carry emotional or documentary impact?
   0 = flat / forgettable
   1 = ordinary
   2 = strong (clear subject, evocative atmosphere)
   3 = striking (the kind of image that makes you pause)

If the image is a diagram, chart, anatomical illustration, modern clip
art, stock-photo cliche, or watermarked sample image, score subject=0.

Return ONLY this JSON:
{{"subject": N, "quality": N, "cinematic": N, "reason": "one-line rationale in English"}}
"""


@dataclass
class VisionScorer:
    """Wraps the Anthropic client with our rubric."""
    api_key: str
    model: str = "claude-haiku-4-5-20251001"

    def score(self, candidate: ImageCandidate, *,
              book_title: str, character_name: str, query: str) -> ImageCandidate:
        """
        Score one candidate in place.  Mutates and returns it.

        Fail-open: on any API/processing error, the candidate is given
        a neutral score (subject=2, quality=2, cinematic=1) so it
        passes threshold rather than being silently dropped.
        """
        if not candidate.local_path or not candidate.local_path.exists():
            log.warning("Vision: skipping %s — no local path", candidate.title[:40])
            return candidate

        try:
            from anthropic import Anthropic
            from PIL import Image
        except ImportError as exc:
            log.warning("Vision: anthropic/PIL not installed (%s); fail-open", exc)
            _apply_neutral_score(candidate, "vision unavailable")
            return candidate

        try:
            # Resize to ≤800 px wide for the API
            with Image.open(candidate.local_path) as img:
                img = img.convert("RGB")
                if img.width > 800:
                    new_h = int(img.height * 800 / img.width)
                    img = img.resize((800, new_h), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                img_bytes = buf.getvalue()

            b64 = base64.standard_b64encode(img_bytes).decode()

            client = Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model=self.model,
                max_tokens=200,
                system=_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": _USER_TMPL.format(
                            book_title=book_title or "unknown",
                            character_name=character_name or "not specified",
                            query=query,
                        )},
                    ],
                }],
            )
            raw = msg.content[0].text.strip()
            scores = _parse_score_json(raw)

            candidate.score_subject   = int(scores.get("subject", 0))
            candidate.score_quality   = int(scores.get("quality", 0))
            candidate.score_cinematic = int(scores.get("cinematic", 0))
            candidate.vision_reason   = str(scores.get("reason", ""))[:200]

            log.info("Vision %d/%d/%d  %s  — %s",
                     candidate.score_subject, candidate.score_quality,
                     candidate.score_cinematic,
                     candidate.title[:40],
                     candidate.vision_reason[:60])

        except Exception as exc:
            log.warning("Vision: error on %s: %s — keeping with neutral score",
                        candidate.title[:40], exc)
            _apply_neutral_score(candidate, f"error: {exc!s}"[:80])

        return candidate


def _apply_neutral_score(c: ImageCandidate, reason: str) -> None:
    """Set a neutral score that passes threshold (fail-open policy)."""
    c.score_subject = 2
    c.score_quality = 2
    c.score_cinematic = 1
    c.vision_reason = f"[fail-open] {reason}"


def _parse_score_json(raw: str) -> dict:
    """Parse the score JSON Claude returns, tolerating common quirks."""
    raw = raw.strip()
    raw = (raw
           .removeprefix("```json")
           .removeprefix("```")
           .removesuffix("```")
           .strip())
    # Find first { and last } in case there's stray text
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in vision response: {raw[:200]}")
    return json.loads(m.group(0))


def passes_threshold(c: ImageCandidate) -> bool:
    """Return True if the candidate should be kept."""
    if not c.is_scored:
        return True   # unscored candidates pass (vision skipped)
    return (c.total_score >= MIN_KEEP_TOTAL
            and c.score_subject >= MIN_KEEP_SUBJECT)


def rank_candidates(candidates: list[ImageCandidate]) -> list[ImageCandidate]:
    """
    Return candidates sorted best-first by total score.
    Ties broken by: subject > quality > cinematic > original order.
    """
    return sorted(
        candidates,
        key=lambda c: (
            -c.total_score if c.is_scored else 99,
            -c.score_subject if c.is_scored else 0,
            -c.score_quality if c.is_scored else 0,
            -c.score_cinematic if c.is_scored else 0,
        ),
    )
