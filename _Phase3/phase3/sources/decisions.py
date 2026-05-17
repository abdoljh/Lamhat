"""
Phase 3 sources — review dossier (decisions.json).

The review dossier is the contract between the *prebuild* pass (which
fetches and scores all candidates) and the *render* pass (which burns
the video).  Between those two passes the user can edit the dossier to:

  * Swap which candidate is `chosen` for any image shot.
  * Drop a personally-supplied image into `overrides/shot_NN.jpg` and
    set `"override": "overrides/shot_NN.jpg"` in the matching entry.
  * Pin a canonical portrait of the main character that gets reused
    for every `portrait` shot — the single most effective biography
    quality lever.

The dossier is plain JSON next to a directory of thumbnails the user
can preview.  No GUI, no Streamlit needed for review — open the folder
in a file browser, look at the candidate JPEGs, edit a text file.

Anatomy
-------
output/review/
├── decisions.json            ← The editable contract
├── README.txt                ← Human-readable usage notes
├── overrides/                ← User drops .jpg/.png here
│   └── (typically: shot_03.jpg, shot_38.jpg, character.jpg, ...)
├── shot_NN_<visual>/         ← One folder per image-needing shot
│   ├── context.txt           ← What this shot is about
│   ├── candidates.json       ← Machine copy of the candidate list
│   ├── <source>_a.jpg        ← Downloaded candidate thumbnails
│   ├── <source>_b.jpg
│   └── ...
└── ...

decisions.json shape
--------------------
{
  "version":   1,
  "book":      {"title": "...", "character": "Jafar al-Askari"},
  "pinned_portrait": "overrides/character.jpg" | null,
  "shots": {
    "3": {
      "visual":       "portrait",
      "query":        "Jafar al-Askari Iraqi officer historical portrait 1920s",
      "duration_sec": 7.1,
      "arabic_caption_excerpt": "...",
      "chosen":       "pexels:Portrait of a man in a historical military uniform",
      "chosen_url":   "https://images.pexels.com/...",
      "chosen_file":  "shot_03_portrait/pexels_a.jpg",
      "override":     null,
      "candidates": [
        {"source": "pexels", "title": "...", "score": 8,
         "score_breakdown": {"subject": 3, "quality": 2, "cinematic": 3},
         "url": "...", "file": "shot_03_portrait/pexels_a.jpg",
         "vision_reason": "..."},
        ...
      ]
    },
    ...
  }
}

Editing rules
-------------
* To pick a different candidate, copy its `"source:title"` string into
  `chosen` (and optionally update `chosen_url` / `chosen_file` to
  match — render will re-resolve from the candidates list anyway).
* To use a personal image, drop the file into `overrides/` and set
  `override` to its path relative to the review dir
  (e.g. `"overrides/shot_03.jpg"`).  `chosen` is ignored when
  `override` is set.
* To use the same image for *all* portrait shots, set
  `pinned_portrait` to a path under `overrides/` and the prebuild step
  will retroactively flag every `portrait` shot's override field.

The render pass resolves in this order per shot:
    override → chosen → fallback to original Fetcher waterfall.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


DECISIONS_VERSION = 1
DECISIONS_FILENAME = "decisions.json"
OVERRIDES_SUBDIR = "overrides"

# Visuals that need an external image (everything that isn't a
# typography card).  Mirrors render.TYPOGRAPHY_VISUALS in inverse.
_IMAGE_VISUALS = {"portrait", "location", "object", "archive", "broll"}


@dataclass
class CandidateEntry:
    """One candidate image, as recorded in decisions.json."""
    source: str                       # 'loc' | 'wikimedia' | 'internet_archive' | 'pexels'
    title: str
    url: str
    file: str = ""                    # Path relative to review_dir
    score: int = -1                   # Sum of vision scores (0..9), -1 = unscored
    score_breakdown: dict | None = None
    vision_reason: str = ""
    width: int = 0
    height: int = 0
    license_short: str = ""


@dataclass
class ShotDecision:
    """One image shot's decision record."""
    visual: str
    query: str
    duration_sec: float
    arabic_caption_excerpt: str = ""
    chosen: str = ""                  # 'source:title' of preferred candidate
    chosen_url: str = ""
    chosen_file: str = ""
    override: str | None = None       # Path under review_dir (e.g. 'overrides/shot_03.jpg')
    candidates: list[CandidateEntry] = field(default_factory=list)


@dataclass
class Decisions:
    """The whole dossier."""
    book: dict
    pinned_portrait: str | None = None
    shots: dict[int, ShotDecision] = field(default_factory=dict)
    version: int = DECISIONS_VERSION

    # ── Serialisation ──────────────────────────────────────────── #

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "book":    self.book,
            "pinned_portrait": self.pinned_portrait,
            "shots": {
                str(idx): {
                    "visual":       d.visual,
                    "query":        d.query,
                    "duration_sec": d.duration_sec,
                    "arabic_caption_excerpt": d.arabic_caption_excerpt,
                    "chosen":       d.chosen,
                    "chosen_url":   d.chosen_url,
                    "chosen_file":  d.chosen_file,
                    "override":     d.override,
                    "candidates":   [asdict(c) for c in d.candidates],
                }
                for idx, d in self.shots.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Decisions":
        shots: dict[int, ShotDecision] = {}
        for idx_str, raw in (data.get("shots") or {}).items():
            cands = [CandidateEntry(**c) for c in (raw.get("candidates") or [])]
            shots[int(idx_str)] = ShotDecision(
                visual=raw.get("visual", ""),
                query=raw.get("query", ""),
                duration_sec=raw.get("duration_sec", 0.0),
                arabic_caption_excerpt=raw.get("arabic_caption_excerpt", ""),
                chosen=raw.get("chosen", ""),
                chosen_url=raw.get("chosen_url", ""),
                chosen_file=raw.get("chosen_file", ""),
                override=raw.get("override"),
                candidates=cands,
            )
        return cls(
            version=data.get("version", DECISIONS_VERSION),
            book=data.get("book", {}),
            pinned_portrait=data.get("pinned_portrait"),
            shots=shots,
        )

    def save(self, review_dir: Path) -> Path:
        review_dir = Path(review_dir)
        review_dir.mkdir(parents=True, exist_ok=True)
        out = review_dir / DECISIONS_FILENAME
        out.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Decisions written → %s", out)
        return out

    @classmethod
    def load(cls, review_dir: Path) -> "Decisions":
        review_dir = Path(review_dir)
        p = review_dir / DECISIONS_FILENAME
        if not p.exists():
            raise FileNotFoundError(
                f"No decisions.json in {review_dir}.  Run the prebuild "
                f"step first: python prebuild_assets.py --review-dir "
                f"{review_dir} ..."
            )
        data = json.loads(p.read_text(encoding="utf-8"))
        d = cls.from_dict(data)
        log.info(
            "Decisions loaded from %s: %d shots, pinned_portrait=%s",
            p, len(d.shots),
            d.pinned_portrait or "(none)",
        )
        return d

    # ── Resolution ─────────────────────────────────────────────── #

    def resolve(self, shot_index: int, review_dir: Path) -> Path | None:
        """
        Return an absolute path to the image the renderer should use
        for `shot_index`, or None if no decision was recorded.

        Resolution order:
            1. `override`     — user dropped a file in overrides/
            2. pinned portrait (for `portrait` visuals only)
            3. `chosen_file`  — pre-downloaded candidate the dossier
                                designated as best
            4. None           — caller falls back to the Fetcher
                                waterfall at render time
        """
        review_dir = Path(review_dir).resolve()
        shot = self.shots.get(shot_index)
        if shot is None:
            return None

        # 1. Explicit override
        if shot.override:
            p = (review_dir / shot.override).resolve()
            if p.exists():
                log.debug("Shot %d: override hit %s", shot_index, p)
                return p
            log.warning(
                "Shot %d: override declared %s but file is missing — "
                "ignoring and falling through",
                shot_index, p,
            )

        # 2. Pinned portrait, for portrait shots only
        if shot.visual == "portrait" and self.pinned_portrait:
            p = (review_dir / self.pinned_portrait).resolve()
            if p.exists():
                log.debug("Shot %d: pinned-portrait hit %s", shot_index, p)
                return p
            log.warning(
                "Shot %d: pinned_portrait %s missing — ignoring",
                shot_index, p,
            )

        # 3. Pre-downloaded candidate the dossier marked as chosen
        if shot.chosen_file:
            p = (review_dir / shot.chosen_file).resolve()
            if p.exists():
                log.debug("Shot %d: chosen-file hit %s", shot_index, p)
                return p

        return None


def is_image_shot(visual: str) -> bool:
    return visual in _IMAGE_VISUALS


def shot_folder_name(shot_index: int, visual: str) -> str:
    """Stable directory name for one shot's candidates."""
    return f"shot_{shot_index:02d}_{visual}"


def write_readme(review_dir: Path) -> None:
    """Write a human-friendly usage guide into the review directory."""
    text = """\
LAMAHAT — Phase 3 review dossier
=================================

This directory is the contract between the prebuild pass (which fetches
and scores candidate images) and the render pass (which burns the
video).  Edit it freely between the two passes.

WHAT'S IN HERE
--------------
  decisions.json       The file you edit.  Hand-edit-friendly JSON.
  overrides/           Drop your own .jpg / .png files here.
  shot_NN_VISUAL/      One folder per image-needing shot.  Contains:
                       - context.txt     what this shot is about
                       - candidates.json same as decisions.json["shots"]["NN"]["candidates"]
                       - SOURCE_X.jpg    the actual downloaded candidate

WHAT YOU CAN CHANGE IN decisions.json
-------------------------------------
Per shot:
  * "chosen"        Move to a different candidate by copying its
                    "source:title" string here.
  * "override"      Set to a file path under this directory (typically
                    "overrides/shot_NN.jpg") to use your own image.
                    Overrides win over everything else.

Global:
  * "pinned_portrait"  Set to a path under overrides/ (e.g.
                       "overrides/character.jpg") to use one canonical
                       portrait at every "portrait" shot.

EXAMPLES
--------
Use my own picture for shot 3:
  1. Save your image as overrides/shot_03.jpg
  2. In decisions.json, find shot "3" and set:
        "override": "overrides/shot_03.jpg"

Use the same Jafar al-Askari portrait at every "portrait" shot:
  1. Save the photo as overrides/character.jpg
  2. In decisions.json, set:
        "pinned_portrait": "overrides/character.jpg"

Swap to the Wikimedia candidate instead of the Pexels one:
  1. Open shot_NN_portrait/candidates.json
  2. Copy the "source:title" of the candidate you prefer
  3. In decisions.json, paste it into "chosen"

THEN
----
  python render_plan.py --plan ... --review-dir <this-dir> --output ...
"""
    (review_dir / "README.txt").write_text(text, encoding="utf-8")
