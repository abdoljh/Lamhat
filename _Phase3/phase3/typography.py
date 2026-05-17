"""
Phase 3 — Arabic typography card renderer.

Renders the five template families used by the planner's typography
shots into static PNG images that downstream FFmpeg passes turn into
video clips.

Design system: Family A (Aljazeera-editorial)
----------------------------------------------
- Cream/charcoal palette, restrained ornament, generous white space
- Hierarchy through scale + weight, never colour
- Text is always the hero; decoration sits behind or beside

Templates
---------
- title_card        Opens/closes the video.  Book title + optional subtitle.
- chapter_heading   Treated as a styled section_mark.
- section_mark      Interstitial breath between thematic blocks.
- pull_quote        The workhorse — magazine-style pull quote.
- name_reveal       Documentary credit-style name introduction.
- date_stamp        Massive year/date for chapter markers.

Arabic handling
---------------
- arabic_reshaper handles glyph contextual forms (initial/medial/final)
- python-bidi applies the Unicode Bidirectional Algorithm so the text
  is rendered right-to-left as it should be
- Both libraries are required because Pillow can't do either natively

Files written
-------------
PNG at full video resolution (default 1920x1080).  Returned path is
absolute.  Caller is responsible for cleanup if rendered to a temp
directory.

Tuning
------
Every visual constant lives in the DESIGN block at the top of this
file.  Don't bury magic numbers inside template functions.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.features import check as pil_check

log = logging.getLogger(__name__)

# Pillow 10+ with libraqm shapes Arabic correctly via direction='rtl'.
# When raqm is unavailable (some older builds, certain Pillow wheels),
# we fall back to manual reshaping + bidi reordering — slower and slightly
# less accurate (no kashida, no advanced ligatures) but correct enough.
_HAS_RAQM = pil_check("raqm")
if not _HAS_RAQM:
    try:
        import arabic_reshaper       # type: ignore
        from bidi.algorithm import get_display  # type: ignore
        log.warning(
            "Pillow lacks libraqm; falling back to arabic_reshaper + "
            "python-bidi.  Arabic shaping will be slightly less accurate. "
            "For best results: install Pillow with raqm support."
        )
    except ImportError as exc:
        raise RuntimeError(
            "Pillow was built without libraqm support, and "
            "arabic_reshaper / python-bidi are not installed.  "
            "Either install Pillow with raqm (best) or run "
            "`pip install arabic_reshaper python-bidi` as a fallback."
        ) from exc


# ══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Family A
# Edit values here to retune visual feel.  No magic numbers below this point.
# ══════════════════════════════════════════════════════════════════════════

# ── Palette (RGB) ─────────────────────────────────────────────────────── #
# Bone-cream backgrounds, charcoal text.  Two cream values let title and
# section cards subtly differentiate.

CREAM_LIGHT  = (244, 239, 230)   # #F4EFE6 — title cards (lightest)
CREAM_MEDIUM = (237, 231, 218)   # #EDE7DA — section marks (warmer)
CREAM_DEEP   = (228, 220, 204)   # #E4DCCC — pull quotes (most saturated)
CHARCOAL     = (38,  35,  31)    # #26231F — primary text
GRAPHITE     = (88,  82,  72)    # #585248 — secondary text
WARM_GREY    = (140, 130, 115)   # #8C8273 — dividers & ornament

# ── Typography ────────────────────────────────────────────────────────── #
# Font sizes are expressed as a fraction of video height so they scale
# from 720p to 1080p to 4K without re-tuning.

# Per-weight filename aliases.  Each weight slot maps to a list of
# filenames that have appeared in the wild across the various Amiri
# packagings:
#   - Upstream releases (≥0.114, 2020-onward): use {Italic, BoldItalic}.
#   - Older Debian / Ubuntu packages (e.g. fonts-hosny-amiri 0.113-1
#     on Ubuntu jammy / Colab): ship {Slanted, BoldSlanted} instead.
# Both name sets refer to the same fonts.  Discovery accepts whichever
# exists in a given directory.
_FONT_ALIASES = {
    "regular":     ["Amiri-Regular.ttf"],
    "bold":        ["Amiri-Bold.ttf"],
    "italic":      ["Amiri-Italic.ttf", "Amiri-Slanted.ttf"],
    "bold_italic": ["Amiri-BoldItalic.ttf", "Amiri-BoldSlanted.ttf"],
    "quran":       ["AmiriQuran.ttf"],
}

# Discovery considers a directory usable when *at minimum* regular and
# bold are present.  italic and bold_italic improve attribution lines
# under pull quotes but the renderer falls back to regular when they
# are absent (see _font()).  Quran weight is purely decorative.
_REQUIRED_WEIGHTS = ("regular", "bold")


def _resolve_weights(font_dir: Path) -> dict | None:
    """Try to satisfy every weight slot from a candidate directory.

    Returns a {weight: absolute path} dict if at least the required
    weights are present, otherwise None.  Optional weights that aren't
    present are simply omitted from the result.
    """
    resolved: dict[str, Path] = {}
    for weight, aliases in _FONT_ALIASES.items():
        for fname in aliases:
            p = font_dir / fname
            if p.exists():
                resolved[weight] = p
                break
    if all(w in resolved for w in _REQUIRED_WEIGHTS):
        return {w: str(p) for w, p in resolved.items()}
    return None


def _discover_amiri_fonts() -> dict:
    """
    Locate the Amiri font files on the current system.

    Discovery order, highest-priority first:
      1. **Repo-bundled `fonts/` directory.**  The Lamahat repo ships
         `_Phase3/fonts/Amiri-{Regular,Bold,Italic,BoldItalic}.ttf`
         as the authoritative copies.  Discovery looks there first so
         renders are reproducible regardless of OS, package version,
         or network availability.  Probed at, in order:
           - `<package_parent>/fonts`     (next to `phase3/`)
           - `<package_parent>/../fonts`  (one level above)
           - `<CWD>/fonts`                (when invoked from repo root)
           - `<CWD>/_Phase3/fonts`        (when invoked from repo root)
      2. `LAMAHAT_AMIRI_DIR` env var (explicit user override).
      3. `fc-match` (fontconfig) — works on Linux/macOS once `fc-cache`
         has registered Amiri.
      4. Well-known install paths covering Debian/Ubuntu, Fedora, Arch,
         macOS Homebrew, Google Colab `/content/fonts`, conda envs.
      5. Download from upstream → `~/.cache/lamahat/fonts/`.

    All strategies use `_resolve_weights()`, which accepts both the
    modern (Italic/BoldItalic) and legacy (Slanted/BoldSlanted) Amiri
    filename conventions.  Only `regular` + `bold` are strictly
    required; italic and quran are optional.

    Returns
    -------
    dict mapping the keys 'regular', 'bold' (always) and optionally
    'italic', 'bold_italic', 'quran' to absolute font paths.
    """
    import os
    import shutil
    import subprocess

    tried: list[str] = []

    def _try(label: str, directory: Path) -> dict | None:
        if not directory or not directory.exists():
            tried.append(f"  - {label}: {directory} (does not exist)")
            return None
        resolved = _resolve_weights(directory)
        if resolved:
            log.info("Amiri fonts loaded via %s: %s", label, directory)
            return resolved
        tried.append(f"  - {label}: {directory} "
                     f"(missing required weights regular/bold)")
        return None

    # ── Strategy 1: repo-bundled fonts/ directory ─────────────────── #
    # __file__ lives at <repo>/_Phase3/phase3/typography.py, so:
    #   parents[0] = phase3/
    #   parents[1] = _Phase3/   ← repo's bundled fonts/ sits here
    #   parents[2] = <repo root>
    pkg_dir = Path(__file__).resolve().parent
    repo_relative_candidates = [
        ("repo fonts/ (next to phase3)",   pkg_dir.parent / "fonts"),
        ("repo fonts/ (one level up)",     pkg_dir.parent.parent / "fonts"),
        ("CWD fonts/",                     Path.cwd() / "fonts"),
        ("CWD _Phase3/fonts/",             Path.cwd() / "_Phase3" / "fonts"),
        # Colab convention: notebook copies project root into /content/.
        # If cell 0 doesn't copy fonts/ but the original is on Drive,
        # fall through to the live Drive mount.  Cheap when not present.
        ("Colab Drive /content/_Phase3/fonts",
         Path("/content/_Phase3/fonts")),
        ("Colab Drive (mounted) _Phase3/fonts",
         Path("/content/drive/MyDrive/_Phase3/fonts")),
    ]
    for label, d in repo_relative_candidates:
        found = _try(label, d)
        if found:
            return found

    # ── Strategy 2: explicit override ──────────────────────────────── #
    override = os.environ.get("LAMAHAT_AMIRI_DIR")
    if override:
        found = _try("LAMAHAT_AMIRI_DIR env var", Path(override))
        if found:
            return found

    # ── Strategy 3: fontconfig ─────────────────────────────────────── #
    fc_match = shutil.which("fc-match")
    if fc_match:
        try:
            result = subprocess.run(
                [fc_match, "-f", "%{file}", "Amiri:style=Regular"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                amiri_regular_path = Path(result.stdout.strip())
                # fc-match returns *something* even when Amiri isn't
                # installed (it picks the closest substitute).  Guard
                # against accepting a fallback like DejaVu.
                if "amiri" in amiri_regular_path.name.lower():
                    found = _try("fc-match", amiri_regular_path.parent)
                    if found:
                        return found
                else:
                    tried.append(
                        f"  - fc-match: returned {amiri_regular_path} "
                        f"(not an Amiri file — fontconfig substituted)"
                    )
            else:
                tried.append(
                    f"  - fc-match: exit={result.returncode} "
                    f"stdout={result.stdout.strip()!r}"
                )
        except Exception as exc:
            tried.append(f"  - fc-match: raised {exc!r}")
    else:
        tried.append("  - fc-match: binary not on PATH")

    # ── Strategy 4: well-known install paths ───────────────────────── #
    well_known = [
        # Debian/Ubuntu (fonts-hosny-amiri package)
        Path("/usr/share/fonts/opentype/fonts-hosny-amiri"),
        # Fedora/RHEL
        Path("/usr/share/fonts/amiri"),
        Path("/usr/share/fonts/google-amiri-fonts"),
        # Arch
        Path("/usr/share/fonts/TTF"),
        Path("/usr/share/fonts/OTF"),
        # macOS (Homebrew)
        Path("/opt/homebrew/share/fonts"),
        Path("/usr/local/share/fonts"),
        # Per-user fonts
        Path.home() / ".local/share/fonts",
        Path.home() / ".fonts",
        Path.home() / "Library/Fonts",   # macOS
        # Common Colab/notebook environments
        Path("/content/fonts"),
        Path("/content/_Phase3/fonts"),
        # Conda envs
        Path(os.environ.get("CONDA_PREFIX", "/nonexistent")) / "share/fonts",
    ]
    for d in well_known:
        if not d.exists():
            continue
        # Direct hit first
        found = _try(f"system path {d}", d)
        if found:
            return found
        # Recursive search — package directories sometimes nest the
        # fonts under a versioned subdirectory.
        for amiri_regular in d.rglob("Amiri-Regular.ttf"):
            found = _try(f"system path (nested) {amiri_regular.parent}",
                         amiri_regular.parent)
            if found:
                return found

    # ── Strategy 5: download from upstream ─────────────────────────── #
    log.warning(
        "Amiri not found in repo, env override, fontconfig, or system "
        "paths — falling back to upstream download.  Searched:\n%s",
        "\n".join(tried) or "  (no paths searched)",
    )
    return _download_amiri()


def _download_amiri() -> dict:
    """
    Download the Amiri font release from the official upstream
    (github.com/aliftype/amiri) into ~/.cache/lamahat/fonts/.

    Last-resort path.  When the repo's bundled `fonts/` directory is
    present (or any system install is correctly configured), this is
    never reached.  When it is reached and succeeds, a one-time
    download caches the fonts so subsequent invocations skip it.
    """
    import io
    import urllib.request
    import zipfile

    cache_dir = Path.home() / ".cache" / "lamahat" / "fonts" / "amiri-1.003"
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/aliftype/amiri/releases/download/1.003/Amiri-1.003.zip"
        log.info("Downloading Amiri from %s", url)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Lamahat/1.0 (https://github.com/abdoljh/Lamahat)",
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except Exception as exc:
            raise RuntimeError(
                f"Amiri font is not installed and could not be downloaded "
                f"from {url}: {exc}.  Recommended fix: ensure the repo's "
                f"`_Phase3/fonts/` directory is present at runtime (this is "
                f"the primary, network-free source).  Alternatives:\n"
                f"  Debian/Ubuntu: sudo apt install fonts-hosny-amiri\n"
                f"  Fedora:        sudo dnf install amiri-fonts\n"
                f"  macOS:         brew install --cask font-amiri\n"
                f"  Override:      set LAMAHAT_AMIRI_DIR=/path/to/amiri/ttf/dir"
            ) from exc

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                # Flatten the zip's nested structure — we just want the
                # .ttf files at the top level of our cache dir.
                fname = Path(name).name
                if fname.endswith(".ttf"):
                    (cache_dir / fname).write_bytes(zf.read(name))

    resolved = _resolve_weights(cache_dir)
    if resolved is None:
        raise RuntimeError(
            f"Downloaded Amiri archive at {cache_dir} is missing the "
            f"required regular/bold weights.  Files present: "
            f"{sorted(p.name for p in cache_dir.iterdir())}"
        )
    log.info("Amiri fonts downloaded → %s", cache_dir)
    return resolved


FONT_PATHS = _discover_amiri_fonts()

# Per-template size hints, as fraction of video height
SIZES = {
    "title_main":      0.085,    # ~92 px @ 1080p
    "title_sub":       0.030,    # ~32 px
    "section_main":    0.060,    # ~65 px
    "section_sub":     0.022,    # ~24 px
    "pull_quote_lg":   0.075,    # ~81 px — used for short quotes
    "pull_quote_md":   0.058,    # ~63 px — medium quotes
    "pull_quote_sm":   0.044,    # ~48 px — long quotes
    "pull_quote_attr": 0.022,    # ~24 px — attribution line
    "name_main":       0.070,    # ~76 px
    "name_sub":        0.028,    # ~30 px
    "date_huge":       0.220,    # ~238 px — massive
    "date_sub":        0.036,    # ~39 px
    "quote_mark":      0.450,    # ~486 px — decorative behind quote
}

# Word-count thresholds for pull_quote font sizing
PULL_QUOTE_THRESHOLDS = (8, 13)  # ≤8 words → lg, ≤13 → md, more → sm

# ── Layout ────────────────────────────────────────────────────────────── #

MARGINS = {
    "horizontal_pct": 0.10,   # 10 % side margins on text
    "vertical_pct":   0.18,   # 18 % top/bottom margin
}

LINE_HEIGHT_MULT = 1.45       # multiplier on font size for line spacing

# ── Ornament & rules ─────────────────────────────────────────────────── #

RULE_THICKNESS_PX_1080 = 2    # hairline rule thickness at 1080p
RULE_OPACITY = 0.45           # 0 = transparent, 1 = opaque
RULE_LENGTH_PCT = 0.30        # rule length as fraction of video width

SECTION_DIAMOND_SIZE = 12      # diamond ornament size at 1080p

# ── Quote marks ──────────────────────────────────────────────────────── #
# Arabic uses « » (French guillemets) traditionally.

QUOTE_OPEN  = "\u00AB"   # «
QUOTE_CLOSE = "\u00BB"   # »
DECORATIVE_QUOTE_OPACITY = 0.08

# ── Paper grain ──────────────────────────────────────────────────────── #
# Subtle noise overlay gives the cream backgrounds a tactile feel.
# Without it, large flat colour areas look digital.

GRAIN_OPACITY = 0.06
GRAIN_SEED = 42               # deterministic per render

# ══════════════════════════════════════════════════════════════════════════
# END DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════


TemplateName = Literal[
    "title_card", "section_mark", "chapter_heading",
    "pull_quote", "name_reveal", "date_stamp",
]


# ── Public API ──────────────────────────────────────────────────────────── #

@dataclass
class TypographySpec:
    """Inputs needed to render one typography card."""
    template: TemplateName
    text: str                       # the primary Arabic text
    subtitle: str = ""              # optional second line (e.g. dates)
    width: int = 1920
    height: int = 1080


def render(spec: TypographySpec, out_path: Path) -> Path:
    """
    Render one typography card to disk and return the output path.

    All renderers follow the same shape:
      1. Make the base canvas (cream + grain)
      2. Place template-specific ornament (rules, decorative quote marks)
      3. Place the typography on top
      4. Save as PNG
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = _RENDERERS.get(spec.template, _render_pull_quote)
    image = renderer(spec)
    image.save(out_path, format="PNG", optimize=True)
    log.debug("Rendered %s typography card → %s", spec.template, out_path.name)
    return out_path


# ── Shared helpers ──────────────────────────────────────────────────────── #

def _font(weight: str, size_px: int) -> ImageFont.FreeTypeFont:
    """Load an Amiri weight at a specific pixel size.

    Falls back to 'regular' if the requested weight isn't available
    (e.g. some font packages ship without italic).
    """
    path = FONT_PATHS.get(weight) or FONT_PATHS.get("regular")
    if path is None:
        raise RuntimeError(
            "No Amiri font available — _discover_amiri_fonts() returned empty. "
            "This shouldn't happen if discovery succeeded at import."
        )
    return ImageFont.truetype(path, size_px)


def _size(spec: TypographySpec, key: str) -> int:
    """Compute pixel size for a SIZES entry given the spec's height."""
    return max(12, int(spec.height * SIZES[key]))


def _prepare_for_pillow(text: str) -> str:
    """
    Prepare Arabic text for Pillow rendering.

    With libraqm: pass the raw Unicode through — Pillow handles shaping
    + bidi when direction='rtl' is specified.

    Without libraqm: reshape (Unicode → presentation forms) and apply
    bidi reordering manually so Pillow's basic FreeType path renders
    the glyphs in the correct visual order.
    """
    if not text:
        return ""
    if _HAS_RAQM:
        return text
    return get_display(arabic_reshaper.reshape(text))


def _measure(draw: ImageDraw.ImageDraw, text: str,
             font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Return (width, height) of `text` rendered with `font` in RTL."""
    if not text:
        return 0, 0
    prepared = _prepare_for_pillow(text)
    if _HAS_RAQM:
        bbox = draw.textbbox((0, 0), prepared, font=font,
                             direction="rtl", language="ar")
    else:
        bbox = draw.textbbox((0, 0), prepared, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_text_rtl(draw: ImageDraw.ImageDraw, xy: tuple[int, int],
                   text: str, font: ImageFont.FreeTypeFont,
                   fill) -> None:
    """Wrapper around draw.text that handles Arabic shaping."""
    if not text:
        return
    prepared = _prepare_for_pillow(text)
    if _HAS_RAQM:
        draw.text(xy, prepared, font=font, fill=fill,
                  direction="rtl", language="ar")
    else:
        draw.text(xy, prepared, font=font, fill=fill)


def _make_canvas(width: int, height: int, bg_rgb: tuple) -> Image.Image:
    """Create a base canvas with paper grain applied."""
    img = Image.new("RGB", (width, height), bg_rgb)
    _apply_grain(img)
    return img


def _apply_grain(img: Image.Image) -> None:
    """
    Apply a subtle film grain to a base image.  Mutates in place.

    We use deterministic numpy-free noise: small random pixel offsets
    in a low-amplitude pattern.  Cheap and good enough — full Perlin
    noise would be overkill for this background.
    """
    rng = random.Random(GRAIN_SEED)
    w, h = img.size

    # Build a sparse grayscale noise image
    noise_gray = Image.new("L", (w, h), 128)
    pixels = noise_gray.load()
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            pixels[x, y] = 128 + rng.randint(-32, 32)
    noise_gray = noise_gray.filter(ImageFilter.GaussianBlur(radius=0.6))

    # Convert grayscale noise to RGB and blend with the base.
    # Image.blend requires identical modes; we ensure both are RGB.
    noise_rgb = Image.merge("RGB", (noise_gray, noise_gray, noise_gray))
    if img.mode != "RGB":
        rgb_base = img.convert("RGB")
        blended = Image.blend(rgb_base, noise_rgb, GRAIN_OPACITY)
        img.paste(blended.convert(img.mode))
    else:
        blended = Image.blend(img, noise_rgb, GRAIN_OPACITY)
        img.paste(blended)


def _draw_hairline_rule(draw: ImageDraw.ImageDraw,
                       y: int, width: int, height: int,
                       length_pct: float = RULE_LENGTH_PCT) -> None:
    """Draw a centred horizontal hairline rule at vertical position y."""
    rule_len = int(width * length_pct)
    thickness = max(1, int(height * RULE_THICKNESS_PX_1080 / 1080))
    x0 = (width - rule_len) // 2
    x1 = x0 + rule_len
    # Blend charcoal with the background using a semi-transparent overlay
    overlay = Image.new("RGBA", draw._image.size, (0, 0, 0, 0))
    o_draw = ImageDraw.Draw(overlay)
    a = int(255 * RULE_OPACITY)
    o_draw.rectangle([x0, y, x1, y + thickness],
                     fill=(*WARM_GREY, a))
    # paste onto base
    base = draw._image.convert("RGBA")
    base.alpha_composite(overlay)
    draw._image.paste(base.convert("RGB"))


def _draw_diamond(draw: ImageDraw.ImageDraw,
                  cx: int, cy: int, size: int) -> None:
    """Draw a small filled diamond at (cx, cy) — for section_mark ornaments."""
    pts = [(cx, cy - size), (cx + size, cy),
           (cx, cy + size), (cx - size, cy)]
    draw.polygon(pts, fill=WARM_GREY)


def _wrap_text(text: str, max_chars_per_line: int) -> list[str]:
    """
    Break an Arabic string into lines of ≤ max_chars characters,
    respecting word boundaries.

    Note: max_chars is a rough proxy for visual width.  We rely on
    the caller to choose a value that fits the design's column.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= max_chars_per_line:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_centred_lines(draw: ImageDraw.ImageDraw,
                        lines: list[str],
                        font: ImageFont.FreeTypeFont,
                        canvas_size: tuple[int, int],
                        colour: tuple,
                        baseline_y: int,
                        line_height: int) -> int:
    """
    Draw a stack of lines centred horizontally, top-aligned at baseline_y.
    Returns the y coordinate immediately below the last line.
    """
    w, _ = canvas_size
    y = baseline_y
    for line in lines:
        tw, _ = _measure(draw, line, font)
        _draw_text_rtl(draw, ((w - tw) // 2, y), line, font=font, fill=colour)
        y += line_height
    return y


# ── Template: title_card ───────────────────────────────────────────────── #

def _render_title_card(spec: TypographySpec) -> Image.Image:
    """
    Cream background, hairline rule above and below the title, paper grain.
    Used to open and close the video.
    """
    img = _make_canvas(spec.width, spec.height, CREAM_LIGHT)
    draw = ImageDraw.Draw(img)

    main_size = _size(spec, "title_main")
    sub_size  = _size(spec, "title_sub")

    main_font = _font("bold", main_size)
    sub_font  = _font("italic", sub_size)

    # Measure the title to position it vertically centred
    shaped_main = spec.text
    mw, mh = _measure(draw, shaped_main, main_font)

    # Title block fits between two rules
    rule_gap = int(spec.height * 0.04)            # space between rule and text
    sub_gap  = int(spec.height * 0.03)            # gap between title and subtitle

    sub_h = 0
    if spec.subtitle:
        shaped_sub = spec.subtitle
        _, sub_h = _measure(draw, shaped_sub, sub_font)

    total_block_h = mh + (sub_gap + sub_h if spec.subtitle else 0)

    block_top = (spec.height - total_block_h) // 2
    title_y   = block_top
    sub_y     = block_top + mh + sub_gap if spec.subtitle else None

    # Hairline rules above and below the block
    _draw_hairline_rule(draw,
                       y=block_top - rule_gap,
                       width=spec.width, height=spec.height)
    _draw_hairline_rule(draw,
                       y=block_top + total_block_h + rule_gap,
                       width=spec.width, height=spec.height)

    # Re-create draw on possibly-updated image
    draw = ImageDraw.Draw(img)

    # Title text (centred)
    _draw_text_rtl(draw, ((spec.width - mw) // 2, title_y),
              shaped_main, font=main_font, fill=CHARCOAL)

    # Subtitle (centred, smaller, italic)
    if spec.subtitle and sub_y is not None:
        shaped_sub = spec.subtitle
        sw, _ = _measure(draw, shaped_sub, sub_font)
        _draw_text_rtl(draw, ((spec.width - sw) // 2, sub_y),
                  shaped_sub, font=sub_font, fill=GRAPHITE)

    return img


# ── Template: section_mark / chapter_heading ───────────────────────────── #

def _render_section_mark(spec: TypographySpec) -> Image.Image:
    """
    Interstitial pause card.  Slightly warmer cream than title cards,
    a thin horizontal rule below the heading punctuated by a small
    centred diamond.
    """
    img = _make_canvas(spec.width, spec.height, CREAM_MEDIUM)
    draw = ImageDraw.Draw(img)

    main_size = _size(spec, "section_main")
    sub_size  = _size(spec, "section_sub")

    main_font = _font("regular", main_size)
    sub_font  = _font("italic", sub_size)

    shaped_main = spec.text
    mw, mh = _measure(draw, shaped_main, main_font)

    # Block placement — slightly above centre for visual weight
    block_y = int(spec.height * 0.42)
    text_x  = (spec.width - mw) // 2

    _draw_text_rtl(draw, (text_x, block_y), shaped_main, font=main_font, fill=CHARCOAL)

    # Ornament below: hairline rule with a diamond in the middle
    ornament_y = block_y + mh + int(spec.height * 0.04)
    _draw_hairline_rule(draw, y=ornament_y,
                       width=spec.width, height=spec.height,
                       length_pct=0.18)
    # The rule helper paste cycle invalidates our draw handle
    draw = ImageDraw.Draw(img)

    diamond_size = max(4, int(spec.height * SECTION_DIAMOND_SIZE / 1080))
    _draw_diamond(draw, spec.width // 2,
                  ornament_y + max(1, int(spec.height * 0.001)),
                  diamond_size)

    # Optional subtitle below ornament
    if spec.subtitle:
        sub_y = ornament_y + diamond_size + int(spec.height * 0.025)
        shaped_sub = spec.subtitle
        sw, _ = _measure(draw, shaped_sub, sub_font)
        _draw_text_rtl(draw, ((spec.width - sw) // 2, sub_y),
                  shaped_sub, font=sub_font, fill=GRAPHITE)

    return img


# Chapter heading uses the same renderer as section_mark
_render_chapter_heading = _render_section_mark


# ── Template: pull_quote ───────────────────────────────────────────────── #

def _wrap_by_width(draw: ImageDraw.ImageDraw, text: str,
                   font: ImageFont.FreeTypeFont,
                   max_width: int) -> list[str]:
    """
    Break an Arabic string into lines whose rendered width fits
    `max_width` pixels.  Uses real measurements via _measure() rather
    than character-count proxies, which under-estimate width for
    diacritic-heavy text and over-estimate for narrow letters.
    """
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        w, _ = _measure(draw, candidate, font)
        if w <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _render_pull_quote(spec: TypographySpec) -> Image.Image:
    """
    Magazine-style pull quote, Family A (Aljazeera-editorial).

    Composition:
      - Cream backdrop with subtle grain
      - Quote in Amiri Bold, size scaled to word count, centred
      - Small hairline rule below the quote as the only decoration
      - Optional attribution line below the rule in italic

    The decoration is restrained on purpose.  In editorial Arabic
    typography the typography itself carries the weight; ornaments
    that compete with the text feel commercial, not editorial.
    """
    img = _make_canvas(spec.width, spec.height, CREAM_DEEP)
    draw = ImageDraw.Draw(img)

    # Font sizing by word count
    word_count = len(spec.text.split())
    if word_count <= PULL_QUOTE_THRESHOLDS[0]:
        size_key = "pull_quote_lg"
    elif word_count <= PULL_QUOTE_THRESHOLDS[1]:
        size_key = "pull_quote_md"
    else:
        size_key = "pull_quote_sm"

    main_size = _size(spec, size_key)
    attr_size = _size(spec, "pull_quote_attr")
    main_font = _font("bold", main_size)
    attr_font = _font("italic", attr_size)

    # ── Wrap by real width ────────────────────────────────────────── #
    column_w = int(spec.width * (1 - 2 * MARGINS["horizontal_pct"]))
    lines = _wrap_by_width(draw, spec.text, main_font, column_w)

    # ── Compute text block bounds ─────────────────────────────────── #
    line_height = int(main_size * LINE_HEIGHT_MULT)
    block_h = line_height * len(lines)

    # Reserve space below for hairline rule + optional attribution
    # rule_gap is larger than line spacing so descenders (e.g. on ث ج ع)
    # don't collide with the rule.
    rule_gap = int(spec.height * 0.06)
    rule_thick = max(1, int(spec.height * RULE_THICKNESS_PX_1080 / 1080))
    attr_gap = int(spec.height * 0.025)
    attr_h = 0
    if spec.subtitle:
        _, attr_text_h = _measure(draw, f"— {spec.subtitle}", attr_font)
        attr_h = attr_gap + attr_text_h

    total_h = block_h + rule_gap + rule_thick + attr_h
    block_top = (spec.height - total_h) // 2

    # ── Draw the quote ────────────────────────────────────────────── #
    last_y = _draw_centred_lines(
        draw, lines, main_font,
        canvas_size=(spec.width, spec.height),
        colour=CHARCOAL,
        baseline_y=block_top,
        line_height=line_height,
    )

    # ── Hairline rule below ───────────────────────────────────────── #
    rule_y = last_y + rule_gap
    _draw_hairline_rule(draw, y=rule_y,
                       width=spec.width, height=spec.height,
                       length_pct=0.14)
    draw = ImageDraw.Draw(img)

    # ── Optional attribution below the rule ───────────────────────── #
    if spec.subtitle:
        attr_y = rule_y + rule_thick + attr_gap
        attr_text = f"— {spec.subtitle}"
        aw, _ = _measure(draw, attr_text, attr_font)
        _draw_text_rtl(draw, ((spec.width - aw) // 2, attr_y),
                       attr_text, font=attr_font, fill=GRAPHITE)

    return img


# ── Template: name_reveal ──────────────────────────────────────────────── #

def _render_name_reveal(spec: TypographySpec) -> Image.Image:
    """
    Documentary credit-style card: name in bold above a hairline rule,
    optional dates/role in italic below.
    """
    img = _make_canvas(spec.width, spec.height, CREAM_LIGHT)
    draw = ImageDraw.Draw(img)

    name_size = _size(spec, "name_main")
    sub_size  = _size(spec, "name_sub")
    name_font = _font("bold", name_size)
    sub_font  = _font("italic", sub_size)

    shaped_name = spec.text
    nw, nh = _measure(draw, shaped_name, name_font)

    # Compose block: name, rule, optional subtitle
    rule_gap   = int(spec.height * 0.03)
    sub_gap    = int(spec.height * 0.03)
    rule_thick = max(1, int(spec.height * RULE_THICKNESS_PX_1080 / 1080))

    sub_h = 0
    if spec.subtitle:
        shaped_sub = spec.subtitle
        _, sub_h = _measure(draw, shaped_sub, sub_font)

    total_h = nh + rule_gap + rule_thick + (sub_gap + sub_h if spec.subtitle else 0)
    block_top = (spec.height - total_h) // 2

    # Name
    _draw_text_rtl(draw, ((spec.width - nw) // 2, block_top),
              shaped_name, font=name_font, fill=CHARCOAL)

    # Hairline rule
    rule_y = block_top + nh + rule_gap
    _draw_hairline_rule(draw, y=rule_y,
                       width=spec.width, height=spec.height,
                       length_pct=0.22)
    draw = ImageDraw.Draw(img)

    # Subtitle
    if spec.subtitle:
        sub_y = rule_y + rule_thick + sub_gap
        shaped_sub = spec.subtitle
        sw, _ = _measure(draw, shaped_sub, sub_font)
        _draw_text_rtl(draw, ((spec.width - sw) // 2, sub_y),
                  shaped_sub, font=sub_font, fill=GRAPHITE)

    return img


# ── Template: date_stamp ───────────────────────────────────────────────── #

def _render_date_stamp(spec: TypographySpec) -> Image.Image:
    """
    Massive year/date stamp.  The date dominates the frame; optional
    small descriptor above sets context.
    """
    img = _make_canvas(spec.width, spec.height, CREAM_LIGHT)
    draw = ImageDraw.Draw(img)

    date_size = _size(spec, "date_huge")
    sub_size  = _size(spec, "date_sub")
    date_font = _font("bold", date_size)
    sub_font  = _font("regular", sub_size)

    shaped_date = spec.text
    dw, dh = _measure(draw, shaped_date, date_font)

    # Descriptor sits above the date if provided, else just date centred
    descriptor_gap = int(spec.height * 0.025)
    desc_h = 0
    if spec.subtitle:
        shaped_sub = spec.subtitle
        _, desc_h = _measure(draw, shaped_sub, sub_font)

    total_h = (desc_h + descriptor_gap if spec.subtitle else 0) + dh
    block_top = (spec.height - total_h) // 2

    cursor = block_top
    if spec.subtitle:
        shaped_sub = spec.subtitle
        sw, _ = _measure(draw, shaped_sub, sub_font)
        _draw_text_rtl(draw, ((spec.width - sw) // 2, cursor),
                  shaped_sub, font=sub_font, fill=GRAPHITE)
        cursor += desc_h + descriptor_gap

    _draw_text_rtl(draw, ((spec.width - dw) // 2, cursor),
              shaped_date, font=date_font, fill=CHARCOAL)

    return img


# ── Renderer dispatch ──────────────────────────────────────────────────── #

_RENDERERS = {
    "title_card":       _render_title_card,
    "section_mark":     _render_section_mark,
    "chapter_heading":  _render_chapter_heading,
    "pull_quote":       _render_pull_quote,
    "name_reveal":      _render_name_reveal,
    "date_stamp":       _render_date_stamp,
}
