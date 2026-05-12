# Lamahat — Master Plan & Session Handoff

## Project Vision

Convert Arabic books (PDF) into high-impact, 3-to-5-minute video summaries —
long enough to deliver real value, short enough for modern attention spans.
The output is a fully automated MP4: Arabic TTS voice, relevant background
visuals with motion, and burned-in Arabic subtitles. Every phase is built to
run on Streamlit Community Cloud (1 GB RAM, no GPU).

Working repo: **abdoljh/Lamahat** · Streamlit Community Cloud deployment.
Runtime: **Python 3.12.13** (confirmed from Cloud logs — do NOT assume 3.14).

---

## Four-Phase Architecture

| Phase | Name | Goal | Status |
|-------|------|------|--------|
| 1a | PDF Preprocessing & OCR | PDF → strip margins → page images → Kraken OCR → normalised text | ✅ **Complete** |
| 1b | Chunking & Summarisation | Normalised text → semantic chunks → 625–850-word video script | ✅ **Complete** |
| 2 | Audio Synthesis (TTS) | Script → Arabic MP3 via gTTS (ElevenLabs next) | ✅ **Working** (gTTS) |
| 3 | Visual Generation | Script + audio → final MP4 with visuals, voice, subtitles | 🔧 **In Progress** |
| 4 | Workflow Integration | One-click pipeline: PDF → finished video | ✅ **Complete** (follows Phase 3) |

---

## Repo Structure

```
streamlit_app.py          # Streamlit entrypoint (Phases 1–3 UI)
phase3_run.py             # Standalone Phase 3 CLI (no Streamlit required)
phase1/
  __init__.py             # Exports Phase1Pipeline, Phase1aPipeline, Phase1Config, etc.
  pipeline.py             # Phase1aPipeline (8-step) + Phase1bPipeline + Phase1Pipeline
  core/
    header_footer.py      # Margin detection + strip_pdf() + detect_margins()
    page_export.py        # export_pages_as_images() + extract_footers_pdf()
    image_extract.py      # extract_images() — pixel-domain photo extraction
    kraken_engine.py      # Kraken OCR engine wrapper (Arabic, apt-20221130 model)
    ingestor.py           # PDF ingestion (PyMuPDF) — digital + scanned, RTL
    ocr_engine.py         # Tesseract / EasyOCR wrapper (legacy, not used in 1a)
    normalizer.py         # Arabic text normalisation (lam-alef, Farsi Yeh, noise)
    chunker.py            # Semantic chunking (~180 lines)
    diacritizer.py        # Mishkal / Farasa wrapper
    summarizer.py         # Hierarchical summarisation + script generation
    output_writer.py      # JSON + TXT serialisation
phase2/
  __init__.py
  tts.py                  # gTTS backend; ElevenLabs stub (NotImplementedError)
phase3/
  __init__.py             # generate_background_video() full pipeline
  parser.py               # Script section splitter + duration estimator
  keywords.py             # Claude Haiku: search terms + key phrases per section
  wikimedia.py            # Wikimedia Commons image fetcher + Claude vision scoring
  pexels.py               # Pexels video clip fetcher
  effects.py              # Ken Burns (zoompan) + trim + probe_duration
  compositor.py           # Section clips → crossfade → grade → mux
  subtitler.py            # Multi-layer ASS subtitle generator
lightning-compat/         # Local shim: proxies lightning → pytorch-lightning==2.6.1
packages.txt              # Streamlit Cloud apt deps (ffmpeg, fonts-hosny-amiri, etc.)
requirements.txt          # Python deps (torch/lightning/kraken all active on Python 3.12)
output/                   # Pipeline outputs; gitignored in production
samples/                  # Test PDFs (Al-Askari, preface, sample docs)
```

---

## Phase 1a — PDF Preprocessing & OCR ✅

### What it does (8 steps in `Phase1aPipeline.run()`)
1. **Strip headers/footers** — `header_footer.strip_pdf()` sets CropBox on each page
2. **Export page images** — `page_export.export_pages_as_images()` at configurable DPI
3. **Bundle into ZIP** — `_bundle_to_zips()` splits across multiple ZIPs if `zip_split_mb` exceeded
4. **Extract footers** — `page_export.extract_footers_pdf()` on the ORIGINAL PDF → labeled PDF + images ZIP
5. **Extract photographs** — `image_extract.extract_images()` → pixel-domain photo segmentation + captions ZIP
6. **Kraken OCR** — `kraken_engine.ocr_page()` on each exported page image
7. **Normalise Arabic text** — `ArabicTextNormalizer`
8. **Save output files** — corrected TXT + normalised TXT + structured JSON

### Four modes (matching OCR-me)
| Mode | Strip | OCR | Footers/Photos |
|------|-------|-----|----------------|
| `single_book` | ✓ | ✓ | optional |
| `raw_export` | ✗ | ✗ | ✗ |
| `batch` | ✓ | ✓ | optional (caller iterates PDFs) |
| `visual` | ✓ | ✓ | optional (per-page UI is Streamlit-only) |

### Phase1Config key fields
```python
mode: str = "single_book"          # see table above
strip_margins: bool = True
hf_dpi: int = 300                  # DPI for margin detection
export_dpi: int = 400              # DPI for page image export
include_footers: bool = True
include_photos: bool = False
zip_split_mb: float = 250.0        # 0 = no split
ocr_backend: str = "kraken"        # "kraken" | "none"
kraken_bidi: str = "auto"          # "auto" | "R" | "L" | "off"
kraken_threshold: float = 0.5      # NLBin binarization threshold
kraken_pad: int = 16
kraken_autocast: bool = False
kraken_text_direction: str = "horizontal-rl"
kraken_no_legacy_polygons: bool = False
```

### Phase1aResult key fields
```python
pages_zip_paths: list[Path]         # one or more ZIP parts
footers_pdf_path: Optional[Path]
footers_zip_path: Optional[Path]
n_footer_pages: int
photos_zip_path: Optional[Path]
n_photos: int
pages_zip_path: Optional[Path]     # compat property → pages_zip_paths[0]
```

### Kraken engine (`phase1/core/kraken_engine.py`) — critical design
The engine was completely rewritten to fix a crash in kraken 7.0.1.

**Key insight**: `blla.segment` needs the **binarized** image; `rpred.rpred` needs
the **original RGB** image. Passing binarized to rpred causes garbled output.

```python
def ocr_page(model, pil_img, *, threshold=0.5, ...):
    orig_rgb = pil_img.convert("RGB")          # → rpred (recognition)
    bw_img   = binarize_page(pil_img, threshold) # → blla.segment (layout)
    seg      = blla.segment(bw_img, text_direction=..., **_seg_extra)
    preds    = rpred.rpred(model, orig_rgb, seg, ...)  # NOT bw_img
```

**`_sig_params(fn)`** — runtime `inspect.signature` check used to guard every
optional kwarg before passing to kraken functions.  Kraken 7.0.x removed
`no_legacy_polygons` from `blla.segment` and `autocast` from `rpred.rpred`.
The helper avoids hard-coded parameter names that differ between versions:

```python
def _sig_params(fn) -> set:
    try:
        return set(inspect.signature(fn).parameters)
    except Exception:
        return set()

_seg_p = _sig_params(blla.segment)
if "autocast" in _seg_p:
    _seg_extra["autocast"] = autocast
if "no_legacy_polygons" in _seg_p and no_legacy_polygons:
    _seg_extra["no_legacy_polygons"] = True

_rpred_p   = _sig_params(rpred.rpred)
_bidi_kwarg = "bidi_reordering" if "bidi_reordering" in _rpred_p else "bidi_reorder"
if "autocast" in _rpred_p:
    _rpred_extra["autocast"] = autocast
```

**`binarize_page()`** — tries kraken's own `kbin.nlbin` first (most accurate),
falls back to scipy NLBin port, then simple global threshold.

**Pipeline call site** (`pipeline.py` step 6): passes the original image to
`ocr_page()` with `threshold=` kwarg; no pre-binarization in the caller.

### Requirements on Python 3.12 (confirmed)
- `torch>=2.4.0,<=2.10.0` — installed as transitive dep of kraken
- `lightning @ ./lightning-compat` — local shim (all PyPI lightning versions
  were quarantined 2026-04-30); proxies to `pytorch-lightning==2.6.1`
- `kraken==7.0.1` — uncommented in `requirements.txt`
- Streamlit Cloud uses **Python 3.12.13** (set in Advanced settings at first deploy)

### KrakenNotAvailableError
Raised by `load_model()` when `import kraken` fails (e.g., wrong Python version).
The sidebar in `streamlit_app.py` catches `_KRAKEN_AVAILABLE = False` and shows
a Python 3.12 warning. The pipeline catches it separately from generic exceptions
so the error message is passed through cleanly as a warning.

---

## Phase 1b — Chunking & Summarisation ✅

### What it does
1. Semantic chunking (default 1500 tokens, 200 overlap)
2. Hierarchical summarisation: Reader (Haiku, per chunk) → Consolidator (Haiku) → Scriptwriter (Sonnet) → Editor/Scorer (Haiku, up to 2 retries)
3. Outputs: `*_phase1.json`, `*_phase1.txt`, `*_phase1_raw.txt`, `book_script.txt`, `book_script_diacritized.txt`, `book_script_metadata.json`

### Key design decisions
- **Cost strategy**: Haiku for all bulk/scoring work; Sonnet only for the final script (~$0.05 per book).
- **Word-count gate**: 625–850 words. Scripts outside range trigger a targeted retry.
- **max_tokens = 3500** for Scriptwriter (Arabic ~4.2 tokens/word; 850 words ≈ 3570 tokens).
- **Diacritisation**: Mishkal applied only to the final script, never to raw OCR text.
- **No hallucinated names**: Scriptwriter is forbidden from inventing names not in the outline.
- **Hex-Placeholder Technique**: Used in `ingestor.py` to handle lam-alef ligature extraction from PDF spans without breaking RTL text ordering.

### Script structure (4 required sections)
1. Cinematic opening hook
2. Three thematic points with examples
3. Reflective closing
4. Formal book presentation (title + call to action)

### Validated on
- Al-Askari Memoirs (255-page scanned Arabic book, split into 2 PDFs)
- Score: 41/50 · 629 words · 0 retries

---

## Phase 2 — Audio Synthesis ✅ (partial)

### What works
- **gTTS** (`lang='ar'`): free, no API key, produces Arabic MP3 in seconds.
- Streamlit UI: choose script source (Phase 1 session or upload `.txt`), plain vs. diacritized variant, generate + download MP3.

### What is needed next
- **ElevenLabs** integration (Chaouki voice) for broadcast-quality Arabic TTS.
  - Stub exists in `phase2/tts.py` (`NotImplementedError`).
  - Needs: `ELEVENLABS_API_KEY` secret + voice ID in UI, then call ElevenLabs REST API.
  - Priority: implement after Phase 3 visual quality is stable.

---

## Phase 3 — Visual Generation 🔧 (IN PROGRESS — THE CORNERSTONE)

### What works today (all implemented ✅)

| Feature | Implementation |
|---------|---------------|
| Section parsing | Regex-based Arabic section detection (`parser.py`) |
| Keyword + key phrase generation | Claude Haiku per section; book title + character name as context (`keywords.py`) |
| Wikimedia image search | Free CC/PD images, license-filtered, 400 px minimum, excludes diagrams/anatomy (`wikimedia.py`) |
| Claude vision image scoring | Resize to ≤800 px → Haiku vision binary yes/no; discard "no" images (`wikimedia.py`) |
| Pexels clip fallback | Optional (key is optional); proactively downloaded when key supplied (`pexels.py`) |
| Ken Burns effect | Zoom span always 0.5 (1.0→1.5) regardless of clip length; cycles zoom_in/out/pan_right/left (`effects.py`) |
| Crossfade assembly | FFmpeg xfade filter, 1-second fades, all sections connected (`compositor.py`) |
| Colour grading | Warm/cool/neutral preset curves (`compositor.py`) |
| Title card | ASS `TitleCard` style, full-screen centred, t=0→5 s (`subtitler.py`) |
| Section markers | ASS `SectionMark` style at each section boundary, 2.5 s (`subtitler.py`) |
| Key phrase overlays | Claude Haiku extracts 1-2 per section; ASS `KeyPhrase` style (`subtitler.py`) |
| Regular captions | ASS `Arabic` style, bottom of screen, full script coverage (`subtitler.py`) |
| Audio mux | Single FFmpeg pass: video re-encode (crf=22) + AAC (192 kbps) + subtitle burn (`compositor.py`) |
| Dark fallback background | Navy `#1a1a2e` when no images found (`compositor.py`) |
| Thumbnail extraction | Frame at t=5 for Streamlit UI preview (`compositor.py`) |
| Output | 720p MP4, hard duration-capped to audio length |

### Standalone CLI — `phase3_run.py`
A single-file CLI at the repo root that drives the `phase3/` package without
Streamlit.  Zero changes to any `phase3/` file.

```bash
# Inspect section plan (no API calls, no network)
python phase3_run.py --script script.txt --audio-duration 210 --dry-run

# Inspect keywords only — save to JSON for analysis
python phase3_run.py --script script.txt \
  --book-title "مذكرات جعفر العسكري" --character-name "جعفر العسكري" \
  --keywords-only --save-keywords keywords.json

# Full video render
python phase3_run.py --script script.txt --audio audio.mp3 \
  --book-title "..." --character-name "..." \
  --output output/video.mp4 --color-grade warm --thumbnail
```

API keys: `--anthropic-key` / `--pexels-key` flags, or `ANTHROPIC_API_KEY` /
`PEXELS_API_KEY` env vars, or a `.env` file in the working directory.

### Key implementation notes

**Vision scoring** (`wikimedia.py → score_images()`):
- Over-fetches images (2× `images_per_section`) when vision scoring is active
- Each image resized to ≤800 px wide before base64 encoding (oversized → API 400)
- Prompt: "Does this image show [character_name] or a scene directly related to [book_title]? Answer only yes or no."
- **Fail-open**: any API error → keep the image
- Cost: ~$0.001 per image (Haiku vision pricing)

**ASS subtitles** (`subtitler.py`):
- All layers use libass (correct Arabic bidi); **never** FFmpeg `drawtext` (no Arabic shaping)
- Font family in ASS must be `Amiri` (matches `fonts-hosny-amiri` Debian package)
- 4 layers: TitleCard → SectionMark → KeyPhrase → Arabic captions

**Section timing** (`parser.py`):
- Sections: `opening`, `point_1`–`point_5`, `closing`, `cta`
- Duration proportional to character count; minimum 5 s per section

### Current weaknesses

**Image accuracy**:
- Wikimedia text search matches file names/descriptions, not image content
- Vision scoring helps but upstream search quality is the bottleneck
- Next improvement: more specific search queries (transliterations, dates)

**Visual narrative quality**:
- Ken Burns + images + subtitles is functional but not yet cinematic
- No typography fallback cards when all images are rejected by vision scoring

### Phase 3 Roadmap

#### Tier 2 — Next priorities
1. **ElevenLabs TTS** (biggest quality jump):
   - `phase2/tts.py` — fill in `NotImplementedError` stub
   - Add `ELEVENLABS_API_KEY` to Streamlit Cloud secrets
   - Target: Chaouki voice or equivalent high-quality Arabic voice

2. **Pillow typography cards** (for sections with zero images after vision scoring):
   - Gradient background + key phrase text in Amiri via Pillow
   - Use `arabic_reshaper` + `python-bidi` for RTL shaping
   - Replaces navy fallback with something visually informative

3. **Search query quality**:
   - Add Arabic transliteration of character name to Wikimedia queries
   - Add date/century context to historical searches

#### Tier 3 — Future
- Animated word-by-word text reveal
- Custom intro/outro jingle
- Auto-generated book cover placeholder
- Multiple visual themes (documentary, cinematic, minimal)
- AI-generated images for scenes with no stock equivalent

---

## Phase 4 — Workflow Integration ✅

The Streamlit UI chains all phases in one session:

1. **Phase 1a** tab: Upload PDF → Configure mode/OCR/margins/footers/photos → Run → Download ZIPs, footer PDF, photos ZIP
2. **Phase 1b** tab: Run summarisation on Phase 1a output → Download script
3. **Phase 2** tab: Generate audio → Download MP3
4. **Phase 3** tab: Enter book title + character name → Generate video → Download MP4

Session state keys: `phase1a_result`, `phase1a_zip_parts`, `phase1a_footers_pdf`,
`phase1a_footers_zip`, `phase1a_photos_zip`, `phase1b_result`, `phase3_video_path`.

Phase 4 is complete once Phase 3 produces broadcast-quality output.

---

## Immediate Next Steps (start here next session)

1. **End-to-end validation of Phase 1a on Streamlit Cloud**:
   - Run with Al-Askari Memoirs; confirm Kraken OCR completes without crash
   - Verify footer PDF and page images ZIP download correctly
   - Branch `claude/upgrade-phase1a-ocr-3usw0` must be deployed

2. **Investigate Phase 3 visual quality using `phase3_run.py`**:
   - Run `--dry-run` to check section parsing on a real script
   - Run `--keywords-only --save-keywords kw.json` to audit keyword quality
   - Run a full render and review the MP4 for image relevance, subtitle timing
   - Identify which Tier 2 improvement has the most impact

3. **Implement ElevenLabs TTS** (Tier 2, highest quality jump):
   - File: `phase2/tts.py` — fill in `NotImplementedError` stub
   - Add `ELEVENLABS_API_KEY` to Streamlit Cloud secrets

4. **Implement Pillow typography cards** (Tier 2):
   - For sections where vision scoring leaves zero images
   - `arabic_reshaper` + `python-bidi` + Pillow gradient + Amiri font

---

## Key Technical Constraints

| Constraint | Detail |
|-----------|--------|
| Streamlit Cloud RAM | 1 GB — keep FFmpeg work in subprocesses; no large in-memory buffers |
| No GPU | All ML inference via API; local tools CPU-only |
| Python version | **3.12.13** (set once in Advanced settings at first deploy) |
| Kraken / torch | Active on Python 3.12; `lightning-compat/` shim required (PyPI quarantined) |
| Arabic RTL in video | Use ASS + libass; **never** FFmpeg `drawtext` (no Arabic bidi) |
| Claude vision image size | **Always resize to ≤ 800 px wide** before sending — oversized → `400 Could not process image` |
| Arabic font for FFmpeg | `fonts-hosny-amiri` (Debian trixie) → font family name `Amiri` in ASS |
| Do NOT use | `fonts-noto-arabic` — does not exist in Debian trixie repos |
| Pexels key | Optional — app must work without it |
| Anthropic API key | Required for Phase 1b summarisation and Phase 3 keywords/vision scoring |

---

## Model & Cost Strategy

| Task | Model | Cost per book |
|------|-------|---------------|
| Reader per chunk | `claude-haiku-4-5-20251001` | ~$0.01 |
| Consolidator | `claude-haiku-4-5-20251001` | ~$0.001 |
| Scriptwriter | `claude-sonnet-4-6` | ~$0.04 |
| Editor/Scorer | `claude-haiku-4-5-20251001` | ~$0.002 |
| Keyword + key phrase gen (Phase 3) | `claude-haiku-4-5-20251001` | ~$0.003 |
| Image relevance vision scoring | `claude-haiku-4-5-20251001` vision | ~$0.005 |
| **Total (current, gTTS)** | | **~$0.06** |
| TTS (gTTS) | Free | $0 |
| TTS (ElevenLabs target) | Chaouki voice | ~$0.10–0.30 |

**Rule**: Haiku for every bulk, scoring, or classification task. Sonnet/Opus only for creative output (the final script).

---

## Development Conventions

### Arabic text handling
- Never apply diacritization to raw OCR output — only to the final approved script
- Always use `arabic_reshaper` + `python-bidi` when rendering Arabic in Pillow
- In ASS subtitles, `ScaledBorderAndShadow: yes` and `WrapStyle: 0` for correct RTL wrapping
- Use MSA Arabic only in generated scripts; reject dialect substitutions

### FFmpeg subprocess calls
- All FFmpeg calls via `subprocess.run([...], check=True)` — never `os.system()`
- Build filter graphs as Python list → `','.join(filters)` to avoid shell injection
- `probe_duration()` in `effects.py` uses `ffprobe -v quiet -print_format json -show_format`
- Ken Burns via `zoompan`; scale image to 2× output resolution first to avoid upscaling artefacts

### Streamlit patterns
- Phase outputs in `st.session_state` keyed by phase: `phase1a_result`, `phase3_video_path`, etc.
- Progress callbacks: `on_progress(message: str, fraction: float)` passed into pipeline functions
- All file paths in session state are absolute paths

### Git workflow
- Active development branch: `claude/upgrade-phase1a-ocr-3usw0`
- Commit message format: `Phase N: <what changed>`
- Push to `origin claude/upgrade-phase1a-ocr-3usw0` after each logical unit of work

### Secrets / environment
- `ANTHROPIC_API_KEY` — required for Phases 1b and 3
- `PEXELS_API_KEY` — optional
- `ELEVENLABS_API_KEY` — stub ready in `phase2/tts.py`
- On Cloud: `st.secrets["KEY_NAME"]`; locally: `.env` file (never commit)
