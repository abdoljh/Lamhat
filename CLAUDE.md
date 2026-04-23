# Bk2Video — Master Plan & Session Handoff

## Project Vision

Convert Arabic books (PDF) into high-impact, 3-to-5-minute video summaries —
long enough to deliver real value, short enough for modern attention spans.
The output is a fully automated MP4: Arabic TTS voice, relevant background
visuals with motion, and burned-in Arabic subtitles. Every phase is built to
run on Streamlit Community Cloud (1 GB RAM, no GPU).

Working repo: **abdoljh/Bk2Video** · Streamlit Community Cloud deployment.

---

## Four-Phase Architecture

| Phase | Name | Goal | Status |
|-------|------|------|--------|
| 1 | Text Extraction & Summarisation | PDF → cleaned Arabic text → 625–850-word video script | ✅ **Complete** |
| 2 | Audio Synthesis (TTS) | Script → Arabic MP3 via gTTS (ElevenLabs next) | ✅ **Working** (gTTS) |
| 3 | Visual Generation | Script + audio → final MP4 with visuals, voice, subtitles | 🔧 **In Progress** |
| 4 | Workflow Integration | One-click pipeline: PDF → finished video | ✅ **Complete** (follows Phase 3) |

---

## Repo Structure

```
streamlit_app.py          # Streamlit entrypoint (Phases 1–3 UI, ~824 lines)
phase1/
  __init__.py             # Exports Phase1Pipeline, Phase1Config, Phase1Result
  pipeline.py             # Phase1Pipeline orchestrator (7-step process)
  core/
    ingestor.py           # PDF ingestion (PyMuPDF) — digital + scanned, RTL
    ocr_engine.py         # Tesseract / EasyOCR / PaddleOCR wrapper
    normalizer.py         # Arabic text normalisation (lam-alef, Farsi Yeh, noise)
    chunker.py            # Semantic chunking (~180 lines)
    diacritizer.py        # Mishkal / Farasa wrapper
    summarizer.py         # Hierarchical summarisation + script generation
    output_writer.py      # JSON + TXT serialisation
phase2/
  __init__.py
  tts.py                  # gTTS backend; ElevenLabs stub (implement next)
phase3/
  __init__.py             # generate_background_video() full pipeline (~260 lines)
  parser.py               # Script section splitter + duration estimator (~105 lines)
  keywords.py             # Claude Haiku: search terms + key phrases per section (~181 lines)
  wikimedia.py            # Wikimedia Commons image fetcher + Claude vision scoring (~306 lines)
  pexels.py               # Pexels video clip fetcher (~106 lines)
  effects.py              # Ken Burns (zoompan) + trim + probe_duration (~163 lines)
  compositor.py           # Section clips → crossfade → grade → mux (~396 lines)
  subtitler.py            # Multi-layer ASS subtitle generator (~276 lines)
packages.txt              # Streamlit Cloud apt deps (ffmpeg, fonts-hosny-amiri, etc.)
requirements.txt          # Python deps
output/                   # Phase 1 outputs (JSON + TXT); gitignored in production
samples/                  # Test PDFs (Al-Askari, preface, sample docs)
Book1/                    # Al-Askari Memoirs test data
Audio2/                   # Phase 2 TTS output samples
Video3/                   # Phase 3 video output samples
PHASE1_PLAN.md            # Phase 1 detailed design document
```

---

## Phase 1 — Text Extraction & Summarisation ✅

### What it does
1. Ingests PDF (digital or scanned Arabic)
2. OCR via Tesseract (fits Streamlit Cloud 1 GB RAM limit)
3. Normalises Arabic text (lam-alef fixes, Farsi Yeh, noise removal, header/footnote stripping)
4. Semantic chunking (default 1500 tokens, 200 overlap)
5. Hierarchical summarisation: Reader (Haiku, per chunk) → Consolidator (Haiku) → Scriptwriter (Sonnet) → Editor/Scorer (Haiku, up to 2 retries)
6. Outputs: `*_phase1.json`, `*_phase1.txt`, `*_phase1_raw.txt`, `book_script.txt`, `book_script_diacritized.txt`, `book_script_metadata.json`

### Key design decisions
- **Cost strategy**: Haiku for all bulk/scoring work; Sonnet only for the final script (~$0.05 for a 300-page book).
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
  - Stub already exists in `phase2/tts.py` (`NotImplementedError`).
  - Needs: `ELEVENLABS_API_KEY` secret + voice ID in UI, then call ElevenLabs REST API.
  - Priority: implement after Phase 3 visual quality is stable.

---

## Phase 3 — Visual Generation 🔧 (IN PROGRESS — THE CORNERSTONE)

### What works today (Tier 1 — all implemented ✅)

| Feature | Implementation |
|---------|---------------|
| Section parsing | Regex-based Arabic section detection (`parser.py`) |
| Keyword + key phrase generation | Claude Haiku per section; book title + character name as context (`keywords.py`) |
| Wikimedia image search | Free CC/PD images, license-filtered, 400 px minimum, excludes diagrams/anatomy (`wikimedia.py`) |
| **Claude vision image scoring** | Resize to ≤800 px → Haiku vision binary yes/no; discard "no" images (`wikimedia.py`) |
| Pexels clip fallback | Optional (key is optional); proactively downloaded when key supplied (`pexels.py`) |
| Ken Burns effect | Zoom span always 0.5 (1.0→1.5) regardless of clip length; cycles zoom_in/out/pan_right/left (`effects.py`) |
| Crossfade assembly | FFmpeg xfade filter, 1-second fades, all sections connected (`compositor.py`) |
| Colour grading | Warm/cool/neutral preset curves (`compositor.py`) |
| Title card | ASS `TitleCard` style, full-screen centred, t=0→5 s (`subtitler.py`) |
| Section markers | ASS `SectionMark` style at each section boundary, 2.5 s (`subtitler.py`) |
| Key phrase overlays | Claude Haiku extracts 1-2 per section; ASS `KeyPhrase` style, centred (`subtitler.py`) |
| Regular captions | ASS `Arabic` style, bottom of screen, full script coverage (`subtitler.py`) |
| Audio mux | Single FFmpeg pass: video re-encode (crf=22) + AAC (192 kbps) + subtitle burn (`compositor.py`) |
| Dark fallback background | Navy `#1a1a2e` when no images found (cinematic, not black) (`compositor.py`) |
| Thumbnail extraction | Frame at t=5 for Streamlit UI preview (`compositor.py`) |
| Output | 720p MP4, hard duration-capped to audio length |

### Key implementation notes

**Vision scoring** (`wikimedia.py → score_images()`):
- Called after `download_images()`, before assembly
- Over-fetches images (2× `images_per_section`) when vision scoring is active
- Each image resized to `img.thumbnail((800, 800))` before base64 encoding
- Prompt: "Does this image show [character_name] or a scene directly related to [book_title]? Answer only yes or no."
- **Fail-open**: any API error → keep the image (prefer something over empty screen)
- Cost: ~$0.001 per image (Haiku vision pricing)

**ASS subtitles** (`subtitler.py`):
- All layers use libass (correct Arabic bidi); **never** use FFmpeg `drawtext` (no Arabic shaping)
- Font family in ASS must be `Amiri` (matches `fonts-hosny-amiri` Debian package)
- 4 concurrent tracks: TitleCard → SectionMark → KeyPhrase → Arabic captions

**Section timing** (`parser.py`):
- Sections: `opening`, `point_1`, `point_2`, `point_3`, `closing`, `cta`
- Duration estimated from word count × speaking rate (Arabic ~130 words/min)
- `cta` section always gets minimum 15 s to allow book presentation to land

### Current weaknesses (what still needs work)

**Image accuracy** (partially mitigated by vision scoring):
- Wikimedia text search matches file names/descriptions, not image content
- Vision scoring filters irrelevant images but upstream search quality is still the bottleneck
- Next improvement: better search query construction (more specific Arabic transliterations)

**Visual narrative quality**:
- Ken Burns + images + subtitles is functional but not yet cinematic
- No typography fallback cards when images are absent (still shows navy background)

### Phase 3 Tiered Roadmap

#### Tier 1 — Complete ✅
All features in the table above, including vision scoring (merged in commit `de5e3fd`).

#### Tier 2 — Next session priorities
1. **ElevenLabs TTS** (biggest single quality jump):
   - File: `phase2/tts.py` — fill in `NotImplementedError` stub
   - Add `ELEVENLABS_API_KEY` to Streamlit Cloud secrets
   - Target: Chaouki voice or equivalent high-quality Arabic voice
   - The voice quality transforms the perceived quality of the entire video

2. **Pillow text cards** (for sections with no usable images after vision scoring):
   - Render a styled typography card: gradient background + key phrase in Amiri
   - Use `arabic_reshaper` + `python-bidi` for correct RTL shaping in Pillow
   - Replaces navy fallback with something visually informative

3. **Search query quality** (improves vision scoring hit rate):
   - Add Arabic transliteration of character name to Wikimedia queries
   - Add date/century context to historical searches

#### Tier 3 — Future
- Animated word-by-word text reveal
- Custom intro/outro jingle
- Auto-generated book cover placeholder
- Multiple visual themes (documentary, cinematic, minimal)
- AI-generated images (DALL-E / Stable Diffusion) for scenes with no stock equivalent

---

## Phase 4 — Workflow Integration ✅

The Streamlit UI chains all phases in one session:

1. **Phase 1** tab: Upload PDF → Configure (OCR backend, chunking params, API keys) → Run → Download outputs
2. **Phase 2** tab: Choose script source (Phase 1 session or upload `.txt`) → Choose variant (plain/diacritized) → Generate audio → Download MP3
3. **Phase 3** tab: Enter book title + character name → (optional) keyword preview → Generate Final Video → Download MP4

Each phase's output automatically feeds the next within the same session via Streamlit `st.session_state`. Progress callbacks stream to the UI in real-time.

Phase 4 is considered complete once Phase 3 produces broadcast-quality output.

---

## Immediate Next Steps (start here next session)

1. **Verify vision scoring on Streamlit Cloud** — confirm `de5e3fd` is live:
   - Check that `fonts-hosny-amiri` installs successfully (packages.txt)
   - Run end-to-end with al-Askari Memoirs and inspect downloaded video
   - Confirm TitleCard, SectionMark, KeyPhrase, and Arabic caption layers all render

2. **Implement ElevenLabs TTS** (Tier 2, highest quality impact):
   - File to edit: `phase2/tts.py`
   - Fill in the `NotImplementedError` stub
   - Add `ELEVENLABS_API_KEY` to Streamlit Cloud secrets
   - Test with Chaouki voice or equivalent Arabic voice

3. **Implement Pillow typography cards** (Tier 2):
   - For sections where vision scoring leaves zero images
   - Use `arabic_reshaper` + `python-bidi` for RTL text rendering in Pillow
   - Gradient background + Amiri font + key phrase text

4. **End-to-end validation** on al-Askari Memoirs (both PDF parts) and upload sample MP4 to `Video4/` in the repo.

---

## Key Technical Constraints

| Constraint | Detail |
|-----------|--------|
| Streamlit Cloud RAM | 1 GB — no PyTorch; Tesseract OCR only |
| No GPU | All ML inference via API; local tools CPU-only |
| Arabic RTL in video | Use ASS + libass (correct bidi); **never** FFmpeg `drawtext` (no Arabic bidi support) |
| Claude vision image size | **Always resize to ≤ 800 px wide** before sending — oversized → `400 Could not process image` |
| Streamlit Cloud Python | 3.14 — PaddleOCR needs ≤ 3.12; do not use PaddleOCR |
| Arabic font for FFmpeg | `fonts-hosny-amiri` (Debian trixie) → font family name `Amiri` in ASS files |
| Do NOT use | `fonts-noto-arabic` — does not exist in Debian trixie repos |
| Pexels key | Optional — app must work without it; wrap all Pexels code in key-presence checks |
| Anthropic API key | Required for Phase 1 summarisation and Phase 3 keyword/vision scoring |

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

**Rule**: Use Haiku for every bulk, scoring, or classification task. Reserve Sonnet/Opus only for creative output (the final script).

---

## Development Conventions

### Arabic text handling
- Never apply diacritization to raw OCR output — only to the final approved script
- Always use `arabic_reshaper` + `python-bidi` when rendering Arabic in Pillow/matplotlib
- In ASS subtitles, set `ScaledBorderAndShadow: yes` and `WrapStyle: 0` for correct RTL wrapping
- Use MSA Arabic only in generated scripts; reject dialect substitutions

### FFmpeg subprocess calls
- All FFmpeg calls go through `subprocess.run([...], check=True)` — never `os.system()`
- Build filter graphs as Python list → `','.join(filters)` to avoid shell injection
- `probe_duration()` in `effects.py` uses `ffprobe -v quiet -print_format json -show_streams`
- Ken Burns via `zoompan` filter; scale image to 2× output resolution first to avoid upscaling artifacts

### Streamlit patterns
- Phase outputs stored in `st.session_state` keyed by phase number: `st.session_state['phase1_result']`, `st.session_state['phase3_video_path']`, etc.
- Progress callbacks: pass a `progress_callback(message: str)` function from UI into pipeline functions
- All file paths in session state are absolute paths to temp files; clean up on session end

### Git workflow
- Active development branch: `claude/add-claude-documentation-08SqJ`
- Commit message format: `Phase N: <what changed>` (e.g., `Phase 3: Claude vision scoring — discard irrelevant Wikimedia images`)
- Push to `origin claude/add-claude-documentation-08SqJ` after each logical unit of work

### Secrets / environment
- `ANTHROPIC_API_KEY` — required for Phases 1 and 3
- `PEXELS_API_KEY` — optional; app works without it (Wikimedia + vision scoring only)
- `ELEVENLABS_API_KEY` — not yet used; stub ready in `phase2/tts.py`
- Access in code: `st.secrets["KEY_NAME"]` on Cloud; `.env` file locally (never commit)
