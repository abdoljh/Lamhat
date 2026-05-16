# Phase 3 — Visual Generation · Session Handoff

> **Working tree**: `Lamahat/_Phase3/`
> **Pipeline status**: end-to-end working on Colab CPU. Latest render: 43-shot
> plan, 391 s, 26.6 MB MP4, ~21 min wall time.
> **The MOV in `output/final_cut_3a.mov` (12.8 MB / 181 s) is a partial preview
> uploaded to fit GitHub's file-size limits** — not the real render. The real
> output is `output_files.zip → final_cut.mp4` saved to Drive by
> `_phase3_b2c.ipynb`. Audio (391 s) and plan (43 shots, 0.00 → 391.00 s) match
> end-to-end; there is no truncation bug.

---

## 1. What Phase 3 Now Is (vs. the v1 you may remember)

The original Phase 3 was section-based: parse 4 Arabic sections, pull 2-3
Wikimedia images per section, Ken-Burns them, crossfade, mux. That code still
lives at `phase3/__init__.py → generate_background_video()` and remains
reachable from Streamlit. It has been **superseded** by a *shot-based* pipeline
that is now the default end-to-end path.

**v2 architecture (current)**: a *shot plan* is the source of truth. The plan
is a list of 30–65 timestamped `Shot` dataclasses produced by one Claude
Sonnet 4.6 call; the renderer executes the plan without making any creative
choices. Plans are JSON, inspectable, diff-able, regeneratable from cache.

```
Script + Audio ──► align.py ──► word timings (WhisperX | Whisper | interp)
                                       │
                                       ▼
                                   plan.py  ──► shot_plan.json (one Sonnet call)
                                       │
                                       ▼
                                  render.py  ──► MP4
                                       ▲
                                       │
                              sources/Fetcher  (LoC → Wikimedia → IA → Pexels,
                                                 +cache +user-upload +book-extract,
                                                 Haiku-vision-scored)
```

### Two CLIs

| CLI | Purpose | Stops at |
|-----|---------|----------|
| `phase3_run.py` | Plan **and/or** render in one go. Owns the v1 path too. | configurable: `--dry-run`, `--keywords-only`, `--align-only`, `--plan-only`, or full render |
| `render_plan.py` | Render a previously-saved plan to MP4. | always produces an MP4 (or writes a manifest with `--build-manifest`) |

Splitting plan vs. render is deliberate: the planner costs a Sonnet call
(~$0.10 + ~90 s wall) and the renderer costs CPU minutes (~20 min). When
iterating on visuals you re-render; when iterating on shot choices you
re-plan. Keep them separable.

### File map

```
_Phase3/
├── phase3_run.py             # Plan-OR-render CLI (--dry-run / --align-only / --plan-only / full)
├── render_plan.py            # Render-only CLI (consumes plan JSON)
├── audit_plan.py             # Quality audit of a saved plan (used in cell 10)
├── _phase3_b2c.ipynb         # Colab driver — the canonical run lives here
├── samples/al_askari_script.txt
├── output/                   # render.log (51-shot prior run), audio, MOV preview
└── phase3/
    ├── __init__.py           # v1 entrypoint generate_background_video() — legacy
    ├── parser.py             # Arabic section regexes + estimate_durations
    ├── align.py              # WhisperX | Whisper | interpolation → WordTiming list
    ├── plan.py               # Sonnet 4.6 shot planner + Shot dataclass + JSON I/O
    ├── render.py             # Plan → MP4 (assets, motion, captions, mux)
    ├── typography.py         # Pillow Family A typography cards (5 templates, 864 LOC)
    ├── compositor.py         # v1 background video assembler (still used by v1 path)
    ├── effects.py            # ffprobe wrapper + Ken Burns helpers (v1)
    ├── keywords.py           # v1 keyword generator (Haiku per section)
    ├── pexels.py             # v1 video-clip fetcher
    ├── subtitler.py          # v1 ASS subtitle writer
    ├── wikimedia.py          # v1 image fetcher + vision scorer
    ├── render_previews.py    # Dev helper — render a typography template grid
    ├── test_smoke.py / test_typography.py / test_resilient.py
    └── sources/              # v2 image-fetch waterfall (new)
        ├── __init__.py       # Fetcher orchestrator + FetcherConfig
        ├── base.py           # Source ABC, ImageCandidate, FetchResult, free-license detection
        ├── loc.py            # Library of Congress JSON API
        ├── wikimedia.py      # MediaWiki API; 400 px min dimension filter
        ├── internet_archive.py  # archive.org advancedsearch
        ├── pexels.py         # Pexels v1 photos endpoint (api_key required)
        ├── user_upload.py    # shot_NN.jpg overrides from a user directory
        ├── book_extract.py   # Phase 1a photo bank (vision-scored against query)
        ├── cache.py          # disk cache keyed by query
        └── vision.py         # Claude Haiku vision scorer, three-axis rubric (0-3)
```

---

## 2. The Shot Data Model

```python
@dataclass
class Shot:
    start: float                                # seconds from t=0
    end: float
    visual: ShotVisual                          # see taxonomy below
    search_query: str = ""                      # English; "" for typography
    source_hint: str = "auto"                   # "wikimedia" | "loc" | "pexels" | "auto"
    motion: ShotMotion = "slow_push"
    motion_intensity: float = 1.0
    typography_template: TypographyTemplate | None = None
    typography_text: str = ""                   # Arabic, verbatim from script
    caption_text: str = ""                      # auto-filled from word_timings
    show_caption: bool = True
    note: str = ""                              # planner's free-form rationale
    section_id: str = ""                        # auto-assigned by midpoint
```

**Visual taxonomy** (8 kinds): `portrait`, `location`, `object`, `archive`,
`broll`, `typography`, `title_card`, `section_mark`.

**Motion taxonomy** (7 kinds): `static_hold`, `slow_push`, `fast_push`,
`slow_pull`, `pan_left`, `pan_right`, `ken_burns`. In `render.py` static motion
is applied to typography and placeholder cards; the listed motions only fire
for fetched real images (see `_MOTION_FILTERS`).

**Typography templates** (5): `pull_quote`, `name_reveal`, `date_stamp`,
`chapter_heading`, plus implicit `title_card` / `section_mark` styles. All
rendered by `typography.py` (Family A: cream/charcoal, Amiri).

### Plan invariants (enforced by `plan._validate_plan`)

1. Shots are sorted by `start` and **contiguous**: `shot[i].end == shot[i+1].start`.
2. First shot starts at `0.0`; last shot ends at `total_duration_sec`.
3. Per-visual hard caps; beyond them `_validate_plan` splits a shot into
   ~5 s pieces and tags each `[auto-split k/n]`:
   `typography 12 s · portrait 12 s · archive/broll/location/object 10 s · section_mark/title_card 7 s`.
4. Adjacent shots with identical `(visual, search_query)` or
   `(visual, typography_text)` are merged (`_shots_can_merge`) and their
   `caption_text` concatenated.
5. Field exclusivity: typography-kind shots (`title_card`, `section_mark`,
   `typography`) keep `typography_text`, drop `search_query`. Image-kind
   shots do the opposite (`_normalise_fields`).
6. Shot boundaries snap to actual word boundaries (`_snap_to_word_boundaries`).
   Minimum shot duration after snap: 1.5 s.

These invariants are why the renderer can be dumb — by the time it sees a
plan, the math is consistent.

---

## 3. Sources Subsystem (the new image-fetch waterfall)

`sources/Fetcher.fetch_for_shot(query, shot_index)` runs this priority order:

1. **User upload** — `--user-dir <path>`. File matched by name pattern
   `shot_NN.jpg` (NN = 1-indexed shot number) or by `manifest.json`.
2. **Book extract** — `--book-extracts <Phase1a photos.zip or dir>`.
   Vision-scored against the shot query (requires `--anthropic-key`).
3. **Disk cache** — `~/.cache/lamahat/images` keyed by query. Disable with
   `--no-cache`.
4. **Live web fetch** in this order: LoC → Wikimedia → IA → Pexels. All
   candidates from all sources are pooled, downloaded, then vision-scored,
   then ranked by `vision.rank_candidates`. Top survivor wins.

`VisionScorer` (Haiku, `model="claude-haiku-4-5-20251001"`) emits three integer
scores per image (`subject`/`quality`/`cinematic`, 0–3 each). Threshold to
keep: `total ≥ 4 AND subject ≥ 1`. The Haiku call is **fail-open** — on any
exception the candidate is assigned `(2, 2, 1) = 5` and kept.

### License posture

`base.is_free_license()` accepts everything CC-*, PD, "no known restrictions",
plus unknown. Rejects anything with `NC`, `ND`, "all rights reserved". Pexels
is hard-coded as `"Pexels License"` (permissive but with attribution
conventions; double-check before public release).

---

## 4. Rendering Pipeline (one MP4 from one plan)

`render.render_video(shots, out_path, *, audio_path, audio_duration_sec, config, on_progress)`:

1. For each shot:
   - Build a 1920×1080 PNG asset:
     - Typography visuals → `typography.render()` (Family A card)
     - Image visuals → `Fetcher.fetch_for_shot()` → copy chosen JPEG to PNG
     - Fallback → `_placeholder_card()` (cream card with the search query)
     - Final fallback on exception → `_error_card()` (so the timeline doesn't
       collapse — audio sync depends on every shot producing a clip of its
       planned duration)
   - Encode the PNG to an MP4 clip of the shot's exact duration. Motion only
     applies if `is_real_image=True`; typography and placeholders always
     `static_hold`. Zoom is computed against a 1.6× buffer to avoid blurry
     pan-edges.
2. **Stream-copy concat** of all shot clips → `background.mp4`. Works only
   because every clip uses identical encoder settings (`libx264 -preset
   ultrafast -crf 22 -pix_fmt yuv420p -r 25`). Change one shot's profile and
   the concat silently breaks.
3. **ASS captions** (`_write_captions`): white Amiri text with charcoal
   outline (BorderStyle 1 — backplate doesn't work in libass because it
   ignores alpha for BorderStyle 3). Typography shots are excluded
   (`s.visual not in TYPOGRAPHY_VISUALS`). 0.05 s pre-roll on each caption.
4. **Final mux** (`_mux_final`): single FFmpeg pass that re-encodes the video
   (required to burn subs), adds AAC audio at 192 kbps with `-shortest`,
   then `-t max_duration` if set. The re-encode pass is ~5 minutes of the
   ~21-minute total.

Everything FFmpeg is shelled via `subprocess.run`, working under
`tempfile.TemporaryDirectory` so RAM stays low — important for Streamlit
Cloud's 1 GB ceiling.

---

## 5. The Canonical Run, Decoded

The notebook `_phase3_b2c.ipynb` is the authoritative reference for what
works today. The cell-by-cell pipeline and what each cell tells us:

| Cell | What it does | What its output proves |
|------|--------------|------------------------|
| 0 | Mount Drive, copy `_Phase3/` into `/content` | Colab working dir is `/content`, not `_Phase3/` — paths in CLIs are relative |
| 1 | `pip install anthropic` (0.102.0) | Sonnet + Haiku reachable |
| 2 | WhisperX/Whisper install (commented out) | Alignment uses **interpolated** backend — see §7.5 |
| 3 | `apt install fonts-hosny-amiri` (0.113-1) | Amiri available system-wide via fontconfig |
| 4 | `pip install arabic-reshaper python-bidi` | Fallback path for non-libraqm Pillow builds |
| 5–6 | Matplotlib + `phase3.typography.FONT_PATHS` sanity check | Reveals an Amiri-discovery bug — see §7.1 |
| 7 | Load API keys from Colab Secrets | Both set |
| 8 | `phase3_run.py --align-only --align-backend interpolated` | 653 word tokens, **only 2 sections parsed** — see §7.2 |
| 9 | `phase3_run.py --plan-only` | Sonnet returns 43 shots covering 0.00–391.00 s in 91.4 s; one Sonnet call (~$0.10) |
| 10 | `audit_plan.py` | 0 gaps/overlaps; 35% typography (in target); 14% auto-split (acceptable); 22 search queries averaging 7.5 words; **no bare queries (good)** |
| 11–12 | `render_plan.py` background + tail-monitor | "Done in 1263 s — output/final_cut.mp4 (26.6 MB)" |
| 13 | Zip outputs (excluding the .mp3) | `output_files.zip` with `final_cut.mp4 + render.log + plan.json + word_timings.json + planner_raw_response.txt` |
| 14 | Copy zip to Drive | Final deliverable: `/content/drive/MyDrive/_Phase3/output_files.zip` |

### Audit findings (cell 10, verbatim)

```
Total shots:        43
Plan timeline:      0.00s → 391.00s (391.0s)
Average shot:       9.09s
Range:              4.49s – 12.17s
✓  No gaps or overlaps

Visual types:
   typography      15 (  35%) ██████████   ← within target 25-35%
   archive          8 (  19%)
   portrait         7 (  16%)
   broll            4 (   9%)
   section_mark     4 (   9%)
   location         3 (   7%)
   title_card       2 (   5%)               ← open + close, correct

Motion types:
   static_hold     28 (  65%)
   slow_push       13 (  30%)
   pan_right        2 (   5%)

Section coverage:
   opening         33 shots
   closing         10 shots                 ⚠ See §7.2

✓  Auto-split shots: 6/43 (14%) from 6 original(s)
Typography texts: 21 unique (avg 11.4 words)
Search queries: 22 non-empty, avg 7.5 words   ✓ none bare
```

The plan is healthy on every dimension except *section structure*. The
auto-split rate is 14 %, well below the 20 % "tighten the prompt" line.
Typography density at 35 % sits right on the prompt's target ceiling.

---

## 6. What Actually Worked vs. What's a Compromise

| Subsystem | State | Notes |
|-----------|-------|-------|
| Parser → 8 sections | **Compromised** | Only 2 of 7 expected section_ids matched. Header regexes assume rigid template (§7.2) |
| Alignment | **Compromised** | Interpolated backend only. ~12 chars/sec heuristic, no real audio-to-text alignment. WhisperX install commented out in the notebook |
| Planner (Sonnet) | ✅ Working | 91 s, ~$0.10/call, 43 well-formed shots, JSON parses cleanly |
| Plan validation | ✅ Working | Audit passes all structural checks |
| Typography rendering | ✅ Working | Amiri loaded (eventually — see §7.1) |
| Source: Pexels | ✅ Working | 3 candidates per query, every shot |
| Source: LoC / Wikimedia / IA | **Broken in practice** | 0 candidates for every single query in both observed runs (§7.3) |
| Vision scoring | **Broken mid-run** | First call succeeded, rest hit `credit_balance_too_low`. Fail-open policy then bricks ranking (§7.4) |
| Renderer | ✅ Working | 43-shot run: 1263 s, 26.6 MB, no errors, exactly 391 s output |
| Captions (ASS) | ✅ Working | Burned in cleanly; typography shots correctly skipped |
| Mux | ✅ Working | AAC 192 kbps; `-shortest` + `-t` cap |

---

## 7. Open Issues, In Priority Order

### Tier 1 — these distort *every* output

#### 7.1 Amiri discovery falls through despite a system install

Cell 6 reports `Amiri not found on system — downloading from upstream`,
even though cell 3 successfully installed `fonts-hosny-amiri` to
`/usr/share/fonts/opentype/fonts-hosny-amiri/`. Then cell 6 succeeds via
the fallback download to `~/.cache/lamahat/fonts/amiri-1.003/`. So the
output isn't broken — but on Streamlit Cloud (ephemeral containers, no
persistent `~/.cache`) every cold start pays a 591 KB (Debian) → 6 MB
(upstream zip) overhead because `_discover_amiri_fonts()` isn't finding
the system files.

Likely cause in `typography.py:_discover_amiri_fonts`:
- `fc-match` would return the right file on Colab (`fontconfig` is
  configured), but the code only accepts the match when `"amiri" in
  amiri_regular_path.name.lower()`. The Debian filename is exactly
  `Amiri-Regular.ttf` so this *should* hit; the fall-through to download
  suggests `fc-match -f "%{file}" "Amiri:style=Regular"` may have returned
  a non-Amiri fallback (e.g., DejaVu) because Pillow ran before
  `fc-cache` had refreshed.

Fixes:
- Run `fc-cache -f` once at module import on Linux, before the
  `fc-match` probe.
- Probe `Path("/usr/share/fonts/opentype/fonts-hosny-amiri").rglob("Amiri-Regular.ttf")`
  *before* `fc-match`, since cell 3 demonstrates that path always exists
  after `apt install fonts-hosny-amiri`.
- Cache the discovery result in a module global so successive imports in
  the same process don't re-scan.

#### 7.2 Section parser only recognises rigid template headers

The notebook's alignment cell reports `653 word tokens, 2 sections`. The
real script has 5 logical sections — opening + 3 descriptive points +
closing — but `parser._SECTION_HEADERS` only matches the rigid v1
template (`النقطة الأولى/الثانية/...` and `الخاتمة` and `تقديم الكتاب`).
The current Phase 1b summariser emits descriptive titles instead:

```
Line  9: من الموصل إلى الاستانة — رحلة التحديث والطموحْ
Line 17: الصراع الأيديولوجي والسياسي — بين الولاء والحلمْ
Line 25: الحرب والاختبار النهائي — الفعل والالتزامْ
Line 33: الخاتمة: شهادة لا تموتْ                    ← only this matches
```

Result: `opening = lines 1–32` (one monolithic 287-second blob) and
`closing = lines 33–43`. The planner sees a single huge section and
distributes shots evenly within it — visible as `opening 33 shots,
closing 10 shots` in the audit. Section markers (`section_mark` visual)
still appear because Sonnet inserts them on tonal breaks, but the
*intended* structural mapping (one set of visuals per thematic point) is
lost.

Fix options, ranked by leverage:
1. **Loosen the parser**: detect *any* line that ends with `.` or `ْ` and
   sits between blank lines as a candidate header. Cross-check with line
   length (headers are typically < 80 chars). Promote those to
   sections with auto-generated IDs `point_1`, `point_2`, ….
2. **Synchronise with Phase 1b**: either re-introduce the rigid template
   in the scriptwriter prompt, or have Phase 1b emit a sidecar JSON
   that explicitly lists the section boundaries (line numbers).
   Sidecar is cleaner — keeps the script copy-pasteable for the user.
3. As a stop-gap, set `parser._SECTION_HEADERS` to a single broad
   pattern matching "any short line followed by a blank line".

#### 7.3 LoC / Wikimedia / Internet Archive return 0 candidates per query

The single biggest visual-quality issue. In the latest 43-shot plan,
*every* image shot's query went through this waterfall:

```
LoC:               0 candidates for 'Jafar al-Askari Iraqi general historical portrait'
Wikimedia:         0 candidates for 'Jafar al-Askari Iraqi general historical portrait'
Internet Archive:  0 candidates for 'Jafar al-Askari Iraqi general historical portrait'
Pexels:            3 candidates for 'Jafar al-Askari Iraqi general historical portrait'
```

But Wikimedia Commons demonstrably has `Category:Mahmud_Shevket_Pasha`
with PD photographs, postcards, and assassination-scene imagery; LoC has
1880-1940 MENA holdings; IA has period books. The problem is **query
construction**, not content availability. Probable causes in descending
order:

1. **Over-specific multi-word queries**. MediaWiki's `gsrsearch` is
   phrase-AND. Six tokens — `'Jafar al-Askari Iraqi general historical
   portrait'` — require all six in file metadata, which is rare. Same
   for LoC and IA. **Fix**: add a `query_simplify()` helper that strips
   generic tails (`portrait historical photograph archive picture`),
   keeps proper nouns + dates. Try the simplified form first; full form
   as a fallback.
2. **400 px minimum dimension filter (Wikimedia)**.
   `wikimedia.py:_MIN_DIMENSION = 400` rejects every result whose
   `thumbwidth/thumbheight` (or source w/h) is below 400. Many period
   photographs in Commons are stored as small JPEGs (300–380 px on the
   short edge) and get rejected even when they're exactly right. **Fix**:
   drop to 320 or remove the filter entirely when `thumburl` is present
   — the thumb is always rescalable.
3. **`-diagram -anatomy -chart -schematic` exclusion**. Passed raw to
   `gsrsearch`. Combined with already-narrow queries it removes
   borderline matches. **Fix**: make it opt-in.
4. **LoC's facet filter**. `'fa': 'online-format:image|original-format:photo,print'`
   sometimes returns 0 even when the same free-text query in LoC's web
   UI returns thousands. **Fix**: try without `fa` and post-filter in
   code.
5. **IA's image-only filter is fragile**. `mediatype:(image)` excludes
   `mediatype:texts` items that contain images. Most period-book scans
   on IA are `texts` with downloadable image derivatives. **Fix**:
   broader search + derive image URL from the metadata endpoint.
6. **Network timeouts**. Log shows several `LoC search failed: The read
   operation timed out` at `timeout=20`. **Fix**: one retry with
   exponential backoff.

Concrete next steps in priority order:
- Add `query_simplify(q)`; test on the 22 queries from the current plan.
- Lower Wikimedia `_MIN_DIMENSION` to 320 (or skip when `thumburl` is set).
- Add unit tests with known-good queries (`Jafar al-Askari`, `Faisal bin
  Hussein 1920`, `Mahmud Shevket Pasha`) that **fail** if a source returns 0.
- Consider a fifth source: Wikipedia article images via
  `prop=pageimages|images` on the article slug. For named subjects the
  lead image is usually the documentary photo we want — one call, no
  vision pass needed.

#### 7.4 Vision scoring fails open in a way that defeats source priority

The first vision call (shot 4) succeeded; from shot 5 onward, every
call returned HTTP 400 `credit_balance_too_low`. `VisionScorer.score()`
catches the exception and stamps `(subject=2, quality=2, cinematic=1) = 5`
(`_apply_neutral_score`) so the candidate isn't silently dropped.

Downstream consequence: with **all** candidates from all sources tied at
5, `sorted()` is stable and the **original list order** breaks the tie.
Original order is `Fetcher.web_sources = [LoC, Wikimedia, IA, Pexels]`.
LoC/Wikimedia/IA all returned 0 candidates anyway (§7.3), leaving Pexels
as the sole survivor — so Pexels wins every shot **by elimination, not by
quality**. Visible result: a documentary about an Ottoman general 1904–1936
gets contemporary Pexels clip-art.

| Shot query | Pexels winner (verbatim from log) |
|-----------|------------------------------------|
| `Jafar al-Askari Iraqi general historical portrait` | "A stylish businessman with a briefcase exits a plane" |
| `Mahmud Shevket Pasha Ottoman general portrait historical` | "Close-up of bronze Ottoman soldier statues in Istanbul" |
| `Arab Revolt 1916 Sharif Hussein Faisal forces historical photograph` | "Libyan soldiers holding rifles and red flares" |
| `Jafar al-Askari portrait Iraqi statesman historical` | "Vandalized sculpture in a Baghdad park" |

Fixes:
- **Restore Anthropic credits** — without them both the planner (Sonnet)
  and the scorer (Haiku) degrade. Treat the credit balance as critical
  path.
- **Change the fail-open policy**: when at least one *real-scored*
  candidate exists in the pool, drop the unscored neutral-5 ones from
  the ranked list. When *no* candidate scored cleanly, fall back to
  source priority (which is what currently happens, just with extra
  noise).
- **Add a circuit breaker**: after N consecutive vision errors with the
  same error class (`invalid_request_error / credit_balance_too_low`),
  disable vision entirely for the rest of the run and rely on source
  priority alone. Log once, not 100 times. The log is currently 70 KB
  almost entirely from one repeated error.

### Tier 2 — quality plateaus

#### 7.5 Forced alignment is using the interpolated backend

WhisperX/Whisper install is commented out in cell 2 of the notebook.
The interpolated backend distributes time by character count
(~12 chars/sec heuristic). Drift can be ±200–500 ms per word. Caption
sync is acceptable for a documentary; shot-boundary precision suffers
because `_snap_to_word_boundaries` snaps to *interpolated* word
endpoints, not real ones.

Cost trade-off:
- WhisperX: ~30–60 s on CPU for a 3-min file, ~600 MB peak RSS, free.
- Whisper-only: ~45 s, similar memory, less accurate word boundaries.
- Interpolated: instant, free, ~300 ms typical drift.

For Streamlit Cloud's 1 GB ceiling, WhisperX peaks may collide with
FFmpeg work. Two paths:
- Run alignment in a separate subprocess so the model RAM is reclaimed
  before the renderer starts.
- Accept interpolated for now; revisit when ElevenLabs TTS lands (cleaner
  audio → easier alignment).

#### 7.6 Shot duration distribution skews long

Audit: average 9.09 s, range 4.49–12.17 s. The planner prompt's target is
~5 s per shot, and the hard caps are 10–12 s. The 14 % auto-split rate
shows Sonnet is brushing against the caps. Documentary pacing favours
4–6 s holds; 9 s averages feel slow. Two reasons it ran long:

1. Only 2 sections parsed (§7.2) → planner had less structural
   pressure to introduce variety.
2. `_sized_target_shots(391, 5.0)` returns the minimum of `(180/5 +
   (391-180)/5.5)` and 65 ≈ 65. The planner was *told* to aim for 65
   shots but returned 43. Sonnet's interpretation of "documentary pacing"
   tilts longer than the prompt asks.

Fixes:
- Tighten the prompt: change "5–8 s on typography and portraits" → "4–6 s
  on typography and portraits". Add an explicit rule: "Average shot
  duration must be 5.0–6.5 s."
- In `_validate_plan`, if `avg < 5.0` or `avg > 7.0`, log a warning so
  this regresses visibly.

### Tier 3 — polish

#### 7.7 Pillow typography cards for unmatched image shots

When all sources for an image shot return nothing or vision rejects
everything, `render._placeholder_card()` produces a cream card showing
the search query in Latin. It's *technically* fine but reads as a "TBD"
placeholder. Replace it with a fully-styled typography card that
reuses the *Arabic* key phrase from the *same section's* text — turns
gaps into intentional design moments.

#### 7.8 Animated word-by-word reveal on typography shots

Currently `static_hold`. A 0.4 s per-word reveal on `pull_quote` and
`name_reveal` would dramatically improve perceived production value
without any new sources. The shaping is already RTL-correct (libraqm or
arabic_reshaper + python-bidi), so it's just FFmpeg subtitle timing on
top of the existing PNG.

#### 7.9 ElevenLabs TTS (handoff from Phase 2)

Tier 2 in the master plan; cleaner audio also helps WhisperX alignment
(§7.5). Stub exists in `phase2/tts.py`.

---

## 8. Working Configuration

### CLI invocations from `_phase3_b2c.ipynb`

```bash
# Cell 8 — alignment sanity check (interpolation only; instant)
python phase3_run.py \
  --script samples/al_askari_script.txt \
  --audio  output/al_askari_audio.mp3 \
  --align-only \
  --align-backend interpolated

# Cell 9 — plan the shots (one Sonnet call, ~90 s, ~$0.10)
python phase3_run.py \
  --script         samples/al_askari_script.txt \
  --audio          output/al_askari_audio.mp3 \
  --book-title     "مذكرات جعفر العسكري" \
  --character-name "Jafar al-Askari" \
  --plan-only \
  --save-plan      output/al_askari_plan_v2.json
# NOTE: --character-name is in English (Latin), not Arabic — for the
# benefit of LoC/Wikimedia/IA which can't search Arabic well.

# Cell 10 — audit the plan
python audit_plan.py output/al_askari_plan_v2.json

# Cell 11 — render (~21 min, runs in &-background)
python render_plan.py \
  --plan           output/al_askari_plan_v2.json \
  --audio          output/al_askari_audio.mp3 \
  --output         output/final_cut.mp4 \
  --anthropic-key  "$ANTHROPIC_API_KEY" \
  --pexels-key     "$PEXELS_API_KEY" \
  --book-title     "مذكرات جعفر العسكري" \
  --character-name "Jafar al-Askari" \
  > output/render.log 2>&1 &
```

### Optional audit with audio cross-check

```bash
python audit_plan.py output/al_askari_plan_v2.json \
  --script samples/al_askari_script.txt \
  --audio  output/al_askari_audio.mp3
```

Adds: plan-end vs. real audio duration delta, verbatim check of typography
text against the script.

### Required environment

| Variable | Where | Required for |
|----------|-------|--------------|
| `ANTHROPIC_API_KEY` | `.env` at repo root or `_Phase3/`, or `--anthropic-key`, or Colab Secrets | Sonnet planner, Haiku vision scoring |
| `PEXELS_API_KEY`    | same | Pexels image source (currently the only working source — see §7.3) |

### Models used

| Task | Model | Cost / 3-min video |
|------|-------|---------------------|
| Shot planner (one call) | `claude-sonnet-4-6` (24,000 max_tokens, streaming) | ~$0.10 |
| Image relevance scorer | `claude-haiku-4-5-20251001` (vision, ~150 max_tokens) | ~$0.50 (for ~100 candidates) |
| Forced alignment | WhisperX `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` — currently disabled, interpolation used instead | $0 either way |

---

## 9. Recommended Session Order

The order below is by leverage, not by difficulty.

1. **Restore Anthropic credits** before any further benchmarking. Without
   them both the planner and the scorer degrade silently (§7.4).
2. **Fix the source query strategy** (§7.3). This is the single biggest
   visual-quality win — currently every image shot ends up on Pexels
   modern stock by elimination, not by choice.
3. **Fix the section parser** (§7.2). Currently 5 logical sections
   collapse to 2, which removes structural pressure on the planner and
   makes shot variety harder.
4. **Patch the vision fail-open policy** (§7.4). Even when credits are
   available, the policy should demote unscored candidates *only when
   scored ones exist*, not blanket-promote them to neutral 5.
5. **Decide on Whisper/X for alignment** (§7.5). Until ElevenLabs lands,
   interpolation is good enough; WhisperX would tighten caption sync at a
   memory cost.
6. **Pillow typography placeholder cards** (§7.7) — converts the "TBD"
   look into a design feature when sources fail.
7. **Amiri discovery on system paths** (§7.1) — eliminates a 6 MB cold-
   start download on Streamlit Cloud.
8. **Tighten shot duration distribution** (§7.6) — change prompt target
   to 5.0–6.5 s.

---

## 10. Things Not To Touch (or touch with care)

- **The plan/render split.** Two CLIs, two responsibilities. Mixing them
  was the original mistake; the split is what makes iteration fast.
- **`_validate_plan` invariants.** Renderer assumes them. Loosen one →
  break the concat pass or the caption layer.
- **Arabic rendering uses `libraqm` when available, falls back to
  `arabic_reshaper` + `python-bidi`.** Don't add a third path. Don't use
  FFmpeg `drawtext` for any Arabic — it has no bidi.
- **800 px image-resize before vision scoring** (`vision.score`,
  `vision.py:117`). Larger → API 400. Known constraint from v1;
  preserved here.
- **Stream-copy concat in `_concat_clips`.** Works only because every
  shot clip uses identical encoder settings. Changing one shot's encoder
  profile will silently break the concat — fall back to filter_complex
  concat if you need per-shot variations.
- **`fail-open` in `VisionScorer.score`.** Don't flip it to fail-closed —
  that would drop *all* candidates the moment Anthropic has a 5 s blip.
  Instead, demote unscored candidates *only* when scored ones exist
  (§7.4).

---

## 11. Quick Reference — Useful One-Liners

```bash
# Histogram of shot types in a plan
python -c "import json; from collections import Counter; \
  d=json.load(open('output/al_askari_plan_v2.json')); \
  print(Counter(s['visual'] for s in d))"

# Total plan duration vs. audio
python -c "import json; d=json.load(open('output/al_askari_plan_v2.json')); \
  print('plan end:', d[-1]['end'])" && \
  ffprobe -v quiet -show_entries format=duration \
    -of default=nw=1:nk=1 output/al_askari_audio.mp3

# Find which shots ended up on Pexels (= which queries failed every other source)
grep "using fetched image from pexels" output/render.log | wc -l

# Inspect the planner's raw response (saved on every plan build)
less output/planner_raw_response.txt

# List cached images after a real render
ls -la ~/.cache/lamahat/images/
```

---

## 12. Known Environment Constraints

| Constraint | Detail |
|-----------|--------|
| Streamlit Cloud RAM | 1 GB — keep FFmpeg work in subprocesses. v2 render obeys this. |
| Python | **3.12.13** (set in Cloud Advanced settings). Don't assume 3.13/3.14. |
| Colab CPU runtime | ~21 min for a 391 s render at 1920×1080. Mostly FFmpeg + vision RTTs. |
| FFmpeg subtitle path escaping | `:` and `\` need escaping in `-vf "ass=…"`. See `_mux_final`. |
| Claude vision max image size | Always resize to ≤ 800 px wide. Larger → 400 error. |
| Arabic font in ASS | `Fontname: Amiri` (`fonts-hosny-amiri` Debian package on Cloud). |
| Pexels key | Optional in the contract, mandatory in practice given §7.3. |
| Anthropic key | Required for planner AND scorer. Treat as critical-path. |
| GitHub upload size | Test artefacts > 25 MB get truncated/partial in `_Phase3/output/`. The `final_cut_3a.mov` in this repo is a 181 s **preview** of a 391 s render. Real output: `output_files.zip` from cell 13. |
