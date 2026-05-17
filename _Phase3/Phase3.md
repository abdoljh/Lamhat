# Phase 3 — Visual Generation · Session Handoff

> **Working tree**: `Lamahat/_Phase3/`
> **Status**: end-to-end working. Latest render — 43-shot plan, 391 s audio,
> 26.6 MB / 391 s MP4 @ 1920×1080, 25 fps. Wall time ~21 min on Colab CPU.
> **Inputs**: `samples/al_askari_script.txt` (Phase 1b output, 4045 chars,
> 653 word tokens) and `output/al_askari_audio.mp3` (Phase 2 gTTS, 391.0 s,
> URL: `github.com/abdoljh/Lamahat/blob/main/_Phase3/output/al_askari_audio.mp3`).
> **GitHub artefact note**: `output/final_cut_3a.mov` (12.8 MB / 181 s) is a
> *partial preview* — first ~46 % of the real render — uploaded under GitHub's
> 25 MB inline limit. The real output is `output_files.zip → final_cut.mp4`
> saved to Drive by `_phase3_b2c.ipynb`.

---

## 1. What Phase 3 Is Now (vs. the v1 in CLAUDE.md)

The original Phase 3 was section-based: parse 4 Arabic sections, pull 2-3
Wikimedia images per section, Ken-Burns them, crossfade, mux. That code still
lives at `phase3/__init__.py → generate_background_video()` and remains
reachable from Streamlit. It has been **superseded** by a *shot-based*
pipeline that is now the default end-to-end path.

**v2 architecture (current)** — a *shot plan* is the source of truth. The
plan is a list of 30–65 timestamped `Shot` dataclasses produced by one Claude
Sonnet 4.6 call; the renderer executes the plan without making creative
choices. Plans are JSON, inspectable, diff-able, regeneratable from cache.

```
Script + Audio ──► align.py  ──► word_timings (WhisperX | Whisper | interp)
                                       │
                                       ▼
                                   plan.py   ──► shot_plan.json
                                                 (one Sonnet call, ~$0.10)
                                       │
                                       ▼
                                  render.py  ──► MP4
                                       ▲
                                       │
                              sources/Fetcher
                              (LoC → Wikimedia → IA → Pexels,
                               + cache + user-upload + book-extract,
                               Haiku-vision-scored)
```

**Design philosophy** (preserved verbatim from the prior session for posterity):

1. **Plan-then-render is the unlock.** The plan is a JSON document —
   inspectable, diffable, regeneratable without re-rendering. You should be
   able to look at a plan and know whether the video will be good before a
   single FFmpeg call runs.
2. **The "shot" is the unit.** Not the section. A shot has start/end (from
   word timings), a visual spec (search query + motion + framing), and
   optional overlay text. The compositor just executes the plan — no
   decisions, no fallbacks, no surprises.
3. **Honest degradation.** If WhisperX isn't installed, fall back to
   interpolated word timings. If a web source fails, fall back to the next.
   If all sources fail, fall back to a placeholder card. The render must
   complete.

### Why "shots" instead of "sections"

The diagnosis from the prior session that drove the rewrite, kept here so a
future Claude (or future me) doesn't re-litigate it:

- The original v1's *visual unit* was a 30–60 s section. Ken-Burnsing 3
  images across 45 s means each image holds for 15 s — an eternity in modern
  video. Documentary editors cut every 4–8 s with narration variation.
- Cuts were *decoupled from speech*. Visuals changed at section boundaries,
  but the dramatic moments in narration (a name, a date, a turning phrase)
  happen mid-section. Without forced alignment the system can't see them.
- Wikimedia is the *wrong primary source* for biography. It's optimized for
  "is there a photo of this thing", not "is there a *compelling* photo".
- Ken-Burns-on-everything is the AI-video tell. Real docs mix static holds
  on faces, fast pushes on action beats, whip pans for transitions.

The shot-based architecture solves all four: word-aligned cuts (or
interpolated word timings as a fallback), a 7-element motion vocabulary, and
a 4-source image waterfall ranked by Haiku vision scoring.

### Target platform & budget (locked decisions from prior session)

| Decision | Choice |
|---|---|
| Platform | **YouTube long-form** (1920×1080, 25 fps, 4–7 min) |
| Cost tier | "Quality matters" — ~$0.20–0.50 / video is fine; not $0.06 |
| Typography aesthetic | **Family A — Aljazeera Documentary editorial** (cream/charcoal, Amiri, hairlines, no Islamic geometric ornament) |
| Color grading | Knob with cinematic-warm as default; per-section variation later |
| Section transitions | The `section_mark` typography shot *is* the transition; hard cuts everywhere else, no crossfades |

### Active issues checklist (this session)

A live ledger of the five issues identified after the latest end-to-end
run.  Updated each time we close one out.

| # | Issue | Status | Tracking |
|---|---|---|---|
| 1 | **Color philosophy** — knob with cinematic-warm default, tunable per section | ☐ open | §15.1 |
| 2 | **Typography aesthetic** — Family A too faint; offer Families B & C as selectable variants for testing | ☐ open | §15.2 |
| 3 | **Section transitions** — current rhythm too slow, doesn't hook the audience | ☐ open | §15.3 |
| 4 | **Captions** — title-card subtitle too small; main captions OK; under-line text small; subtitles appear merged | ☐ open | §15.4 |
| 5 | **Online/offline asset review** — pre-render dossier of all candidates + character pin + per-shot override | ✅ **closed in this drop** | §15.5 |

Working principle for all five: every change exposes a knob (CLI flag,
config field, or dossier entry), keeps the existing default working,
and lands testable in isolation.  The structural plan-then-render
architecture means the user can iterate on aesthetic choices by
re-rendering against the same plan — no replanning, no replanning
cost, no re-fetching.

Family A was chosen explicitly over Family B (Netflix-doc cinematic dark
gradients — "reads as imported, not native") and Family C (manuscript /
Islamic geometric ornament — "too on-the-nose"). The aesthetic is
deliberately quiet: when this plays for someone who reads Arabic
journalism and watches Aljazeera Documentary, the visual language has to
feel native, not borrowed.

### Two CLIs

| CLI | Purpose | Stops at |
|-----|---------|----------|
| `phase3_run.py` | Plan **and/or** render in one go. Owns the v1 path too. | configurable: `--dry-run`, `--keywords-only`, `--align-only`, `--plan-only`, or full render |
| `render_plan.py` | Render a previously-saved plan to MP4. | always produces an MP4 (or writes a manifest with `--build-manifest`) |

Splitting plan vs. render is deliberate. Planning costs a Sonnet call
(~$0.10 + ~90 s wall). Rendering costs CPU minutes (~20 min). When iterating
on visuals you re-render; when iterating on shot choices you re-plan. The
split is what made the auto-split / cap-tuning / typography-template
iterations debuggable across the prior session — you could read a 43-shot
JSON, audit it, fix the prompt, regenerate, *then* render once.

### File map

```
_Phase3/
├── phase3_run.py             # Plan-OR-render CLI
├── render_plan.py            # Render-only CLI (consumes plan JSON)
├── audit_plan.py             # Quality audit of a saved plan
├── _phase3_b2c.ipynb         # Colab driver — the canonical run lives here
├── samples/al_askari_script.txt
├── output/                   # render.log (51-shot prior run), audio, MOV preview
└── phase3/
    ├── __init__.py           # v1 entrypoint generate_background_video() — legacy
    ├── parser.py             # Arabic section regexes + estimate_durations
    ├── align.py              # WhisperX | Whisper | interpolation → WordTiming list
    ├── plan.py               # Sonnet 4.6 shot planner + Shot dataclass + JSON I/O
    ├── render.py             # Plan → MP4 (assets, motion, captions, mux)
    ├── typography.py         # Family A typography cards (864 LOC, 5 templates)
    ├── compositor.py         # v1 background video assembler (still used by v1 path)
    ├── effects.py            # ffprobe wrapper + Ken Burns helpers (v1)
    ├── keywords.py           # v1 keyword generator (Haiku per section)
    ├── pexels.py             # v1 video-clip fetcher
    ├── subtitler.py          # v1 ASS subtitle writer
    ├── wikimedia.py          # v1 image fetcher + vision scorer
    ├── render_previews.py    # Dev helper — render a typography template grid
    ├── test_smoke.py / test_typography.py / test_resilient.py
    └── sources/              # v2 image-fetch waterfall
        ├── __init__.py       # Fetcher orchestrator + FetcherConfig
        ├── base.py           # Source ABC, ImageCandidate, free-license detection
        ├── loc.py            # Library of Congress JSON API
        ├── wikimedia.py      # MediaWiki API; 400 px min dimension filter
        ├── internet_archive.py  # archive.org advancedsearch
        ├── pexels.py         # Pexels v1 photos endpoint (api_key required)
        ├── user_upload.py    # shot_NN.jpg overrides from a user directory
        ├── book_extract.py   # Phase 1a photo bank (vision-scored against query)
        ├── cache.py          # disk cache keyed by query
        └── vision.py         # Claude Haiku vision scorer, 3-axis rubric (0-3 each)
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
`slow_pull`, `pan_left`, `pan_right`, `ken_burns`. `static_hold` is applied
to typography and placeholder cards always; the other six only fire for
fetched real images (see `_MOTION_FILTERS` in `render.py`).

**Typography templates** (5): `pull_quote`, `name_reveal`, `date_stamp`,
`chapter_heading`, plus implicit `title_card` / `section_mark` styles. All
rendered by `typography.py`.

### Plan invariants (enforced by `plan._validate_plan`)

1. Shots are sorted by `start` and **contiguous**: `shot[i].end == shot[i+1].start`.
2. First shot starts at `0.0`; last shot ends at `total_duration_sec`.
3. Per-visual hard caps (with 0.1 s floating-point tolerance); above the cap
   `_validate_plan` splits a shot into ~5 s pieces and tags each
   `[auto-split k/n]`:
   - `typography`, `portrait` → **12 s**
   - `archive`, `broll`, `location`, `object` → **10 s**
   - `section_mark` → **7 s**
   - `title_card` → **7 s**
4. **Adjacent shots with identical `(visual, search_query)` or
   `(visual, typography_text)` are merged** (`_shots_can_merge`) and their
   `caption_text` concatenated. This pass runs *after* splitting, so it
   reverses any unnecessary split. It's the single most important plan
   post-processing step — see §6 history of how it was tuned.
5. Field exclusivity: typography-kind shots (`title_card`, `section_mark`,
   `typography`) keep `typography_text`, drop `search_query`. Image-kind
   shots do the opposite (`_normalise_fields`).
6. Shot boundaries snap to actual word boundaries (`_snap_to_word_boundaries`).
   Minimum shot duration after snap: 1.5 s.

These invariants are why the renderer can be dumb — by the time it sees a
plan, the math is consistent.

### Title card / typography template dispatch — known footgun

When `visual` is `title_card`, `section_mark`, or `chapter_heading`, the
renderer **forces the template by visual type** regardless of what
`typography_template` says. Sonnet often annotates a `title_card` shot with
`typography_template: "chapter_heading"` (hedging), and trusting that
annotation produces the wrong opening visual. Fix is in `render.py` —
typography template hint is only respected when `visual == "typography"`.

---

## 3. Sources Subsystem (image-fetch waterfall)

`sources/Fetcher.fetch_for_shot(query, shot_index)` runs this priority order:

1. **User upload** — `--user-dir <path>`. File matched by name pattern
   `shot_NN.jpg` (NN = 1-indexed shot number) or by `manifest.json`.
2. **Book extract** — `--book-extracts <Phase1a photos.zip or dir>`.
   Vision-scored against the shot query (requires `--anthropic-key`).
3. **Disk cache** — `~/.cache/lamahat/images` keyed by query hash. Disable
   with `--no-cache`.
4. **Live web fetch** in order: LoC → Wikimedia → IA → Pexels. All
   candidates from all sources are pooled, downloaded, vision-scored, then
   ranked by `vision.rank_candidates`. Top survivor wins.

`VisionScorer` (Haiku, `claude-haiku-4-5-20251001`) emits three integer
scores per image (`subject` / `quality` / `cinematic`, 0–3 each, total 0–9).
Keep threshold: `total ≥ 4 AND subject ≥ 1`. **Critically, the Haiku call
is fail-open** — on any exception the candidate is assigned
`(subject=2, quality=2, cinematic=1) = 5` and kept. See §7.4 for the
downstream consequence.

### License posture

`base.is_free_license()` accepts everything CC-*, PD, "no known
restrictions", plus unknown. Rejects anything with `NC`, `ND`, "all rights
reserved". Pexels is hard-coded as `"Pexels License"` (permissive but with
attribution conventions — double-check before public release).

### Required-images manifest mode

`render_plan.py --build-manifest output/required_images.txt` produces a
review table without hitting the network:

```
shot_05  portrait  "Jafar al-Askari Iraqi general historical portrait 1920s"
shot_08  archive   "Ottoman Empire collapse historical document 1918"
shot_12  location  "Mosul Iraq historical photo 1904 Ottoman city"
...
```

You can review it before any render. Drop your own images into
`--user-dir` as `shot_NN.jpg`, or write a `manifest.json` mapping shot
indices to filenames. The renderer picks them up via path (1) of the
waterfall.

---

## 4. Rendering Pipeline (one MP4 from one plan)

`render.render_video(shots, out_path, *, audio_path, audio_duration_sec, config, on_progress)`:

1. For each shot:
   - Build a 1920×1080 PNG asset:
     - Typography visuals → `typography.render()` (Family A card)
     - Image visuals → `Fetcher.fetch_for_shot()` → copy chosen JPEG to PNG
     - Fallback → `_placeholder_card()` (cream card with the search query)
     - Final fallback on exception → `_error_card()` (so the timeline
       doesn't collapse — audio sync depends on every shot producing a clip
       of its planned duration)
   - Encode the PNG to an MP4 clip of the shot's exact duration. Motion only
     fires when `is_real_image=True`; typography and placeholders always
     `static_hold`. Zoom is computed against a 1.6× buffer to avoid blurry
     pan-edges. For real photos already larger than the output buffer, the
     code probes native dimensions to avoid unnecessary upscaling.
2. **Stream-copy concat** of all shot clips → `background.mp4`. Works only
   because every clip uses identical encoder settings (`libx264 -preset
   ultrafast -crf 22 -pix_fmt yuv420p -r 25`). Change one shot's profile
   and the concat silently breaks.
3. **ASS captions** (`_write_captions`):
   - **Current**: white Amiri text with charcoal outline (BorderStyle 1).
   - **Intended** (Family A spec): small Amiri Regular charcoal on
     translucent cream bar, bottom 8 % of frame, "Aljazeera Documentary
     subtitle, not TV captions".
   - **Why the gap**: libass's BorderStyle 3 + alpha-tinted BackColour
     doesn't actually blend — the "50% cream" backplate rendered as opaque
     white. White-on-charcoal-outline was the working fallback. To restore
     the intended look, the cleanest path is to burn the backplate as a
     separate semi-transparent FFmpeg `drawbox` alongside the ASS subs.
4. **Final mux** (`_mux_final`): single FFmpeg pass that re-encodes the
   video (required to burn subs), adds AAC audio at 192 kbps with
   `-shortest`, then `-t max_duration` if set. The re-encode pass is ~5 min
   of the ~21-min total.

Everything FFmpeg is shelled via `subprocess.run`, working under
`tempfile.TemporaryDirectory` so RAM stays low — important for Streamlit
Cloud's 1 GB ceiling.

### Captions skip typography shots

A typography shot already shows its Arabic text full-screen at hero size.
Drawing the caption again at the bottom would be redundant. `_write_captions`
filters: `s.visual not in TYPOGRAPHY_VISUALS`. Don't undo this.

---

## 5. The Canonical Run, Decoded

The notebook `_phase3_b2c.ipynb` is the authoritative reference for what
works today. The cell-by-cell pipeline:

| Cell | What it does | What its output proves |
|------|--------------|------------------------|
| 0 | Mount Drive, copy `_Phase3/` into `/content` | Colab working dir is `/content`, not `_Phase3/` — CLI paths are relative |
| 1 | `pip install anthropic` (0.102.0) | Sonnet + Haiku reachable |
| 2 | WhisperX/Whisper install — **commented out** | Alignment uses interpolated backend (§7.5) |
| 3 | `apt install fonts-hosny-amiri` (0.113-1) | Amiri available system-wide via fontconfig |
| 4 | `pip install arabic-reshaper python-bidi` | Fallback path for non-libraqm Pillow builds |
| 5–6 | matplotlib + `phase3.typography.FONT_PATHS` sanity check | Reveals Amiri-discovery bug — §7.1 |
| 7 | Load API keys from Colab Secrets | Both set |
| 8 | `phase3_run.py --align-only --align-backend interpolated` | 653 word tokens, **only 2 sections parsed** — §7.2 |
| 9 | `phase3_run.py --plan-only` | Sonnet returns 43 shots covering 0.00–391.00 s in 91.4 s; one call (~$0.10) |
| 10 | `audit_plan.py` | 0 gaps/overlaps; 35 % typography (in target); 14 % auto-split; 22 search queries, avg 7.5 words; no bare queries |
| 11–12 | `render_plan.py` background + tail-monitor | "Done in 1263 s — output/final_cut.mp4 (26.6 MB)" |
| 13 | Zip outputs (excluding the .mp3) | `output_files.zip` with `final_cut.mp4 + render.log + plan.json + word_timings.json + planner_raw_response.txt` |
| 14 | Copy zip to Drive | Final deliverable on `/MyDrive/_Phase3/output_files.zip` |

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
   closing         10 shots                 ← see §7.2

✓  Auto-split shots: 6/43 (14%) from 6 original(s)
Typography texts: 21 unique (avg 11.4 words)
Search queries: 22 non-empty, avg 7.5 words   ✓ none bare
```

The plan is healthy on every dimension except *section structure*.
Auto-split is 14 %, comfortably below the 20 % "tighten the prompt" line.
Typography density at 35 % sits right on the prompt's target ceiling.

---

## 6. History of the Plan-Validation Iteration

Worth preserving because the cap values look magic in the code and a future
session might lower them "to keep shots short". They were tuned through
three iterations of *empirical* feedback — don't lower them again without
re-reading this section.

| Iteration | Cap | Result |
|---|---|---|
| v1 (initial) | 6 s for everything | **74 % auto-split**, 67 Sonnet shots became 106 pieces, avg 3.7 s — TikTok pacing not documentary |
| v2 (raised) | 8 s for everything | 16 % auto-split, 67 → 64. Better, but typography pull-quotes that genuinely needed 13 s of read time were being chopped into three identical 4.4 s halves |
| v3 (type-aware + merge) | typography/portrait 12 s, archive/broll/location/object 10 s, section_mark 7 s, **+ merge-adjacent-duplicates pass** | 3 % auto-split, avg 6.5 s. Documentary pacing |

The lesson: a typography pull quote held for 12 s is correct, not a runaway
that needs splitting. Documentary pacing favours longer holds on faces and
hero text; only when shots exceed *visual-type-specific* caps do they need
splitting. And even then, if two split pieces have identical content, the
merge pass fuses them back so the caption layer doesn't see three separate
2-second caption windows for what was one 6-second hold.

The 0.1 s floating-point tolerance on the cap check matters: shots that
land at exactly the cap (e.g. an 8.04 s archive after word-boundary
snapping) used to get split into two 4 s halves. Tolerance prevents that.

---

## 7. Open Issues, In Priority Order

### Tier 1 — these distort *every* output

#### 7.1 Amiri discovery falls through despite a system install — **FIXED**

**Status**: patched in the bundled `typography.py` and `render.py`.
See `typography.diff` for the full change.

**Original failure mode** (now resolved): cell 6 reported
`Amiri not found on system — downloading from upstream`, even though
cell 3 successfully installed `fonts-hosny-amiri` to
`/usr/share/fonts/opentype/fonts-hosny-amiri/`. The 6 MB fallback
download was triggered on every cold start.

**Root cause** (revealed by reading the actual Debian package contents):
the Ubuntu jammy package `fonts-hosny-amiri 0.113-1` (what Colab installs)
ships `Amiri-Slanted.ttf` and `Amiri-BoldSlanted.ttf` — not
`Amiri-Italic.ttf` and `Amiri-BoldItalic.ttf` that the discovery code
required. Upstream renamed `Slanted` → `Italic` in version 0.114 (2020),
but Ubuntu still packages the pre-rename release. The
`all(found[k].exists() for k in required)` gate failed on `italic` and
`bold_italic`, every system-path strategy fell through, and the code hit
the upstream download path.

Two compounding issues hid the root cause:
- The repo's bundled `_Phase3/fonts/` directory (already present at
  `github.com/abdoljh/Lamahat/_Phase3/fonts/`) was not searched at all by
  discovery, and the Colab cell 0 doesn't copy `fonts/` into `/content/`.
- The fail-open message was misleading: "Amiri not found on system" is
  technically true but doesn't say *which paths* were tried or why each
  was rejected.

**The fix** (4 changes):

1. **Repo-bundled `fonts/` is now Strategy 1** — searched before
   fontconfig, environment override, or system paths. Six probe paths
   cover the package layout (`_Phase3/fonts/`), CWD-relative invocations,
   and the live Colab Drive mount (`/content/drive/MyDrive/_Phase3/fonts`)
   in case the notebook doesn't copy the directory.
2. **Per-weight filename aliases** via a `_FONT_ALIASES` map. `italic`
   slot now accepts `Amiri-Italic.ttf` or `Amiri-Slanted.ttf`;
   `bold_italic` accepts `Amiri-BoldItalic.ttf` or
   `Amiri-BoldSlanted.ttf`. The Colab Debian package now resolves cleanly.
3. **Required weights narrowed to `regular` + `bold`**. Italic and
   `bold_italic` are optional — `_font()` already falls back to regular
   when an italic weight is missing, so requiring them at discovery time
   was an overconstraint.
4. **`render.py` passes `fontsdir=<amiri dir>` to the libass `ass` filter**
   so burned-in captions render Amiri even when fontconfig hasn't been
   refreshed (the secondary failure mode where libass silently
   substitutes DejaVu and Arabic letters lose shaping).

**Diagnostics also improved**: each discovery strategy now logs its
specific reason for not matching (e.g. `fc-match: returned
/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf (not an Amiri file —
fontconfig substituted)`), and the final RuntimeError lists every path
tried. No more "Amiri not found on system" one-liners.

**Verification**: run `python verify_font_discovery.py` (bundled). It
logs which strategy succeeded, prints all resolved weight paths, and does
an end-to-end Pillow render smoke test.

| Scenario | Before patch | After patch |
|---|---|---|
| Repo `fonts/` available | Ignored, falls through to download | **Strategy 1 — picks it immediately** |
| Colab post-`apt install fonts-hosny-amiri` (0.113-1) | Falls through (italic filename mismatch), downloads | **fc-match resolves, aliases pick up `Slanted` files** |
| `fc-match` returns DejaVu substitute (matplotlib race) | Falls through silently, downloads | Rejected with reason logged, falls through to system paths |
| Fresh Colab cold start, fonts not yet on Drive | 6 MB upstream download per cold start | **Drive-mount probe finds them; one Drive read** |
| Truly no Amiri anywhere | Downloads silently | Downloads; on download failure, lists every path tried |

#### 7.2 Section parser only recognises rigid template headers

(Note: in the prior session this was *accepted* as adequate — "Sonnet
correctly identifies subtopic boundaries via `section_mark` shots". I'm
flagging it here as still worth addressing because it removes structural
pressure on the planner. Demote to Tier 2 if you disagree.)

The alignment cell reports `653 word tokens, 2 sections`. The real script
has 5 logical sections — opening + 3 descriptive points + closing — but
`parser._SECTION_HEADERS` only matches the rigid v1 template
(`النقطة الأولى/الثانية/...`, `الخاتمة`, `تقديم الكتاب`). The current Phase
1b summariser emits descriptive titles instead:

```
Line  9: من الموصل إلى الاستانة — رحلة التحديث والطموحْ
Line 17: الصراع الأيديولوجي والسياسي — بين الولاء والحلمْ
Line 25: الحرب والاختبار النهائي — الفعل والالتزامْ
Line 33: الخاتمة: شهادة لا تموتْ                    ← only this matches
```

Result: `opening = lines 1–32` (one 287-second blob) and
`closing = lines 33–43`. Audit confirms: `opening 33 shots, closing 10
shots`. Sonnet still introduces 4 `section_mark` shots on tonal breaks,
but the *intended* structural mapping (one set of visuals per thematic
point) is lost.

**Fix options**, ranked by leverage:
1. **Loosen the parser**: detect any line that ends with `.` or `ْ` and
   sits between blank lines. Cross-check with line length (<80 chars).
   Promote those to auto-generated sections `point_1`, `point_2`, ….
2. **Synchronise with Phase 1b**: have Phase 1b emit a sidecar JSON
   listing section boundaries by line number. Cleaner — keeps the script
   copy-pasteable.
3. As a stop-gap, set `parser._SECTION_HEADERS` to a single broad pattern
   matching "any short line followed by a blank line".

#### 7.3 LoC / Wikimedia / Internet Archive return 0 candidates per query

The single biggest visual-quality issue. In the latest 43-shot plan, *every*
image shot's query went through this waterfall:

```
LoC:               0 candidates for 'Jafar al-Askari Iraqi general historical portrait'
Wikimedia:         0 candidates for 'Jafar al-Askari Iraqi general historical portrait'
Internet Archive:  0 candidates for 'Jafar al-Askari Iraqi general historical portrait'
Pexels:            3 candidates for 'Jafar al-Askari Iraqi general historical portrait'
```

But Wikimedia Commons demonstrably has `Category:Mahmud_Shevket_Pasha` with
PD photographs; LoC has 1880-1940 MENA holdings; IA has period books. The
problem is **query construction and filter strictness**, not content
availability.

**Probable causes, in descending order**:

1. **Over-specific multi-word queries**. MediaWiki's `gsrsearch` is
   phrase-AND. Six tokens — `'Jafar al-Askari Iraqi general historical
   portrait'` — require all six in file metadata. **Fix**: add
   `query_simplify()` that strips generic tails (`portrait historical
   photograph archive picture`), keeps proper nouns + dates. Simplified
   first, full as fallback.
2. **Query/index mismatch**. LoC tags historical photos as "Ottoman Empire
   — History — 1909-1918" or by city/person — broad English phrases like
   "Ottoman Empire Arab officers 1910" don't hit those tags. **Fix**: tune
   the planner prompt to emit archive-style queries for `archive`/
   `portrait`/`location` visual types (period place names, dynastic
   tags, dates as ranges).
3. **400 px minimum dimension filter (Wikimedia)**. `wikimedia.py:_MIN_DIMENSION = 400`
   rejects results whose `thumbwidth/thumbheight` is below 400 px. Many
   period photos are stored as 300–380 px JPEGs and get rejected.
   **Fix**: drop to 320 or remove the filter when `thumburl` is present
   (thumbs are always rescalable).
4. **`-diagram -anatomy -chart -schematic` exclusion** passed raw to
   `gsrsearch`. Combined with narrow queries, removes borderline matches.
   **Fix**: opt-in.
5. **LoC facet filter** `'fa': 'online-format:image|original-format:photo,print'`
   sometimes returns 0 when the same query in LoC's web UI returns
   thousands. **Fix**: try without `fa`, post-filter in code.
6. **IA's `mediatype:(image)` excludes `texts` items** that contain images
   (most period-book scans). **Fix**: broader search, derive image URL
   from metadata endpoint.
7. **Network timeouts** at `timeout=20`. **Fix**: one retry with
   exponential backoff.

**Concrete next steps**:
- Add `query_simplify(q)`; test on the 22 queries from the current plan.
- Lower Wikimedia `_MIN_DIMENSION` to 320, or skip when `thumburl` is set.
- Add unit tests with known-good queries (`Jafar al-Askari`, `Faisal bin
  Hussein 1920`, `Mahmud Shevket Pasha`) that **fail** if a source returns 0.
- Consider a fifth source: Wikipedia article images via
  `prop=pageimages|images` on the article slug. For named subjects the
  lead image is usually the documentary photo we want — one call, no
  vision pass needed.

#### 7.4 Vision scoring fail-open defeats source priority

The first vision call (shot 4) succeeded; from shot 5 onward, every call
returned HTTP 400 `credit_balance_too_low`. `VisionScorer.score()` catches
the exception and stamps `(subject=2, quality=2, cinematic=1) = 5`
(`_apply_neutral_score`) so the candidate isn't silently dropped.

**Downstream consequence**: with all candidates from all sources tied at 5,
`sorted()` is stable and the **original list order** breaks the tie.
Original order is `Fetcher.web_sources = [LoC, Wikimedia, IA, Pexels]`.
LoC/Wikimedia/IA all returned 0 candidates anyway (§7.3), leaving Pexels
as the sole survivor — so Pexels wins every shot **by elimination, not by
quality**.

Visible result: a documentary about an Ottoman general 1904–1936 gets
contemporary Pexels clip-art.

| Shot query | Pexels winner (verbatim from log) |
|-----------|------------------------------------|
| `Jafar al-Askari Iraqi general historical portrait` | "A stylish businessman with a briefcase exits a plane" |
| `Mahmud Shevket Pasha Ottoman general portrait historical` | "Close-up of bronze Ottoman soldier statues in Istanbul" |
| `Arab Revolt 1916 Sharif Hussein Faisal forces historical photograph` | "Libyan soldiers holding rifles and red flares" |
| `Jafar al-Askari portrait Iraqi statesman historical` | "Vandalized sculpture in a Baghdad park" |

**Fixes**:
- **Restore Anthropic credits** — without them both the planner (Sonnet)
  and the scorer (Haiku) degrade. Treat the credit balance as critical
  path.
- **Change the fail-open policy**: when at least one *real-scored*
  candidate exists in the pool, drop the unscored neutral-5 ones from the
  ranked list. When *no* candidate scored cleanly, fall back to source
  priority (which is current behaviour, just with extra noise).
- **Add a circuit breaker**: after N consecutive vision errors with the
  same error class (`invalid_request_error / credit_balance_too_low`),
  disable vision for the rest of the run and rely on source priority alone.
  Log once. The render log was 70 KB almost entirely from one repeated
  error.

### Tier 2 — quality plateaus

#### 7.5 Forced alignment uses the interpolated backend

WhisperX/Whisper install is commented out in cell 2. The interpolated
backend distributes time by character count (~12 chars/sec heuristic).
Drift is ±200–500 ms per word — adequate for caption sync, but
`_snap_to_word_boundaries` snaps to *interpolated* word endpoints, not
measured ones.

Cost trade-off:
- WhisperX: ~30–60 s on CPU for a 3-min file, ~600 MB peak RSS, free.
- Whisper-only: ~45 s, similar memory, less accurate word boundaries.
- Interpolated: instant, free, ~300 ms typical drift.

Diacritics throw the interpolation off in a small way: characters in the
Arabic Presentation Forms block count toward duration, so a heavily
diacritised word like `مُذَكِّرات` gets 1.02 s while plain `جعفر` gets 0.41 s.
The TTS reads them at similar speed. Real WhisperX alignment removes this
distortion entirely.

For Streamlit Cloud's 1 GB ceiling, WhisperX may collide with FFmpeg work.
Two paths:
- Run alignment in a subprocess so model RAM is reclaimed before render.
- Accept interpolated for now; revisit when ElevenLabs TTS lands (cleaner
  audio → easier alignment).

#### 7.6 Shot duration distribution skews long

Audit: average 9.09 s, range 4.49–12.17 s. The planner prompt's target is
~5 s per shot; hard caps are 10–12 s. The 14 % auto-split rate shows Sonnet
is brushing against the caps. Documentary pacing favours 4–6 s holds; 9 s
averages feel slow.

Two reasons it ran long:
1. Only 2 sections parsed (§7.2) → planner had less structural pressure to
   introduce variety.
2. `_sized_target_shots(391, 5.0)` returns ~65, but the planner returned
   43. Sonnet's interpretation of "documentary pacing" tilts longer than
   the prompt asks.

**Fixes**:
- Tighten the prompt: change "5–8 s on typography and portraits" → "4–6 s
  on typography and portraits". Add explicit: "Average shot duration must
  be 5.0–6.5 s."
- In `_validate_plan`, warn if `avg < 5.0` or `avg > 7.0`.

### Tier 3 — polish

#### 7.7 Pillow typography cards for unmatched image shots

When all sources for an image shot return nothing or vision rejects
everything, `render._placeholder_card()` produces a cream card showing the
search query in Latin. It's *technically* fine but reads as "TBD". Replace
it with a fully-styled typography card that reuses the *Arabic* key phrase
from the same section — turns gaps into intentional design moments.

#### 7.8 Restore the intended caption styling

Captions currently use white-on-charcoal-outline (BorderStyle 1) as a
fallback. The Family A spec is small Amiri Regular charcoal on translucent
cream bar, bottom 8 % of frame. Path forward: burn the backplate as a
semi-transparent FFmpeg `drawbox` alongside the ASS subs (libass alone
doesn't honor alpha on BackColour with BorderStyle 3).

#### 7.9 Animated word-by-word reveal on typography shots

Currently `static_hold`. A 0.4 s per-word reveal on `pull_quote` and
`name_reveal` would dramatically improve perceived production value
without any new sources. RTL shaping is already correct (libraqm or
arabic_reshaper + python-bidi fallback), so it's just FFmpeg subtitle
timing on top of the existing PNG.

#### 7.10 ElevenLabs TTS handoff from Phase 2

Tier 2 in the master plan. Cleaner audio also helps WhisperX alignment
(§7.5). Stub exists in `phase2/tts.py`.

---

## 8. Unresolved Strategic Question — Path (C)

The prior session ended with a strategic proposal that the user did not
explicitly answer. Preserving it here verbatim because it changes what gets
built next:

> Pexels is structurally wrong for biography content. Should we still call
> it? Pexels indexes modern stock photography. For "Ottoman Empire Arab
> officers 1910" it has zero historical photos and returns modern
> atmospheric content because that's what it has.
>
> **Path (C) — skip the web-source rabbit hole entirely**: build the
> "Phase 1a → planner-matched book extracts" path. After Phase 1a extracts
> photos, run *one Claude call* that says "here are 20 extracted photos;
> here is the shot list; assign the best photo to each shot, leave gaps
> where no photo matches." Output is `book_manifest.json` mapping shot
> indices to filenames. The renderer reads it like a user-supplied
> manifest — no per-render vision-scoring loop, no web fetches in the
> critical path.
>
> The book's editor already curated those photos for the subject. Sonnet
> can match them to shots more reliably than any text-search API. We may
> have been chasing a sourcing problem that has a much simpler answer.

**Recommended decision**: take path (C) for the al-Askari run (the book
has period photographs of Jafar himself, Mahmud Shevket Pasha, the Arab
Revolt). Keep web sources as the secondary path for shots where no book
photo matches. This converts §7.3 from "fix three different APIs and hope"
to "ship working video using curated content, polish APIs later".

The implementation is small: a new function in `sources/book_extract.py`
that takes `(plan, photo_bank, anthropic_key)` and returns a manifest dict
in the same shape `user_upload.py` already consumes. One Sonnet call
(~$0.10), runs once per plan, not per-shot.

---

## 9. What Worked vs. What's a Compromise

| Subsystem | State | Notes |
|-----------|-------|-------|
| Parser → 8 sections | **Compromised** | Only 2 of 5 expected sections matched. Header regexes assume rigid template (§7.2) |
| Alignment | **Compromised** | Interpolated backend only. WhisperX install commented out in notebook |
| Planner (Sonnet) | ✅ Working | 91 s, ~$0.10/call, 43 well-formed shots, JSON parses cleanly |
| Plan validation | ✅ Working | Audit passes all structural checks |
| Typography rendering | ✅ Working | Amiri loaded (eventually — see §7.1) |
| Source: Pexels | ✅ Working | 3 candidates per query, every shot |
| Source: LoC / Wikimedia / IA | **Broken in practice** | 0 candidates for every query in both observed runs (§7.3) |
| Vision scoring | **Broken mid-run** | First call succeeded, rest hit `credit_balance_too_low`. Fail-open then bricks ranking (§7.4) |
| Renderer | ✅ Working | 43-shot run: 1263 s, 26.6 MB, no errors, exactly 391 s output |
| Captions (ASS) | ✅ Working / ⚠ aesthetic gap | Burns cleanly. White-on-outline is a fallback from the intended cream-bar design (§7.8) |
| Mux | ✅ Working | AAC 192 kbps; `-shortest` + `-t` cap |

---

## 10. Working Configuration

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
python audit_plan.py output/al_askari_plan_v2.json \
  --script samples/al_askari_script.txt \
  --audio  output/al_askari_audio.mp3
# --script + --audio add typography-verbatim check and audio-vs-plan
# duration delta to the standard output.

# Cell 11 — render (~21 min, &-background)
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

### Required environment

| Variable | Where | Required for |
|----------|-------|--------------|
| `ANTHROPIC_API_KEY` | `.env` at repo root or `_Phase3/`, or `--anthropic-key`, or Colab Secrets | Sonnet planner, Haiku vision scoring |
| `PEXELS_API_KEY`    | same | Pexels image source (currently the only working source — §7.3) |

### Models used

| Task | Model | Configuration | Cost / 4–7 min video |
|------|-------|---------------|----------------------|
| Shot planner (one call) | `claude-sonnet-4-6` | `max_tokens=24000`, streaming | ~$0.10 |
| Image relevance scorer | `claude-haiku-4-5-20251001` (vision) | ~150 max_tokens, **always resize image to ≤ 800 px wide** | ~$0.50 (for ~100 candidates) |
| Forced alignment | WhisperX `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` | Currently disabled; interpolation used | $0 either way |

### `--verbose` and the log-size trap

`--verbose` previously enabled DEBUG-level logging on the `anthropic` and
`httpx` loggers, which dumped full base64 image payloads (~400 KB per
vision call) into the log. A 69-shot run produced 200+ MB log files that
got truncated mid-base64 and lost the actual crash traceback.

Staged fix from prior session: keep verbose for `phase3.*` loggers, but
suppress DEBUG on `anthropic` and `httpx`. Verify this has shipped before
diagnosing any future crash from logs.

---

## 11. Recommended Session Order

By leverage, not difficulty:

1. **Restore Anthropic credits** before any further benchmarking. Without
   them planner and scorer both degrade silently (§7.4).
2. **Decide on path (C)** — the unresolved strategic question (§8). If
   yes, the next ~80 lines of code (`sources/book_extract.py`'s
   one-shot Sonnet match) probably unlocks more visual quality than fixing
   §7.3 ever will.
3. **Fix the source query strategy** (§7.3) — only if path (C) isn't
   sufficient or as a parallel improvement.
4. **Patch the vision fail-open policy** (§7.4). Even with credits, the
   policy should demote unscored candidates *only when scored ones exist*.
5. **Fix the section parser** (§7.2). Currently 5 logical sections
   collapse to 2.
6. **Decide on Whisper/X for alignment** (§7.5). Until ElevenLabs lands,
   interpolation is good enough.
7. **Restore intended caption styling** (§7.8) — convert white-outline to
   cream-bar via FFmpeg drawbox layer.
8. **Pillow typography placeholder cards** (§7.7) — converts the "TBD"
   look into a design feature.
9. ~~**Amiri discovery on system paths** (§7.1)~~ — **fixed in this drop**.
10. **Tighten shot duration distribution** (§7.6) — change prompt target
    to 5.0–6.5 s.

---

## 12. Things Not To Touch (or touch with care)

- **The plan/render split.** Two CLIs, two responsibilities. Mixing them
  was the original mistake; the split is what made every iteration in §6
  diagnosable from a JSON file.
- **`_validate_plan` invariants.** Renderer assumes them. Loosen one →
  break the concat pass or the caption layer.
- **The merge-adjacent-duplicates pass in `_validate_plan`.** Without it,
  long pull quotes get split into identical halves with separate caption
  windows. The pass is what makes long holds feel like single takes.
- **Arabic rendering uses `libraqm` when available, fallback to
  `arabic_reshaper` + `python-bidi`.** Confirmed working on Pillow 12.2.0.
  Don't add a third path. Don't use FFmpeg `drawtext` for any Arabic — it
  has no bidi shaping.
- **800 px image-resize before vision scoring** (`vision.py:~117`).
  Larger → API 400. Known constraint.
- **Stream-copy concat in `_concat_clips`.** Works only because every
  shot clip uses identical encoder settings. Changing one shot's encoder
  profile silently breaks the concat — fall back to filter_complex concat
  if you need per-shot variations.
- **`fail-open` in `VisionScorer.score`.** Don't flip it to fail-closed —
  that drops *all* candidates the moment Anthropic has a 5 s blip.
  Instead, demote unscored candidates only when scored ones exist (§7.4).
- **Title cards force the template by visual type, not by Sonnet's
  `typography_template` hint.** Sonnet hedges with `chapter_heading`;
  trust the `visual` field.
- **`--character-name` in Latin, not Arabic** — for LoC/Wikimedia/IA
  search compatibility. The book title can stay Arabic.
- **Captions skip typography shots** (`TYPOGRAPHY_VISUALS` filter in
  `_write_captions`). The typography is the caption.

---

## 13. Quick Reference

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

# Which shots ended up on Pexels (= queries that failed every other source)
grep "using fetched image from pexels" output/render.log | wc -l

# Inspect planner's raw response (saved on every plan build)
less output/planner_raw_response.txt

# List cached images after a real render
ls -la ~/.cache/lamahat/images/

# Verify Amiri discovery before rendering (no FFmpeg cost)
python -c "from phase3.typography import FONT_PATHS; print(FONT_PATHS)"

# Manifest mode — review image-shot list without hitting network
python render_plan.py --plan output/al_askari_plan_v2.json \
  --build-manifest output/required_images.txt && \
  cat output/required_images.txt
```

---

## 14. Known Environment Constraints

| Constraint | Detail |
|-----------|--------|
| Streamlit Cloud RAM | 1 GB — keep FFmpeg work in subprocesses. v2 render obeys this. |
| Python | **3.12.13** (set in Cloud Advanced settings). Don't assume 3.13/3.14. |
| Colab CPU runtime | ~21 min for a 391 s render at 1920×1080. Mostly FFmpeg + vision RTTs. |
| FFmpeg subtitle path escaping | `:` and `\` need escaping in `-vf "ass=…"`. See `_mux_final`. |
| Claude vision max image size | Always resize to ≤ 800 px wide. Larger → 400 error. |
| Pillow libraqm | Confirmed available on Pillow 12.2.0 (Colab, Streamlit Cloud). Modern raqm handles Arabic shaping natively with `direction="rtl"` — *don't* use `arabic_reshaper` on text destined for Pillow with libraqm, it actively breaks shaping by replacing Unicode characters with explicit presentation-form glyphs that bypass raqm. |
| Arabic font in ASS | `Fontname: Amiri` (`fonts-hosny-amiri` Debian package on Cloud). |
| Pexels key | Optional in the contract, mandatory in practice given §7.3. |
| Anthropic key | Required for planner AND scorer. Treat as critical-path. |
| GitHub upload size | Artefacts > 25 MB get truncated/partial in `_Phase3/output/`. `final_cut_3a.mov` is a 181 s preview of a 391 s render. Real output: `output_files.zip` from cell 13. |
| `raw.githubusercontent.com` / `media.githubusercontent.com` | Often blocked from sandboxed network policies. Use file uploads or direct paste for log/artefact handoff to a fresh Claude session. |

---

## 15. Issue tracking (this session's review-the-rough-cut feedback)

### 15.1 Color philosophy — **open**

**Goal**: knob with cinematic-warm as default; per-section variation
later.

**State**: not yet implemented.  Current renderer applies no grading;
all grading is currently baked into the source imagery and the
typography backgrounds.

**Notes for next session**: the right shape is probably a
`--grade {warm,cool,neutral,bw}` flag on `render_plan.py` that maps to
a single FFmpeg `curves`/`eq`/`colorbalance` chain applied in the final
mux.  Section-level variation is a stretch goal — the planner already
emits `section_id` per shot, so a `grade_map.json` keyed on section_id
can drop in later without re-planning.

### 15.2 Typography aesthetic — Families B and C — **open**

**Goal**: ship Families B (Netflix-doc cinematic, dark gradient) and
C (manuscript, sepia + ornament) alongside the existing Family A
(Aljazeera editorial) so the user can A/B test against a real render.

**State**: only Family A exists in `typography.py`.

**Notes for next session**: the cleanest expansion is a
`--typography-family {A,B,C}` flag and three sibling modules
(`typography_a.py`, `typography_b.py`, `typography_c.py`) that share
the same `TypographySpec` contract and template registry.  All five
templates (`title_card`, `section_mark`, `pull_quote`, `name_reveal`,
`date_stamp`) need three implementations, then a dispatcher.  Roughly
700 LOC per family; can be parallelised by sharing the canvas-grain
and font-discovery helpers.

### 15.3 Section transitions — **open**

**Goal**: faster rhythm, more audience hook at section boundaries.

**State**: prior session locked "hard cuts everywhere, section_mark
typography shot is the transition device" — that decision now reads as
too quiet.

**Notes for next session**: two complementary moves.  (a) tighten the
planner's average shot duration target from the current 5.0–6.5 s
range down to 4.0–5.0 s, especially around section boundaries; the
auto-split safety net catches anything Sonnet pushes too far.
(b) optionally introduce a single 0.3 s motion accent on section_mark
shots — a quick zoom-in or slide that signals "new chapter".

### 15.4 Captions — **open**

**Goal**: bigger title-card subtitle, less merging in body captions,
restore the intended Family A cream-bar look.

**State** (from the latest render):
- Main body captions: almost accepted (white-on-charcoal-outline).
- Title-card sub-line: too small.
- Under-line text on `name_reveal` / `date_stamp`: too small.
- Multi-line captions appear merged.

**Notes for next session**: three independent fixes.
(a) bump the title-card subtitle from `SIZES["title_sub"]` (0.030
height-fraction) to ~0.040 — but verify it doesn't push the bottom
hairline rule out of frame on long subtitles.
(b) for `name_reveal` / `date_stamp` sub-lines, lift from 0.022 to
0.028.
(c) the "merged" look is libass burning consecutive caption events
back-to-back with no inter-event gap — add 0.15 s pre-roll/post-roll
silence inside `_write_captions` so the eye sees one event end before
the next begins.

### 15.5 Online/offline asset review — **CLOSED in this drop**

**Goal**: let the user see every image candidate before rendering,
override per-shot, and pin a canonical character portrait.  Book +
main character context must inform the rubric.

**Implementation shipped**:

Three new pieces and a small Fetcher patch.

| File | Role |
|---|---|
| `phase3/sources/decisions.py` (new) | `Decisions` dataclass: load/save the dossier JSON, resolve overrides at render time |
| `prebuild_assets.py` (new, at repo root) | CLI that runs the full waterfall ahead of render, writes the dossier |
| `phase3/sources/__init__.py` (patch) | `FetcherConfig.review_dir` field; `Fetcher.__post_init__` loads the dossier; `fetch_for_shot()` checks it first |
| `render_plan.py` (patch) | `--review-dir` flag |

**Workflow** (the user-facing change):

```bash
# Step 1 — plan as before
python phase3_run.py --plan-only --script ... --audio ... \
    --save-plan output/al_askari_plan_v2.json

# Step 2 (NEW) — pre-fetch every candidate, write the dossier
python prebuild_assets.py \
    --plan          output/al_askari_plan_v2.json \
    --script        samples/al_askari_script.txt \
    --book-title    "مذكرات جعفر العسكري" \
    --character-name "Jafar al-Askari" \
    --anthropic-key "$ANTHROPIC_API_KEY" \
    --pexels-key    "$PEXELS_API_KEY" \
    --review-dir    output/review/ \
    --character-portrait /path/to/jafar.jpg

# Step 3 — user reviews output/review/
#   - Open shot_NN_*/context.txt to see the Arabic excerpt + English query
#   - Look at the downloaded candidate thumbnails
#   - Edit decisions.json to swap candidates or set overrides
#   - Drop personal images into output/review/overrides/

# Step 4 — render with the dossier
python render_plan.py \
    --plan       output/al_askari_plan_v2.json \
    --audio      output/al_askari_audio.mp3 \
    --review-dir output/review/ \
    --output     output/final_cut.mp4
```

**Per-shot resolution order** at render time, when `--review-dir` is set:

1. `decisions.shots[N].override` → user-supplied file in
   `overrides/shot_NN.jpg`.
2. `decisions.pinned_portrait` → applied to *every* `portrait` shot
   that has no explicit override.
3. `decisions.shots[N].chosen_file` → the prebuilt candidate the
   dossier marked best.
4. Live fetcher waterfall (LoC → Wikimedia → IA → Pexels) — only
   reached when the dossier said nothing for this shot.

The **pinned portrait** is the single biggest documentary-quality
win.  Instead of 5 different Pexels stock photos of "a man in
uniform" appearing at 5 different portrait moments (each captioned as
Jafar al-Askari), the same authentic image appears every time.  Set
once via `--character-portrait`, persisted in the dossier, applies
retroactively to every portrait shot.

**Book + main character** propagate as designed: `--book-title` and
`--character-name` flow into `FetcherConfig` exactly as in the prior
implementation, and they're recorded in the dossier under
`book.title` / `book.character` for reference at render time.  The
character name in particular disambiguates Pexels noise — every shot
folder's `context.txt` quotes the Arabic line being spoken, so the
user can tell whether a candidate fits the moment without watching a
rough cut.

**What the dossier folder looks like on disk**:

```
output/review/
├── decisions.json              ← The one file the user edits
├── README.txt                  ← In-folder usage guide
├── overrides/
│   ├── character.jpg           ← Pinned portrait (from --character-portrait)
│   ├── shot_05.jpg             ← Per-shot override the user dropped in
│   └── shot_38.jpg
├── shot_03_portrait/
│   ├── context.txt             ← "Arabic excerpt: قد يكونون من داخل صفوفك..."
│   ├── candidates.json
│   ├── loc_a.jpg
│   ├── wikimedia_a.jpg
│   ├── pexels_a.jpg
│   └── ...
├── shot_05_archive/
│   ...
```

**Edge cases handled**:

- `--character-portrait` argument absent → no pin, behaviour
  unchanged from before.
- `decisions.json` references an override file that's missing →
  logged warning, falls through to the next resolution step
  (typically the pin or the live fetcher).
- `--review-dir` points at a directory with no `decisions.json` →
  logged warning, renderer continues exactly as before
  (zero-friction adoption — no break for existing workflows).
- Dossier loaded from a different `review_dir` than where the file
  paths point → all paths inside the dossier are *relative*
  (`overrides/shot_05.jpg`, `shot_03_portrait/pexels_a.jpg`), so the
  dossier is portable; move the directory, change `--review-dir`, it
  still works.
- Zero candidates returned for a shot (the §7.3 reality):
  `chosen`/`chosen_file` stay empty; user can still drop an override
  or rely on the pin.

**Test coverage** (in a sandbox with no network):

| Scenario | Behaviour |
|---|---|
| Prebuild with no API keys / no network | 28 shots processed, 0 candidates each, dossier written cleanly, exit 0 |
| `--character-portrait` pointing at a real file | Copied to `overrides/character.jpg`, recorded as `pinned_portrait` in `decisions.json` |
| User edits `decisions.json`, sets `override` on shots 5 and 38 | Fetcher returns those files at fetch_for_shot |
| Portrait shots without explicit override | All four resolve to `overrides/character.jpg` via the pin |
| Portrait shot 38 with both override and applicable pin | Override wins (correct precedence) |
| Non-portrait shot without any dossier entry (shot 8 broll) | Dossier returns None, fetcher falls through to live waterfall |

**What this drop does not do** (intentionally deferred):

- **No GUI for review.** The dossier is plain JSON in a folder.  A
  Streamlit review pane is a Phase 4 concern; the JSON contract
  written now is forward-compatible with any UI built later.
- **No automated source-side query improvements.**  This issue is
  about giving the user *control over selection*, not about making
  the live sources return better candidates for historical biography
  content.  Source-quality work remains §7.3.

