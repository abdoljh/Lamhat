# Phase 1 — Text Extraction & Intelligent Summarization

## Goal

Convert an Arabic PDF book into a **700–800 word video script** (plain + diacritized),
production-ready for TTS and video generation in subsequent phases.
Target duration: **4–5 minutes** at ~140 words per minute (MSA broadcast rate).

---

## Pipeline Overview

```
PDF
 │
 ▼
[Step 1]  EXTRACTION
 │
 ▼
[Step 2]  NORMALIZATION & CLEANING
 │
 ▼
[Step 3]  SEMANTIC CHUNKING
 │
 ▼
[Step 4]  HIERARCHICAL SUMMARIZATION
 │
 ▼
[Step 5]  SCRIPTWRITER AGENT
 │
 ▼
[Step 6]  EDITOR / SCORING AGENT
 │
 ▼
[Step 7]  DIACRITIZATION
 │
 ▼
OUTPUTS
```

---

## Step 1 — Extraction

| Condition | Engine |
|---|---|
| Scanned page (default) | Tesseract OCR (`ara` tessdata) |
| Clearly digital-born page | PyMuPDF text extraction |

- Auto-detection per page; OCR is the default path.
- Manual override available: `auto` / `digital` / `ocr` mode selector in UI.
- Post-extraction cleaning:
  - Strip running page headers (book title + page number at top of each page).
  - Strip footnotes (lines starting `(n)` or `*`, blocks below separator lines).
  - Remove standalone page numbers.

**Output:** `clean_text.txt` — logical-order Unicode Arabic, one paragraph per line.

---

## Step 2 — Normalization

- NFC Unicode normalization.
- Unify Arabic letter variants (Heh Doachashmee → standard Heh, Farsi Yeh → Arabic Yeh).
- Fix lam-alef inversions (digital PDF font artefacts).
- Remove zero-width characters, excessive whitespace.
- **No diacritization at this stage** — applied only to the final script (Step 7).

---

## Step 3 — Semantic Chunking

- Target chunk size: **1,500 tokens**, overlap: **200 tokens**.
- Chapter/section heading detection to group chunks logically.
- Each chunk stores: position index, estimated page range, detected heading.

### Chunking & Merging Policy (by book size)

| Book size | Approx. chunks | Strategy |
|---|---|---|
| < 50 pages (~10k words) | ≤ 10 | Pass all chunks directly to Step 4 |
| 50–200 pages (~40k words) | 10–40 | Reader per chunk → Consolidator → Scriptwriter |
| 200–500 pages (~100k words) | 40–100 | Reader per chunk → Consolidator (grouped by chapter) → Scriptwriter |
| 500+ pages | 100+ | Two-level: chapter summaries → book summary → Scriptwriter |

---

## Step 4 — Hierarchical Summarization

Cost-effective: small LLM calls per chunk, one consolidation call.

### Pass 1 — "Reader" Agent (per chunk)
- **Model:** `claude-haiku-4-5` (cheapest, fast).
- **Prompt (Arabic):**
  ```
  استخرج من هذا المقطع 3-5 أفكار رئيسية بالعربية الفصحى في شكل نقاط مختصرة.
  ```
- **Output:** ~50-token bullet list per chunk.

### Pass 2 — "Consolidator"
- **Model:** `claude-haiku-4-5`.
- Merges all chunk notes into a structured book outline (~2,000–3,000 tokens).
- Identifies: main thesis, key arguments, narrative arc, memorable anecdotes.
- **Output:** structured outline regardless of book length.

---

## Step 5 — Scriptwriter Agent

- **Model:** `claude-sonnet-4-6`.
- **Input:** book outline + title + author + genre tag.
- **Prompt (Arabic):**
  ```
  أنت كاتب سيناريو محترف متخصص في المحتوى الثقافي العربي.
  بناءً على الملخص التالي لكتاب "[TITLE]" من نوع "[GENRE]"،
  اكتب سكريبت بالعربية الفصحى لفيديو مدته 4-5 دقائق (700-800 كلمة) يشتمل على:
  1. خطاف افتتاحي مشوّق (الجملة الأولى تجذب الانتباه فوراً)
  2. ثلاث نقاط محورية من الكتاب مع أمثلة أو لحظات بارزة
  3. خاتمة تدفع المستمع للتفكير أو القراءة
  النبرة: [TONE based on genre].
  لا تذكر عبارة "في هذا الفيديو" أو ما شابهها.
  ```
- **Output:** raw script (~700–800 words).

---

## Step 6 — Editor / Scoring Agent

- **Model:** `claude-haiku-4-5` (cost-effective for scoring).
- Evaluates the script on a rubric (0–10 each):

| Criterion | Description |
|---|---|
| Hook strength | First 50 words grab attention immediately |
| Structure | Hook + 3 takeaways + conclusion clearly present |
| Pacing | Word count 700–800; avg sentence ≤ 25 words |
| Clarity | No unexplained jargon; accessible to general audience |
| Tone match | Consistent with book genre |

- **Threshold:** ≥ 35 / 50 → pass; else regenerate (max 2 retries).
- On failure: Editor returns specific feedback → Scriptwriter revises.
- **Output:** `script.txt` + `script_metadata.json` (scores + word count + retries used).

---

## Step 7 — Diacritization

- Applied **only** to the final polished script (not to raw OCR text).
- Engine: **Mishkal** (local, no API cost, sufficient for clean MSA text).
- Upgrade path: Farasa (QCRI) when Java runtime is available.
- **Output:** `script_diacritized.txt` — ready for TTS pipeline (Phase 2).

---

## Final Outputs

| File | Description |
|---|---|
| `clean_text.txt` | Full extracted + normalized Arabic text |
| `script.txt` | 700–800 word plain Arabic video script |
| `script_diacritized.txt` | Diacritized script for TTS |
| `script_metadata.json` | Scores, word count, chunk stats, warnings, retries |

---

## Cost Strategy

- **Reader + Editor agents:** `claude-haiku-4-5` — ~20× cheaper than Sonnet.
- **Scriptwriter:** `claude-sonnet-4-6` — one call only, after consolidation.
- For a 300-page book (~60k words, ~40 chunks):
  - Reader: 40 × ~500 tokens in + ~100 tokens out ≈ negligible cost.
  - Consolidator: ~1 call, ~4k tokens ≈ minimal.
  - Scriptwriter: ~1 call, ~3k in + ~800 out ≈ main cost.
  - Editor: ~1-2 calls, ~1k tokens each ≈ minimal.
- **Estimated total per book: < $0.05** at current API pricing.

---

## Known Issues / Future Work

- Sample0: 0-chunk bug — text lost between normalization and chunking (fix in progress).
- Sample7: custom font encoding — unrecoverable by Tesseract; flag and skip.
- Sample5: very poor scan quality — flag as low-confidence, warn user.
- Farasa diacritizer: blocked by Java dependency on Streamlit Cloud; revisit in Phase 2.
- Cloud Vision API: higher OCR accuracy for degraded scans; optional upgrade path.
