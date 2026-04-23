# Arabic Book Brief Engine — Phase 1

> Extraction & Pre-processing for automated Arabic book summaries.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bk2video.streamlit.app)

---

## Deploy to Streamlit Community Cloud

1. **Fork / push this repo to GitHub.**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app**.
3. Set:
   - **Repository:** `your-username/your-repo`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **Advanced settings** → paste your secrets (see below).
5. Click **Deploy**.

### Secrets (Advanced Settings)

```toml
FARASA_API_KEY = ""        # optional — public Farasa endpoint works without one
# ANTHROPIC_API_KEY = ""   # reserved for Phase 2
```

---

## Repository Structure

```
your-repo/                          ← GitHub repo root = Community Cloud working dir
│
├── streamlit_app.py                ← Entrypoint (Community Cloud runs this)
├── requirements.txt                ← All pip dependencies
│
├── .streamlit/
│   ├── config.toml                 ← Theme + server config (committed)
│   └── secrets.toml.template       ← Template only — real secrets.toml is gitignored
│
├── .gitignore                      ← Excludes secrets.toml, __pycache__, output/
│
└── phase1/                         ← Python package (importable from streamlit_app.py)
    ├── __init__.py
    ├── pipeline.py                 ← Phase1Pipeline orchestrator
    └── core/
        ├── __init__.py
        ├── ingestor.py             ← PDF type detection + PyMuPDF extraction
        ├── ocr_engine.py           ← EasyOCR / Tesseract for scanned pages
        ├── normalizer.py           ← BiDi + arabic-reshaper + cleaning
        ├── diacritizer.py          ← Farasa API + Mishkal fallback
        ├── chunker.py              ← Chapter-aware semantic chunking
        └── output_writer.py        ← JSON + plain text serialisation
```

---

## Local Development

```bash
git clone https://github.com/your-username/your-repo
cd your-repo

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create local secrets (gitignored)
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your keys

streamlit run streamlit_app.py
```

---

## Using Phase 1 as a Library

```python
from phase1 import Phase1Pipeline, Phase1Config

result = Phase1Pipeline(Phase1Config(diacritize=True)).run("book.pdf")
print(result.pdf_type, len(result.chunks))
```

---

## What Phase 1 Produces

| Output | Format | Contents |
|--------|--------|----------|
| `*_phase1.json` | JSON | Full structured chunks with metadata |
| `*_phase1.txt`  | Plain text | Human-readable audit trail |

Both files are available as downloads directly in the app after processing.

---

## Phase Roadmap

- [x] **Phase 1** — Extraction & Pre-processing *(this repo)*
- [ ] **Phase 2** — Multi-agent LangGraph script generation
- [ ] **Phase 3** — TTS audio synthesis
- [ ] **Phase 4** — Video assembly
