# Problem: Phase 1 OCR Misses Lines at Page Edges

**Repo:** `abdoljh/Bk2Video` · **Branch:** `claude/add-claude-documentation-08SqJ`

## What is broken
The Phase 1 pipeline (`phase1/` directory) consistently drops lines at the **top and bottom** of scanned Arabic pages. On the reference page (`output/ph1/page_5.pdf`, page 1 of the preface to *Al-Askari Memoirs*), two lines are always missing from the raw OCR output:
- **Top line** (attribution header): `نجدة فتحى صفوة`
- **Bottom line** (last body line): `الكاتبة. ثم يجري عليها تصحيحات؛ ويدخل عليها إضافات» قصصه مرة ثأنية`

## Reference that gets it right
The notebook `output/ph1/pdf_ocr_na3.ipynb` (and its text output `output/ph1/optimized_approach.txt`) captures both lines correctly using: `pdf2image` (poppler, 300 DPI) → numpy array → `pytesseract.image_to_string(arr, lang="ara", config="--psm 3")`.

## What has been tried (without success)
1. Switching from PIL Image + 30px white border to a bare numpy array (removes DPI metadata from temp file)
2. Adding `--dpi 300` flag explicitly to the Tesseract config string (`config=f"--psm 3 --dpi {self.dpi}"`)
3. Wiring `ocr_dpi=300` from `Phase1Config` through `pipeline.py` into `OCREngine.__init__`

All three changes are committed and live on both `main` and `claude/add-claude-documentation-08SqJ`. Local tests on this machine pass (both lines appear), but the user reports the problem persists when running the application.

## Current `_tesseract_page` code (`phase1/core/ocr_engine.py`)
```python
def _tesseract_page(self, image_bytes: bytes) -> str:
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    try:
        return self._reader.image_to_string(arr, lang="ara", config=f"--psm 3 --dpi {self.dpi}")
    except Exception as exc:
        logger.warning("Tesseract OCR failed on page: %s", exc)
        return ""
```

## What needs investigation next
The gap between "works locally" and "fails in the app" suggests the problem may be **upstream of `_tesseract_page`** — specifically in `ingestor.py` or the image rendering path. Key suspects:
- Whether `pdf2image` / `poppler-utils` is actually being used (vs. silent PyMuPDF fallback)
- Whether the image passed as `image_bytes` has already had its edges cropped before reaching Tesseract
- Whether the `PageStitcher` or normalizer is stripping the top/bottom lines post-OCR
- Tesseract version on Streamlit Cloud vs. local (`tesseract --version`)

## Test file
- **Input:** `output/ph1/page_5.pdf` (single page, 93 KB, scanned Arabic)
- **Expected raw output:** starts with `نجدة فتحى صفوة`, ends with `الكاتبة. ثم يجري عليها تصحيحات؛`
- **Actual raw output:** both lines absent — see `output/ph1/page_5_phase1_raw.txt` (old broken version) for contrast
- **Fixed output (local test):** `output/ph1_fixed/page_5_fixed_phase1_raw.txt` — both lines present
