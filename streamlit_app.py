"""
Arabic Book Brief Engine — Phase 1
Streamlit Community Cloud entrypoint.

Repository root is the working directory on Community Cloud, so:
  • This file lives at repo root  →  streamlit run streamlit_app.py
  • The phase1 package lives at  →  phase1/
  • Config lives at              →  .streamlit/config.toml
  • Secrets injected via         →  st.secrets  (never committed)
"""

import html
import json
import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st

# ── Phase 1 package is at ./phase1 relative to repo root ──────────────── #
sys.path.insert(0, str(Path(__file__).parent))
from phase1 import (  # noqa: E402
    Phase1aPipeline, Phase1bPipeline, Phase1Config, Phase1aResult,
)
from phase2 import synthesize as tts_synthesize  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────── #
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ── Page config ───────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Arabic Book Brief — Phase 1",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────── #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

.app-header {
    background: #0e0e0e; color: #f5f0e8;
    padding: 2rem 2.5rem 1.6rem; border-radius: 8px;
    margin-bottom: 2rem; position: relative; overflow: hidden;
}
.app-header::after {
    content: '📖'; position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%); font-size: 5rem; opacity: .07;
}
.app-header h1 { color: #f5f0e8 !important; margin: 0; font-size: 2rem; }
.app-header .sub { color: #b0a898; font-size: 0.85rem; margin-top: 0.4rem; }
.app-header .eyebrow {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: .18em; text-transform: uppercase;
    color: #c9a84c; margin-bottom: 0.5rem;
}
.badge {
    display: inline-block; font-family: 'DM Mono', monospace;
    font-size: 0.6rem; letter-spacing: .1em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 2px; border: 1px solid;
    margin-right: 6px; margin-top: 8px;
}
.b-gold { border-color: #c9a84c; color: #c9a84c; }
.b-teal { border-color: #4aadad; color: #4aadad; }
.b-rust { border-color: #d97452; color: #d97452; }
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 120px; background: white;
    border: 1px solid #e0dbd0; border-top: 3px solid #c9a84c;
    border-radius: 4px; padding: 1rem 1.2rem;
    box-shadow: 3px 3px 0 #e8dfcc;
}
.metric-card .val {
    font-family: 'Playfair Display', serif; font-size: 2rem;
    font-weight: 700; color: #0e0e0e; line-height: 1;
}
.metric-card .lbl {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: .12em; text-transform: uppercase;
    color: #7a7060; margin-top: 4px;
}
.metric-card.teal  { border-top-color: #1e6b6b; }
.metric-card.rust  { border-top-color: #b94f2a; }
.metric-card.purple{ border-top-color: #7c5cbf; }
.chunk-card {
    background: #fefcf8; border: 1px solid #e0dbd0;
    border-left: 4px solid #c9a84c; border-radius: 0 4px 4px 0;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    direction: rtl; text-align: right;
    font-size: 0.9rem; line-height: 1.8;
    color: #1a1a1a;
}
.chunk-meta {
    font-family: 'DM Mono', monospace; font-size: 0.6rem;
    letter-spacing: .1em; text-transform: uppercase;
    color: #7a7060; direction: ltr; text-align: left; margin-bottom: 0.4rem;
}
.chunk-card.scanned { border-left-color: #1e6b6b; }
.warn-card {
    background: #fff7ec; border-left: 4px solid #c9a84c;
    border-radius: 0 4px 4px 0; padding: 0.8rem 1rem;
    margin: 0.5rem 0; font-size: 0.85rem; color: #5a3d00;
}
.step-log {
    font-family: 'DM Mono', monospace; font-size: 0.75rem;
    background: #0e0e0e; color: #c8c0b0; padding: 1rem 1.2rem;
    border-radius: 4px; line-height: 1.8;
    max-height: 220px; overflow-y: auto;
}
.step-log .done  { color: #4aadad; }
.step-log .active{ color: #f0d98a; }
section[data-testid="stSidebar"] { background: #0e0e0e !important; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] span { color: #c8c0b0 !important; }
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f0d98a !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────── #
st.markdown("""
<div class="app-header">
  <div class="eyebrow">Arabic Book Brief Engine · Phase 1a + 1b</div>
  <h1>Extraction, Normalisation &amp; Script</h1>
  <div class="sub">
    <b>Phase 1a</b> — Strip margins · Export page images · Kraken OCR · Normalise<br>
    <b>Phase 1b</b> — Chunk · Summarise · Generate Arabic video script
  </div>
  <div>
    <span class="badge b-gold">Header/Footer Stripping</span>
    <span class="badge b-teal">Kraken Offline OCR</span>
    <span class="badge b-rust">Semantic Chunking</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("#### Phase 1a — PDF Preprocessing")
    strip_margins = st.toggle(
        "Strip headers/footers",
        value=True,
        help=(
            "Automatically detect and remove running headers, footers, and footnote "
            "separators before exporting page images. Uses ink-density analysis."
        ),
    )
    try:
        from phase1.core.kraken_engine import _KRAKEN_AVAILABLE as _kraken_ok
    except Exception:
        _kraken_ok = False

    if _kraken_ok:
        _ocr_backend_options = {
            "Kraken (offline, Arabic model)": "kraken",
            "None (export images only)": "none",
        }
        _ocr_backend_default = 0
    else:
        _ocr_backend_options = {"None (export images only)": "none"}
        _ocr_backend_default = 0
        st.warning(
            "⚠️ Kraken OCR is not available on this Python version. "
            "Only image export mode is supported here. "
            "To enable Kraken, redeploy the app with **Python 3.12** selected "
            "in Streamlit Cloud Advanced settings, then uncomment "
            "`torch`/`lightning`/`kraken` in `requirements.txt`.",
            icon="🐍",
        )

    _ocr_backend_label = st.selectbox(
        "OCR Backend",
        list(_ocr_backend_options.keys()),
        index=_ocr_backend_default,
        help=(
            "**Kraken** — offline Arabic OCR using the OpenITI apt-20221130 model. "
            "Best quality, runs fully on Streamlit Cloud, no extra API cost.\n\n"
            "**None** — export clean page images only. Download the ZIP, run OCR "
            "externally, then upload the text to Phase 1b."
        ),
    )
    ocr_backend = _ocr_backend_options[_ocr_backend_label]
    ocr_dpi = st.slider("Scan DPI", 150, 600, 400, step=50)

    # Kraken-specific controls
    kraken_bidi      = "auto"
    kraken_threshold = 0.5
    kraken_pad       = 16
    if ocr_backend == "kraken":
        _bidi_map = {
            "Auto (let kraken decide)": "auto",
            "Force RTL": "R",
            "Force LTR": "L",
            "Off (raw display order)": "off",
        }
        kraken_bidi      = _bidi_map[st.selectbox(
            "Bidi reordering", list(_bidi_map.keys()), index=0,
            help="Controls how Kraken reorders bidirectional text. Auto is correct for Arabic.",
        )]
        kraken_threshold = st.slider(
            "Binarization threshold", 1, 99, 50,
            help="Higher = darker pixels counted as ink. 50 is a good default.",
        ) / 100.0
        kraken_pad = st.slider(
            "Line padding (px)", 0, 64, 16, step=4,
            help="Pixels of padding added around each detected text line.",
        )

    st.markdown("#### Chunking")
    max_tokens     = st.slider("Max Tokens / Chunk", 500, 3000, 1500, step=100)
    overlap_tokens = st.slider("Overlap Tokens",       0,  500,  200, step=50)

    st.markdown("#### Script Generation")
    anthropic_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.secrets.get("ANTHROPIC_API_KEY", ""),
        help="Required for script generation. Leave blank to extract text only.",
    )
    script_genre = st.selectbox(
        "Book Genre",
        ["non-fiction", "history", "biography", "novel",
         "philosophy", "science", "religion"],
        index=0,
        help="Affects the tone of the generated script.",
    )
    book_author = st.text_input(
        "Author / Editor / Translator",
        placeholder="e.g. تحقيق وتقديم نجدة فتحي صفوة",
        help="Injected verbatim into the formal book-presentation section of the script.",
    )
    book_pages = st.number_input(
        "Total Pages",
        min_value=0,
        value=0,
        step=1,
        help="Actual page count of the full book (0 = omit from script).",
    )
    book_structure = st.text_input(
        "Book Structure",
        placeholder="e.g. مقدمة و١٦ فصلاً وملاحق",
        help="Brief Arabic description of chapters / sections / appendices.",
    )
    scriptwriter_model = st.selectbox(
        "Scriptwriter model",
        ["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"],
        index=0,
        help=(
            "Model used for the creative Scriptwriter step only. "
            "Reader, Consolidator, and Editor always use Haiku.\n\n"
            "**Haiku** — default, lowest cost (~$0.001 per script).\n\n"
            "**Sonnet** — higher quality prose, ~40× more expensive (~$0.04 per script)."
        ),
    )
    diacritize_script = st.toggle(
        "Diacritise script (Mishkal)",
        value=True,
        help=(
            "Apply Mishkal diacritisation to the final script and save "
            "*_script_diacritized.txt. Turn off to skip diacritisation and "
            "save a few seconds per run."
        ),
    )

    st.markdown("---")
    st.markdown("#### 🎙 Phase 2: TTS")
    tts_backend = st.radio(
        "TTS Backend",
        ["gTTS (free)", "ElevenLabs (soon)"],
        index=0,
        help=(
            "**gTTS** — Google TTS, free, no API key. Use for development.\n\n"
            "**ElevenLabs** — Premium Arabic voices (e.g. Chaouki). Coming soon."
        ),
    )
    el_api_key  = ""
    el_voice_id = ""
    if tts_backend == "ElevenLabs (soon)":
        el_api_key  = st.text_input("ElevenLabs API Key", type="password", key="el_key")
        el_voice_id = st.text_input("Voice ID", placeholder="e.g. Chaouki voice ID", key="el_voice")

    st.markdown("---")
    st.markdown("#### 🎬 Phase 3: Visuals")
    pexels_api_key = st.text_input(
        "Pexels API Key",
        type="password",
        key="p3_pexels_key",
        help=(
            "Free API key from pexels.com/api. "
            "Used as fallback when Wikimedia has no images for a section. "
            "Leave blank to use Wikimedia only."
        ),
    )
    p3_color_grade = st.selectbox(
        "Color Grade",
        ["warm", "neutral", "cool"],
        index=0,
        key="p3_color_grade",
        help="warm — amber/gold tone · neutral — no adjustment · cool — blue tones",
    )

    st.markdown("---")
    st.markdown(
        "<span style='font-family:DM Mono,monospace;font-size:0.6rem;"
        "color:#6b6355;letter-spacing:.1em'>ARABIC BOOK BRIEF ENGINE v1.0</span>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════ #
#  Phase 1a — PDF Preprocessing & OCR                                     #
# ════════════════════════════════════════════════════════════════════════ #
st.markdown("""
<div class="app-header" style="margin-top:0">
  <div class="eyebrow">Phase 1a</div>
  <h1>PDF Preprocessing &amp; OCR</h1>
  <div class="sub">
    Strip headers/footers · Export clean page images · Kraken Arabic OCR · Arabic normalisation
  </div>
  <div>
    <span class="badge b-gold">Header/Footer Detection</span>
    <span class="badge b-teal">Kraken Offline OCR</span>
    <span class="badge b-rust">Offline Export</span>
  </div>
</div>
""", unsafe_allow_html=True)

col_up, col_info = st.columns([2, 1])
with col_up:
    uploaded = st.file_uploader("Upload Arabic PDF", type=["pdf"])
with col_info:
    st.markdown("""
    **Phase 1a produces:**
    - `*_phase1a_pages.zip` — clean page images for offline OCR
    - `*_phase1a_corrected.txt` — raw Kraken OCR text, per page
    - `*_phase1a_normalized.txt` — after Arabic normalisation
    - `*_phase1a.json` — structured page data for Phase 1b

    Use **None** backend to export images only, then upload
    OCR text to Phase 1b via *Upload User Corrected*.
    """)

if uploaded:
    if st.button("▶ Run Phase 1a", type="primary", use_container_width=True, key="p1a_run"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path   = Path(tmp_dir) / uploaded.name
            output_dir = Path(tmp_dir) / "output"
            output_dir.mkdir()
            tmp_path.write_bytes(uploaded.read())

            progress_bar = st.progress(0.0)
            status_text  = st.empty()
            log_lines: list[str] = []
            log_ph = st.empty()

            def on_progress_1a(step: str, pct: float):
                progress_bar.progress(min(pct, 1.0))
                status_text.markdown(f"**{step}**")
                cls = "done" if pct >= 1.0 else "active"
                log_lines.append(f"<span class='{cls}'>{'✓' if pct>=1.0 else '›'} {step}</span>")
                log_ph.markdown(
                    "<div class='step-log'>" + "<br>".join(log_lines) + "</div>",
                    unsafe_allow_html=True,
                )

            cfg = Phase1Config(
                strip_margins    = strip_margins,
                export_dpi       = ocr_dpi,
                ocr_backend      = ocr_backend,
                kraken_bidi      = kraken_bidi,
                kraken_threshold = kraken_threshold,
                kraken_pad       = kraken_pad,
                max_tokens=max_tokens, overlap_tokens=overlap_tokens,
                output_dir=str(output_dir),
                anthropic_api_key=anthropic_key,
                script_genre=script_genre,
                book_author=book_author,
                book_pages=int(book_pages),
                book_structure=book_structure,
                diacritize=diacritize_script,
                scriptwriter_model=scriptwriter_model,
            )

            try:
                result_a = Phase1aPipeline(config=cfg, on_progress=on_progress_1a).run(tmp_path)

                # Persist to session state before temp dir is cleaned up
                st.session_state["phase1a_result"]           = result_a
                st.session_state["phase1a_corrected_bytes"]  = result_a.corrected_txt_path.read_bytes()
                st.session_state["phase1a_corrected_name"]   = result_a.corrected_txt_path.name
                st.session_state["phase1a_normalized_bytes"] = result_a.normalized_txt_path.read_bytes()
                st.session_state["phase1a_normalized_name"]  = result_a.normalized_txt_path.name
                st.session_state["phase1a_json_bytes"]       = result_a.normalized_json_path.read_bytes()
                st.session_state["phase1a_json_name"]        = result_a.normalized_json_path.name
                if result_a.pages_zip_path and result_a.pages_zip_path.exists():
                    st.session_state["phase1a_zip_bytes"] = result_a.pages_zip_path.read_bytes()
                    st.session_state["phase1a_zip_name"]  = result_a.pages_zip_path.name
                else:
                    st.session_state.pop("phase1a_zip_bytes", None)
                    st.session_state.pop("phase1a_zip_name", None)
                st.session_state["phase1a_meta"] = {
                    "pdf_type":    result_a.pdf_type,
                    "total_pages": result_a.total_pages,
                    "elapsed_sec": result_a.elapsed_sec,
                    "warnings":    result_a.warnings,
                    "ocr_backend": ocr_backend,
                }
                status_text.success("Phase 1a complete ✓")

            except Exception as exc:
                st.error(f"Phase 1a failed: {exc}")
                logging.exception("Phase 1a error")

# ── Phase 1a results ─────────────────────────────────────────────────── #
if "phase1a_meta" in st.session_state:
    meta_a = st.session_state["phase1a_meta"]

    for w in meta_a["warnings"]:
        st.markdown(f"<div class='warn-card'>⚠ {w}</div>", unsafe_allow_html=True)

    type_colors = {"digital": "#c9a84c", "scanned": "#1e6b6b", "mixed": "#b94f2a"}
    tc = type_colors.get(meta_a["pdf_type"], "#c9a84c")
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="val">{meta_a['total_pages']}</div><div class="lbl">Pages</div>
      </div>
      <div class="metric-card" style="border-top-color:{tc}">
        <div class="val" style="font-size:1.3rem;padding-top:.3rem">{meta_a['pdf_type'].upper()}</div>
        <div class="lbl">PDF Type</div>
      </div>
      <div class="metric-card purple">
        <div class="val">{meta_a['elapsed_sec']:.1f}s</div><div class="lbl">Elapsed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📥 Phase 1a Downloads")

    # Page images ZIP is always available
    if "phase1a_zip_bytes" in st.session_state:
        st.download_button(
            "⬇ Page images ZIP (for offline OCR)",
            data=st.session_state["phase1a_zip_bytes"],
            file_name=st.session_state["phase1a_zip_name"],
            mime="application/zip",
            use_container_width=True,
            help=(
                "Header/footer-stripped page images at the chosen DPI. "
                "Use these with any external OCR tool, then upload the result "
                "to Phase 1b via 'Upload User Corrected'."
            ),
        )

    # If no OCR was run in-app, guide the user to Phase 1b upload path
    if meta_a.get("ocr_backend") == "none":
        st.info(
            "Page images exported. Download the ZIP, run OCR externally (e.g. with "
            "Kraken, Google Vision, or any other tool), then upload the resulting "
            "text file to **Phase 1b → Upload User Corrected** below."
        )
    else:
        a1, a2, a3 = st.columns(3)
        with a1:
            st.download_button(
                "⬇ OCR text (raw)", data=st.session_state["phase1a_corrected_bytes"],
                file_name=st.session_state["phase1a_corrected_name"],
                mime="text/plain", use_container_width=True,
                help="Raw Kraken OCR output before Arabic normalisation",
            )
        with a2:
            st.download_button(
                "⬇ Normalized text", data=st.session_state["phase1a_normalized_bytes"],
                file_name=st.session_state["phase1a_normalized_name"],
                mime="text/plain", use_container_width=True,
                help="After Arabic normalisation — input to Phase 1b chunking",
            )
        with a3:
            st.download_button(
                "⬇ Phase 1a JSON", data=st.session_state["phase1a_json_bytes"],
                file_name=st.session_state["phase1a_json_name"],
                mime="application/json", use_container_width=True,
                help="Structured page data — upload to Phase 1b to skip re-running OCR",
            )

    with st.expander("🔬 Compare: raw OCR vs normalised"):
        st.caption("Left = raw Kraken OCR output · Right = after Arabic normalisation")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**OCR-corrected**")
            corr_txt = st.session_state.get("phase1a_corrected_bytes", b"").decode("utf-8", errors="replace")
            st.text_area("corr", corr_txt[:4000], height=300, label_visibility="collapsed")
        with rc2:
            st.markdown("**Normalised**")
            norm_txt = st.session_state.get("phase1a_normalized_bytes", b"").decode("utf-8", errors="replace")
            st.text_area("norm", norm_txt[:4000], height=300, label_visibility="collapsed")

# ════════════════════════════════════════════════════════════════════════ #
#  Phase 1b — Chunk & Summarise                                           #
# ════════════════════════════════════════════════════════════════════════ #
st.markdown("---")
st.markdown("""
<div class="app-header" style="margin-top:1rem">
  <div class="eyebrow">Phase 1b</div>
  <h1>Chunk &amp; Summarise</h1>
  <div class="sub">Semantic chunking · Hierarchical summarisation · Arabic video script (625–850 words)</div>
  <div>
    <span class="badge b-teal">Mishkal Diacritizer</span>
    <span class="badge b-rust">Semantic Chunking</span>
  </div>
</div>
""", unsafe_allow_html=True)

p1b_source = st.radio(
    "Input source",
    ["Phase 1a session result", "Upload User Corrected (.txt)"],
    horizontal=True,
    key="p1b_source",
    help=(
        "**Session result** — use the Phase 1a output from this session (no re-upload needed).\n\n"
        "**Upload User Corrected** — upload a plain .txt file you edited manually. "
        "The entire file is treated as a single normalised page and fed directly to chunking."
    ),
)

_p1b_ready = False
_p1b_source_obj = None   # Phase1aResult or Path

if p1b_source == "Phase 1a session result":
    if "phase1a_result" not in st.session_state:
        st.info("Run Phase 1a above first, or switch to **Upload User Corrected**.")
    else:
        meta_a = st.session_state.get("phase1a_meta", {})
        st.caption(
            f"Session result: {meta_a.get('total_pages', '?')} pages · "
            f"{meta_a.get('pdf_type', '?')} · {meta_a.get('elapsed_sec', 0):.1f}s"
        )
        _p1b_source_obj = st.session_state["phase1a_result"]
        _p1b_ready = True
else:
    p1b_txt_up = st.file_uploader(
        "Upload corrected text (.txt)",
        type=["txt"],
        key="p1b_txt_up",
        help=(
            "Upload a plain UTF-8 .txt file containing the corrected Arabic text. "
            "The entire file is treated as one page — no OCR re-run needed."
        ),
    )
    if p1b_txt_up:
        txt_content = p1b_txt_up.read().decode("utf-8", errors="replace")
        _p1b_source_obj = Phase1aResult(
            source_path          = p1b_txt_up.name,
            pdf_type             = "scanned",
            total_pages          = 1,
            metadata             = {"title": Path(p1b_txt_up.name).stem},
            pages                = [
                {
                    "page_number":  1,
                    "pdf_type":     "scanned",
                    "raw_text":     txt_content,
                    "raw_text_pre": "",
                }
            ],
            corrected_txt_path   = Path(tempfile.gettempdir()) / "dummy_corrected.txt",
            normalized_txt_path  = Path(tempfile.gettempdir()) / "dummy_normalized.txt",
            normalized_json_path = Path(tempfile.gettempdir()) / "dummy_phase1a.json",
        )
        _p1b_ready = True
        st.caption(f"Loaded: {p1b_txt_up.name} — {len(txt_content.split())} words")

if _p1b_ready:
    if st.button("▶ Run Phase 1b", type="primary", use_container_width=True, key="p1b_run"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output"
            output_dir.mkdir()

            progress_bar_b = st.progress(0.0)
            status_text_b  = st.empty()
            log_lines_b: list[str] = []
            log_ph_b = st.empty()

            def on_progress_1b(step: str, pct: float):
                progress_bar_b.progress(min(pct, 1.0))
                status_text_b.markdown(f"**{step}**")
                cls = "done" if pct >= 1.0 else "active"
                log_lines_b.append(f"<span class='{cls}'>{'✓' if pct>=1.0 else '›'} {step}</span>")
                log_ph_b.markdown(
                    "<div class='step-log'>" + "<br>".join(log_lines_b) + "</div>",
                    unsafe_allow_html=True,
                )

            cfg_b = Phase1Config(
                max_tokens=max_tokens, overlap_tokens=overlap_tokens,
                output_dir=str(output_dir),
                anthropic_api_key=anthropic_key,
                script_genre=script_genre,
                book_author=book_author,
                book_pages=int(book_pages),
                book_structure=book_structure,
                diacritize=diacritize_script,
                scriptwriter_model=scriptwriter_model,
            )

            try:
                result_b = Phase1bPipeline(config=cfg_b, on_progress=on_progress_1b).run(
                    _p1b_source_obj
                )

                st.session_state["json_bytes"]    = result_b.json_path.read_bytes()
                st.session_state["txt_bytes"]     = result_b.txt_path.read_bytes()
                st.session_state["raw_txt_bytes"] = result_b.raw_txt_path.read_bytes()
                st.session_state["json_name"]     = result_b.json_path.name
                st.session_state["txt_name"]      = result_b.txt_path.name
                st.session_state["raw_txt_name"]  = result_b.raw_txt_path.name
                if result_b.script_path and result_b.script_path.exists():
                    st.session_state["script_bytes"] = result_b.script_path.read_bytes()
                    st.session_state["script_name"]  = result_b.script_path.name
                else:
                    st.session_state.pop("script_bytes", None)
                    st.session_state.pop("script_name", None)
                if result_b.script_diac_path and result_b.script_diac_path.exists():
                    st.session_state["script_diac_bytes"] = result_b.script_diac_path.read_bytes()
                    st.session_state["script_diac_name"]  = result_b.script_diac_path.name
                else:
                    st.session_state.pop("script_diac_bytes", None)
                    st.session_state.pop("script_diac_name", None)
                if result_b.script_meta_path and result_b.script_meta_path.exists():
                    st.session_state["script_meta_bytes"] = result_b.script_meta_path.read_bytes()
                    st.session_state["script_meta_name"]  = result_b.script_meta_path.name
                else:
                    st.session_state.pop("script_meta_bytes", None)
                    st.session_state.pop("script_meta_name", None)
                st.session_state["result_meta"] = {
                    "pdf_type":    result_b.pdf_type,
                    "total_pages": result_b.total_pages,
                    "elapsed_sec": result_b.elapsed_sec,
                    "warnings":    result_b.warnings,
                    "chunks": [
                        {
                            "chunk_id":   c.chunk_id,
                            "chapter":    c.chapter,
                            "page_start": c.page_start,
                            "page_end":   c.page_end,
                            "word_count": c.word_count,
                            "token_est":  c.token_est,
                            "text":       c.text,
                        }
                        for c in result_b.chunks
                    ],
                }
                status_text_b.success("Phase 1b complete ✓")

            except Exception as exc:
                st.error(f"Phase 1b failed: {exc}")
                logging.exception("Phase 1b error")

# ── Phase 1b results ─────────────────────────────────────────────────── #
if "result_meta" in st.session_state:
    meta   = st.session_state["result_meta"]
    chunks = meta["chunks"]

    st.markdown("---")
    st.markdown("### Phase 1b Results")

    for w in meta["warnings"]:
        st.markdown(f"<div class='warn-card'>⚠ {w}</div>", unsafe_allow_html=True)

    type_colors = {"digital": "#c9a84c", "scanned": "#1e6b6b", "mixed": "#b94f2a"}
    tc = type_colors.get(meta["pdf_type"], "#c9a84c")
    total_words = sum(c["word_count"] for c in chunks)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="val">{meta['total_pages']}</div><div class="lbl">Pages</div>
      </div>
      <div class="metric-card" style="border-top-color:{tc}">
        <div class="val" style="font-size:1.3rem;padding-top:.3rem">{meta['pdf_type'].upper()}</div>
        <div class="lbl">PDF Type</div>
      </div>
      <div class="metric-card teal">
        <div class="val">{len(chunks)}</div><div class="lbl">Chunks</div>
      </div>
      <div class="metric-card rust">
        <div class="val">{total_words:,}</div><div class="lbl">Words</div>
      </div>
      <div class="metric-card purple">
        <div class="val">{meta['elapsed_sec']:.1f}s</div><div class="lbl">Elapsed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📥 Downloads")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇ JSON (processed)", data=st.session_state["json_bytes"],
                           file_name=st.session_state["json_name"],
                           mime="application/json", use_container_width=True)
    with c2:
        st.download_button("⬇ Text (processed)", data=st.session_state["txt_bytes"],
                           file_name=st.session_state["txt_name"],
                           mime="text/plain", use_container_width=True)
    with c3:
        st.download_button("⬇ Text (raw extract)", data=st.session_state.get("raw_txt_bytes", b""),
                           file_name=st.session_state.get("raw_txt_name", "raw.txt"),
                           mime="text/plain", use_container_width=True,
                           help="Text straight from PyMuPDF/OCR before any normalisation")

    if "script_bytes" in st.session_state:
        st.markdown("#### 📝 Arabic Video Script")
        if "script_meta_bytes" in st.session_state:
            try:
                smeta = json.loads(st.session_state["script_meta_bytes"])
                scores  = smeta.get("scores", {})
                total   = smeta.get("total_score", 0)
                wc      = smeta.get("word_count", 0)
                retries = smeta.get("retries_used", 0)
                score_bar = " · ".join(f"{k} {v}/10" for k, v in scores.items())
                st.markdown(
                    f"<div class='metric-row'>"
                    f"<div class='metric-card'><div class='val'>{wc}</div><div class='lbl'>Words</div></div>"
                    f"<div class='metric-card teal'><div class='val'>{total}/50</div><div class='lbl'>Score</div></div>"
                    f"<div class='metric-card rust'><div class='val'>{retries}</div><div class='lbl'>Retries</div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Criteria: {score_bar}")
                feedback = smeta.get("editor_feedback", "")
                if feedback:
                    st.caption(f"Editor: {feedback}")
            except Exception:
                pass

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.download_button("⬇ Script (plain)", data=st.session_state["script_bytes"],
                               file_name=st.session_state["script_name"],
                               mime="text/plain", use_container_width=True)
        with sc2:
            st.download_button("⬇ Script (diacritized)",
                               data=st.session_state.get("script_diac_bytes", b""),
                               file_name=st.session_state.get("script_diac_name", "script_diac.txt"),
                               mime="text/plain", use_container_width=True)
        with sc3:
            st.download_button("⬇ Script metadata",
                               data=st.session_state.get("script_meta_bytes", b""),
                               file_name=st.session_state.get("script_meta_name", "script_meta.json"),
                               mime="application/json", use_container_width=True)

        with st.expander("📄 Preview script"):
            script_txt = st.session_state["script_bytes"].decode("utf-8", errors="replace")
            st.markdown(
                f"<div style='direction:rtl;text-align:right;font-size:0.95rem;"
                f"line-height:1.9;background:#fefcf8;padding:1.2rem 1.5rem;"
                f"border:1px solid #e0dbd0;border-radius:4px'>{script_txt}</div>",
                unsafe_allow_html=True,
            )
    elif anthropic_key:
        st.info("Script generation ran but produced no output — check warnings above.")

    st.markdown("#### 🔍 Chunk Preview")
    n = st.slider("Chunks to preview", 1, min(20, len(chunks)), 5)
    for c in chunks[:n]:
        border = "scanned" if meta["pdf_type"] == "scanned" else ""
        st.markdown(
            f"""<div class="chunk-card {border}">
              <div class="chunk-meta">
                chunk {c['chunk_id']:04d} · {c['chapter']}
                · pp. {c['page_start']}–{c['page_end']}
                · {c['word_count']} words · ~{c['token_est']} tokens
              </div>
              {html.escape(c['text'][:500])}{"…" if len(c['text']) > 500 else ""}
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("🔬 Compare: raw extract vs normalised"):
        st.caption("Left = straight from PyMuPDF/OCR · Right = after normalisation")
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Raw extract**")
            raw_txt = st.session_state.get("raw_txt_bytes", b"").decode("utf-8", errors="replace")
            st.text_area("raw", raw_txt[:4000], height=300, label_visibility="collapsed")
        with rc2:
            st.markdown("**Processed**")
            proc_txt = st.session_state.get("txt_bytes", b"").decode("utf-8", errors="replace")
            st.text_area("proc", proc_txt[:4000], height=300, label_visibility="collapsed")

    with st.expander("🔎 Inspect raw JSON"):
        st.json(json.loads(st.session_state["json_bytes"]))

# ── Phase 2: Audio Generation ─────────────────────────────────────────── #
st.markdown("---")
st.markdown("""
<div class="app-header" style="margin-top:1rem">
  <div class="eyebrow">Arabic Book Brief Engine · Phase 2</div>
  <h1>Audio Generation</h1>
  <div class="sub">Convert your Arabic video script into spoken audio</div>
  <div>
    <span class="badge b-teal">gTTS Free</span>
    <span class="badge b-gold">ElevenLabs (soon)</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Script source
p2_source = st.radio(
    "Script source",
    ["Phase 1 output", "Upload .txt file"],
    horizontal=True,
    key="p2_source",
    help=(
        "**Phase 1 output** — use a script generated by the pipeline above.\n\n"
        "**Upload .txt file** — load an existing book_script.txt or "
        "book_script_diacritized.txt from disk."
    ),
)

p2_text  = ""
p2_label = ""

if p2_source == "Phase 1 output":
    has_plain = "script_bytes"      in st.session_state
    has_diac  = "script_diac_bytes" in st.session_state
    if not has_plain and not has_diac:
        st.info(
            "No Phase 1 script in session. Run Phase 1 with an Anthropic API key, "
            "or switch to **Upload .txt file** to load an existing script."
        )
    else:
        variants = []
        if has_plain:
            variants.append("Plain (recommended)")
        if has_diac:
            variants.append("Diacritized")
        p2_variant = st.radio(
            "Script variant", variants, horizontal=True, key="p2_variant",
            help=(
                "**Plain** — recommended for both gTTS and ElevenLabs. "
                "The script is pre-cleaned (no markdown, TTS pause markers on headings). "
                "ElevenLabs applies its own diacritization internally.\n\n"
                "**Diacritized** — Mishkal harakat added. "
                "May conflict with ElevenLabs prosody on sentence-final consonants."
            ),
        )
        if p2_variant.startswith("Diacritized"):
            p2_text  = st.session_state["script_diac_bytes"].decode("utf-8", errors="replace")
            p2_label = "diacritized"
        else:
            p2_text  = st.session_state["script_bytes"].decode("utf-8", errors="replace")
            p2_label = "plain"

else:  # Upload .txt file
    p2_upload = st.file_uploader(
        "Upload script (.txt)",
        type=["txt"],
        key="p2_upload",
        help="Upload book_script.txt or book_script_diacritized.txt",
    )
    if p2_upload:
        p2_text  = p2_upload.read().decode("utf-8", errors="replace")
        p2_label = Path(p2_upload.name).stem

if p2_text:
    with st.expander("Preview script"):
        st.markdown(
            f"<div style='direction:rtl;text-align:right;font-size:0.95rem;"
            f"line-height:1.9;background:#fefcf8;color:#1a1a1a;padding:1.2rem 1.5rem;"
            f"border:1px solid #e0dbd0;border-radius:4px'>{html.escape(p2_text)}</div>",
            unsafe_allow_html=True,
        )

    tts_key = "gtts" if tts_backend == "gTTS (free)" else "elevenlabs"

    if tts_key == "elevenlabs" and (not el_api_key or not el_voice_id):
        st.warning("ElevenLabs requires both an API key and a Voice ID — fill them in the sidebar.")

    if st.button("🎙 Generate Audio", type="primary", use_container_width=True, key="p2_gen"):
        backend_label = "gTTS" if tts_key == "gtts" else "ElevenLabs"
        with st.spinner(f"Synthesizing with {backend_label}…"):
            try:
                audio = tts_synthesize(
                    p2_text,
                    backend=tts_key,
                    elevenlabs_api_key=el_api_key,
                    elevenlabs_voice_id=el_voice_id,
                )
                st.session_state["p2_audio_bytes"] = audio
                st.session_state["p2_audio_label"] = p2_label
                st.success("Audio ready ✓")
            except Exception as exc:
                st.error(f"TTS failed: {exc}")
                logging.exception("Phase 2 TTS error")

if "p2_audio_bytes" in st.session_state:
    import base64
    audio_bytes = st.session_state["p2_audio_bytes"]
    # st.audio can fail for large MP3s or non-standard MIME strings.
    # An inline <audio> element with the correct IANA type is more reliable.
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio controls style="width:100%;margin:0.5rem 0">'
        f'<source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">'
        f'</audio>',
        unsafe_allow_html=True,
    )
    dl_name = f"book_audio_{st.session_state.get('p2_audio_label', 'output')}.mp3"
    st.download_button(
        "⬇ Download MP3",
        data=audio_bytes,
        file_name=dl_name,
        mime="audio/mpeg",
        use_container_width=True,
        key="p2_dl",
    )

# ── Phase 3: Visual Generation ────────────────────────────────────────── #
st.markdown("---")
st.markdown("""
<div class="app-header" style="margin-top:1rem">
  <div class="eyebrow">Arabic Book Brief Engine · Phase 3</div>
  <h1>Final Video Assembly</h1>
  <div class="sub">Visuals · Arabic voice · Burned-in subtitles · Complete MP4</div>
  <div>
    <span class="badge b-gold">Wikimedia Commons</span>
    <span class="badge b-teal">Ken Burns Effect</span>
    <span class="badge b-rust">Pexels Fallback</span>
    <span class="badge b-teal">Arabic Subtitles</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Script source ─────────────────────────────────────────────────────── #
p3_script_src = st.radio(
    "Script source",
    ["Phase 1 output", "Upload .txt file"],
    horizontal=True,
    key="p3_script_src",
    help="Use a script already in session, or upload book_script_rev.txt / book_script.txt.",
)

p3_text = ""
if p3_script_src == "Phase 1 output":
    if "script_bytes" in st.session_state:
        p3_text = st.session_state["script_bytes"].decode("utf-8", errors="replace")
    else:
        st.info("No Phase 1 script in session — switch to **Upload .txt file**.")
else:
    p3_up = st.file_uploader("Upload script (.txt)", type=["txt"], key="p3_script_up")
    if p3_up:
        p3_text = p3_up.read().decode("utf-8", errors="replace")

# ── Audio source (for duration timing) ───────────────────────────────── #
p3_audio_bytes: bytes | None = None
if "p2_audio_bytes" in st.session_state:
    p3_audio_bytes = st.session_state["p2_audio_bytes"]
    st.caption("Using Phase 2 audio for section timing.")
else:
    p3_audio_up = st.file_uploader(
        "Upload audio (.mp3) for timing — optional",
        type=["mp3"],
        key="p3_audio_up",
        help="Lets the pipeline size each section accurately. "
             "Skip to use a character-count estimate.",
    )
    if p3_audio_up:
        p3_audio_bytes = p3_audio_up.read()

# ── Genre (passed to keyword generator) ──────────────────────────────── #
p3_genre = st.selectbox(
    "Book genre",
    ["history", "biography", "non-fiction", "philosophy",
     "science", "religion", "novel"],
    index=0,
    key="p3_genre",
    help="Affects Wikimedia search terms and colour-grade default.",
)

# ── Book context for better image search ─────────────────────────────── #
_col_title, _col_char = st.columns(2)
with _col_title:
    p3_book_title = st.text_input(
        "Book title (English)",
        key="p3_book_title",
        placeholder="e.g. Memoirs of Jafar al-Askari",
        help="Used by the keyword generator to find more relevant Wikimedia images.",
    )
with _col_char:
    p3_character_name = st.text_input(
        "Main character name (English)",
        key="p3_character_name",
        placeholder="e.g. Jafar al-Askari",
        help="Ensures a portrait photograph of the main subject is searched first.",
    )

# ── Generate button ───────────────────────────────────────────────────── #
if p3_text:
    with st.expander("Preview script sections"):
        try:
            from phase3.parser import parse_sections
            p3_secs = parse_sections(p3_text)
            for s in p3_secs:
                st.markdown(
                    f"<div style='font-family:DM Mono,monospace;font-size:0.7rem;"
                    f"color:#c9a84c;margin-top:0.6rem'>{s.section_id.upper()}</div>"
                    f"<div style='direction:rtl;text-align:right;font-size:0.85rem;"
                    f"line-height:1.7'>{s.text[:200]}{'…' if len(s.text)>200 else ''}</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            st.text(p3_text[:600])

    # ── Keyword preview (optional step before generating video) ──────── #
    with st.expander("🔍 Preview search keywords (optional — inspect before generating)"):
        st.caption(
            "Click **Generate Keywords** to see exactly what search terms Claude will "
            "use for Wikimedia and Pexels, and which Arabic key phrases will be "
            "displayed as on-screen text overlays — before committing to the full video."
        )
        if st.button("Generate Keywords", key="p3_kw_preview"):
            if not anthropic_key:
                st.warning("Anthropic API key required for keyword generation. "
                           "Add it in the sidebar.")
            else:
                with st.spinner("Calling Claude Haiku for keyword ideas…"):
                    try:
                        from phase3.parser import parse_sections as _parse
                        from phase3.keywords import generate_keywords as _gen_kw

                        _kw_sections = _parse(p3_text)
                        _kw_results  = _gen_kw(
                            _kw_sections,
                            p3_genre,
                            anthropic_key,
                            book_title=p3_book_title,
                            character_name=p3_character_name,
                        )
                        st.session_state["p3_kw_results"] = _kw_results
                    except Exception as _e:
                        st.error(f"Keyword generation failed: {_e}")

        if "p3_kw_results" in st.session_state:
            for _kw in st.session_state["p3_kw_results"]:
                st.markdown(
                    f"<div style='font-family:DM Mono,monospace;font-size:0.7rem;"
                    f"color:#c9a84c;margin-top:0.8rem;text-transform:uppercase'>"
                    f"{_kw.section_id}</div>",
                    unsafe_allow_html=True,
                )
                _c1, _c2 = st.columns(2)
                with _c1:
                    st.markdown("**Wikimedia searches**")
                    for _q in _kw.wikimedia:
                        st.markdown(f"- `{_q}`")
                with _c2:
                    st.markdown("**Pexels searches**")
                    for _q in _kw.pexels:
                        st.markdown(f"- `{_q}`")
                if _kw.key_phrases:
                    st.markdown("**Key phrase overlays**")
                    for _phrase in _kw.key_phrases:
                        st.markdown(
                            f"<div style='direction:rtl;text-align:right;"
                            f"background:#fefcf8;border-left:3px solid #c9a84c;"
                            f"padding:0.4rem 0.8rem;margin:0.3rem 0;"
                            f"font-size:0.9rem'>{_phrase}</div>",
                            unsafe_allow_html=True,
                        )

    p3_add_subs = st.checkbox(
        "Burn Arabic subtitles into video",
        value=True,
        key="p3_add_subs",
        help="Overlays the script text as timed Arabic captions (white text, black outline).",
    )

    if st.button("▶ Generate Final Video", type="primary",
                 use_container_width=True, key="p3_gen"):
        import tempfile as _tmp
        _out_dir = Path(_tmp.mkdtemp(prefix="bk2v_out_"))
        _out_mp4 = _out_dir / "final_video.mp4"
        _thumb   = _out_dir / "thumb.jpg"

        _p3_progress = st.progress(0.0)
        _p3_status   = st.empty()
        _p3_log      = st.empty()
        _p3_lines: list[str] = []

        def _p3_cb(label: str, frac: float) -> None:
            _p3_progress.progress(min(frac, 1.0))
            _p3_status.markdown(f"**{label}**")
            cls = "done" if frac >= 1.0 else "active"
            _p3_lines.append(
                f"<span class='{cls}'>{'✓' if frac>=1.0 else '›'} {label}</span>"
            )
            _p3_log.markdown(
                "<div class='step-log'>" + "<br>".join(_p3_lines[-12:]) + "</div>",
                unsafe_allow_html=True,
            )

        try:
            from phase3 import generate_background_video
            from phase3.compositor import extract_thumbnail

            generate_background_video(
                script_text=p3_text,
                output_path=_out_mp4,
                audio_bytes=p3_audio_bytes,
                anthropic_api_key=anthropic_key,
                pexels_api_key=pexels_api_key,
                genre=p3_genre,
                color_grade=p3_color_grade,
                images_per_section=3,
                book_title=p3_book_title,
                character_name=p3_character_name,
                add_subtitles=p3_add_subs,
                on_progress=_p3_cb,
            )

            st.session_state["p3_video_bytes"] = _out_mp4.read_bytes()

            # Extract a preview thumbnail from the middle of the video
            extract_thumbnail(_out_mp4, _thumb, time=5.0)
            if _thumb.exists():
                st.session_state["p3_thumb_bytes"] = _thumb.read_bytes()

            _p3_status.success("Final video ready ✓")

        except Exception as _exc:
            st.error(f"Video generation failed: {_exc}")
            logging.exception("Phase 3 error")
        finally:
            import shutil as _sh
            _sh.rmtree(_out_dir, ignore_errors=True)

if "p3_video_bytes" in st.session_state:
    # Thumbnail preview (video itself is too large for in-browser base64 embed)
    if "p3_thumb_bytes" in st.session_state:
        st.image(
            st.session_state["p3_thumb_bytes"],
            caption="First frame preview",
            use_container_width=True,
        )
    p3_sz_mb = len(st.session_state["p3_video_bytes"]) / 1_048_576
    _has_audio = "p2_audio_bytes" in st.session_state or (
        "p3_audio_up" in st.session_state and st.session_state.get("p3_audio_up")
    )
    _video_desc = "720p"
    if _has_audio:
        _video_desc += " · Arabic voice"
    if st.session_state.get("p3_add_subs", True):
        _video_desc += " · Arabic subtitles"
    st.caption(f"Final video · {p3_sz_mb:.1f} MB · {_video_desc}")
    st.download_button(
        "⬇ Download Final Video (.mp4)",
        data=st.session_state["p3_video_bytes"],
        file_name="final_video.mp4",
        mime="video/mp4",
        use_container_width=True,
        key="p3_dl",
    )
