"""
Afterimage — Frontend
Loads HTML templates from templates/ and renders Streamlit widgets between them.

Run:
    streamlit run frontend/app.py
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from string import Template

import requests
import streamlit as st
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL   = "http://127.0.0.1:9000/predict"
LOGO_PATH     = Path(__file__).parent / "logo.png"
TEMPLATES_DIR = Path(__file__).parent / "templates"

st.set_page_config(
    page_title="Afterimage · DR Screening",
    page_icon="👁",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "page":        "upload",   # "upload" | "results"
    "label":       None,
    "grade":       None,
    "model_name":  None,
    "ts":          None,
    "img_bytes":   None,
    "img_name":    None,
    "img_size":    None,
    "img_kb":      None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def logo_tag() -> str:
    if LOGO_PATH.exists():
        data = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        return f'<img class="af-logo" src="data:image/png;base64,{data}" alt="Afterimage">'
    return '<div style="width:42px;height:42px;background:var(--navy);border-radius:10px;"></div>'

def render(template_name: str, **kwargs) -> None:
    """Load a template, substitute variables, and render as HTML."""
    tpl  = (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
    html = Template(tpl).safe_substitute(**kwargs)
    st.markdown(html, unsafe_allow_html=True)

def error_box(msg: str) -> None:
    st.markdown(f'<div class="af-error">{msg}</div>', unsafe_allow_html=True)

def go(page: str) -> None:
    st.session_state.page = page
    st.rerun()

# ── Severity lookup ───────────────────────────────────────────────────────────
SEVERITY = {
    "No DR":            ("sev-none",   "No DR",       "No Diabetic Retinopathy Detected",         "No DR features identified. Routine annual screening recommended.",           "Grade 0"),
    "Mild":             ("sev-mild",   "Mild",        "Mild Non-Proliferative DR",                "Microaneurysms only. Annual monitoring advised.",                             "Grade 1"),
    "Moderate":         ("sev-mod",    "Moderate",    "Moderate Non-Proliferative DR",            "More than mild NPDR. Ophthalmology referral within 6 months.",                "Grade 2"),
    "Severe":           ("sev-sev",    "Severe",      "Severe Non-Proliferative DR",              "High progression risk. Urgent ophthalmology referral required.",               "Grade 3"),
    "Proliferative DR": ("sev-prolif", "Prolif. DR",  "Proliferative Diabetic Retinopathy",       "Neovascularisation present. Immediate specialist referral required.",          "Grade 4"),
}

MODEL_OPTIONS = {
    "RetiZero — CNN classifier":           "retizero",
    "Qwen3-VL — Vision-Language Model":    "qwen3vl",
}

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def page_upload():
    # Nav + disclaimer + title
    render("upload.html", logo_tag=logo_tag())

    # ── Model selection ────────────────────────────────────────────────────────
    st.markdown('<div class="af-section">Select Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="af-model-card">', unsafe_allow_html=True)

    model_display = st.selectbox(
        label="model_select",
        options=list(MODEL_OPTIONS.keys()),
        label_visibility="collapsed",
    )
    model_key = MODEL_OPTIONS[model_display]

    descriptions = {
        "retizero": "A fine-tuned CNN trained on the EyePACS dataset. Fast and lightweight.",
        "qwen3vl":  "A fine-tuned Vision-Language Model with LoRA adapters. More descriptive output.",
    }
    st.markdown(f'<p style="font-size:0.78rem;color:var(--text-muted);margin:0.5rem 0 0;">{descriptions[model_key]}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Image upload ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="af-section">Upload Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="fundus_upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if not uploaded:
        return

    # ── Preview ────────────────────────────────────────────────────────────────
    image = Image.open(uploaded)
    w, h  = image.size

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="af-section">Preview</div>', unsafe_allow_html=True)

    st.markdown(
        f"""<div class="af-preview-card">
            <div class="af-preview-header">
                <span class="af-preview-header-title">Fundus image</span>
                <span class="af-preview-header-meta">{uploaded.name}</span>
            </div>""",
        unsafe_allow_html=True,
    )
    st.image(image, use_container_width=True)
    st.markdown(
        f"""    <div class="af-preview-footer">
                <span><strong>Resolution</strong> {w} × {h} px</span>
                <span><strong>Format</strong> {uploaded.type.split("/")[-1].upper()}</span>
                <span><strong>Size</strong> {round(uploaded.size / 1024, 1)} KB</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Run Analysis ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    if not st.button("Run Analysis →", use_container_width=True):
        return

    with st.spinner("Analysing retinal image…"):
        uploaded.seek(0)
        raw = uploaded.read()
        try:
            resp = requests.post(
                BACKEND_URL,
                files={"file": (uploaded.name, raw, uploaded.type)},
                data={"model": model_key},
                timeout=180,
            )
        except requests.exceptions.ConnectionError:
            error_box("Cannot reach the backend. Is <code>uvicorn backend.main:app --port 9000</code> running?")
            return
        except requests.exceptions.Timeout:
            error_box("Request timed out. The model is taking too long — try again.")
            return

    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", resp.text)
        except Exception:
            msg = resp.text
        error_box(f"Error {resp.status_code} — {msg}")
        return

    data = resp.json()

    # Store results in session and switch page
    st.session_state.label      = data["label"]
    st.session_state.grade      = data["grade"]
    st.session_state.model_name = data["model"]
    st.session_state.ts         = datetime.now().strftime("%d %b %Y · %H:%M")
    st.session_state.img_bytes  = raw
    st.session_state.img_name   = uploaded.name
    st.session_state.img_size   = (w, h)
    st.session_state.img_kb     = round(len(raw) / 1024, 1)
    go("results")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def page_results():
    label = st.session_state.label
    ts    = st.session_state.ts

    sev_class, short, full, guidance, grade_label = SEVERITY.get(
        label, ("sev-none", label, label, "", "–")
    )

    # Nav + disclaimer + title + result card (all from template)
    render(
        "results.html",
        logo_tag   = logo_tag(),
        sev_class  = sev_class,
        short      = short,
        full       = full,
        guidance   = guidance,
        grade      = grade_label,
        ts         = ts,
        model_name = st.session_state.model_name,
    )

    # ── Analysed image ─────────────────────────────────────────────────────────
    if st.session_state.img_bytes:
        name = st.session_state.img_name
        w, h = st.session_state.img_size
        kb   = st.session_state.img_kb
        fmt  = name.rsplit(".", 1)[-1].upper() if "." in name else "–"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="af-section">Analysed Image</div>', unsafe_allow_html=True)

        st.markdown(
            f"""<div class="af-image-card">
                <div class="af-image-card-header">
                    <span class="af-image-card-title">{name}</span>
                    <span class="af-image-card-meta">Analysed {ts}</span>
                </div>""",
            unsafe_allow_html=True,
        )
        st.image(Image.open(io.BytesIO(st.session_state.img_bytes)), use_container_width=True)
        st.markdown(
            f"""    <div class="af-image-card-footer">
                    <span><strong>Resolution</strong> {w} × {h} px</span>
                    <span><strong>Format</strong> {fmt}</span>
                    <span><strong>Size</strong> {kb} KB</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Back button ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="af-btn-ghost">', unsafe_allow_html=True)
    if st.button("← Analyse Another Image", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        go("upload")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.page == "results" and st.session_state.label:
    page_results()
else:
    page_upload()
