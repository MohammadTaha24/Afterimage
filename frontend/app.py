import base64
import io
import time
from datetime import datetime
from pathlib import Path
from string import Template

import requests
import streamlit as st
from PIL import Image

BACKEND_PREDICT_URL = "http://127.0.0.1:9000/predict"
BACKEND_PREPARE_URL = "http://127.0.0.1:9000/model/prepare"
BACKEND_STATUS_URL = "http://127.0.0.1:9000/model/status"

LOGO_PATH = Path(__file__).parent / "logo.png"
TEMPLATES_DIR = Path(__file__).parent / "templates"

LOAD_TIMEOUT_SECONDS = 900
STATUS_POLL_INTERVAL_SECONDS = 0.5

st.set_page_config(
    page_title="Afterimage · DR Screening",
    page_icon="👁",
    layout="centered",
)

DEFAULTS = {
    "page": "upload",
    "label": None,
    "grade": None,
    "model_name": None,
    "ts": None,
    "img_bytes": None,
    "img_name": None,
    "img_size": None,
    "img_kb": None,
    "selected_model_key": None,
    "selected_model_display": "Qwen3-VL - Vision-Language Model",
    "loaded_model_key": None,
    "load_state": "idle",              # idle | requested | loading | ready | error
    "load_progress": 0,
    "load_message": "Not loaded",
    "load_started_at": None,
    "loading_model_key": None,
    "load_error": None,
    "is_inferencing": False,
    "pending_inference": False,
    "pending_file_bytes": None,
    "pending_file_name": None,
    "pending_file_type": None,
    "pending_img_size": None,
    "pending_img_kb": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def logo_tag() -> str:
    if LOGO_PATH.exists():
        data = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        return f'<img class="af-logo" src="data:image/png;base64,{data}" alt="Afterimage">'
    return '<div style="width:42px;height:42px;background:var(--navy);border-radius:10px;"></div>'


def render(template_name: str, **kwargs) -> None:
    tpl = (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
    html = Template(tpl).safe_substitute(**kwargs)
    st.markdown(html, unsafe_allow_html=True)


def error_box(msg: str) -> None:
    st.markdown(f'<div class="af-error">{msg}</div>', unsafe_allow_html=True)


def success_box(msg: str) -> None:
    st.markdown(
        (
            '<div style="background:#eefaf4;border:1px solid #ccebd9;'
            'border-left:3px solid #16a870;border-radius:10px;'
            'padding:0.95rem 1.2rem;font-size:0.88rem;'
            'font-weight:500;color:#1d6f4d;">'
            f"{msg}</div>"
        ),
        unsafe_allow_html=True,
    )


def info_box(msg: str) -> None:
    st.markdown(
        (
            '<div style="background:#f3f8fc;border:1px solid #d9e7f2;'
            'border-left:3px solid #2e5f85;border-radius:10px;'
            'padding:0.95rem 1.2rem;font-size:0.88rem;'
            'font-weight:500;color:#234761;">'
            f"{msg}</div>"
        ),
        unsafe_allow_html=True,
    )


def disabled_upload_box(msg: str) -> None:
    st.markdown(
        (
            '<div style="background:#eef2f5;border:1px solid #d7e0e8;'
            'border-radius:14px;padding:1.1rem 1.2rem;color:#7a8b99;'
            'font-size:0.9rem;box-shadow:var(--shadow-sm);">'
            f"{msg}</div>"
        ),
        unsafe_allow_html=True,
    )


def go(page: str) -> None:
    st.session_state.page = page
    st.rerun()

def show_inference_overlay(message: str = "Running inference... Please wait.") -> None:
    st.markdown(
        f"""
        <style>
        .af-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(13, 43, 69, 0.30);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .af-overlay-card {{
            width: min(420px, 92vw);
            background: #ffffff;
            border: 1px solid #dde8ef;
            border-radius: 18px;
            box-shadow: 0 10px 40px rgba(13, 43, 69, 0.18);
            padding: 1.4rem 1.4rem 1.2rem;
            text-align: center;
        }}

        .af-overlay-spinner {{
            width: 52px;
            height: 52px;
            margin: 0 auto 1rem;
            border-radius: 50%;
            border: 5px solid #d9e7f2;
            border-top-color: #0d2b45;
            animation: af-spin 0.9s linear infinite;
        }}

        .af-overlay-title {{
            font-size: 1rem;
            font-weight: 700;
            color: #0d2b45;
            margin: 0 0 0.35rem 0;
        }}

        .af-overlay-text {{
            font-size: 0.9rem;
            color: #5b7186;
            margin: 0;
            line-height: 1.5;
        }}

        @keyframes af-spin {{
            to {{ transform: rotate(360deg); }}
        }}
        </style>

        <div class="af-overlay">
            <div class="af-overlay-card">
                <div class="af-overlay-spinner"></div>
                <p class="af-overlay-title">Inference in progress</p>
                <p class="af-overlay-text">{message}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def run_pending_inference(model_key: str) -> None:
    print("run_pending_inference called", flush=True)

    if not st.session_state.pending_inference:
        print("pending_inference is False, returning", flush=True)
        return

    raw = st.session_state.pending_file_bytes
    file_name = st.session_state.pending_file_name
    file_type = st.session_state.pending_file_type
    img_size = st.session_state.pending_img_size
    img_kb = st.session_state.pending_img_kb

    if raw is None or file_name is None or file_type is None or img_size is None:
        st.session_state.is_inferencing = False
        st.session_state.pending_inference = False
        error_box("Inference could not start because the uploaded file state was incomplete.")
        print("missing pending file state", flush=True)
        return

    print(f"sending inference request for model={model_key}, file={file_name}", flush=True)

    try:
        resp = requests.post(
            BACKEND_PREDICT_URL,
            files={"file": (file_name, raw, file_type)},
            data={"model": model_key},
            timeout=180,
        )
        print(f"backend responded with status {resp.status_code}", flush=True)
    except requests.exceptions.ConnectionError:
        st.session_state.is_inferencing = False
        st.session_state.pending_inference = False
        error_box("Cannot reach the backend. Make sure uvicorn is running on port 9000.")
        print("connection error while calling backend", flush=True)
        return
    except requests.exceptions.Timeout:
        st.session_state.is_inferencing = False
        st.session_state.pending_inference = False
        error_box("Request timed out. The model took too long to respond.")
        print("inference request timed out", flush=True)
        return
    except Exception as exc:
        st.session_state.is_inferencing = False
        st.session_state.pending_inference = False
        error_box(f"Inference failed: {exc}")
        print(f"inference failed: {exc}", flush=True)
        return

    st.session_state.is_inferencing = False
    st.session_state.pending_inference = False

    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", resp.text)
        except Exception:
            msg = resp.text
        error_box(f"Error {resp.status_code} - {msg}")
        print(f"non-200 response: {resp.status_code} - {msg}", flush=True)
        return

    data = resp.json()
    print(f"inference success: {data}", flush=True)

    st.session_state.label = data["label"]
    st.session_state.grade = data["grade"]
    st.session_state.model_name = data["model"]
    st.session_state.ts = datetime.now().strftime("%d %b %Y · %H:%M")
    st.session_state.img_bytes = raw
    st.session_state.img_name = file_name
    st.session_state.img_size = img_size
    st.session_state.img_kb = img_kb

    st.session_state.pending_file_bytes = None
    st.session_state.pending_file_name = None
    st.session_state.pending_file_type = None
    st.session_state.pending_img_size = None
    st.session_state.pending_img_kb = None

    go("results")

# SEVERITY = {
#     "No DR": {
#         "short": "No DR",
#         "full": "No Diabetic Retinopathy Detected",
#         "guidance": "No DR features identified. Routine annual screening recommended.",
#         "meaning": (
#             "This result falls into the 'no apparent diabetic retinopathy' category. "
#             "In the International Clinical Diabetic Retinopathy Disease Severity Scale "
#             "(Wilkinson et al., 2003), this means no abnormalities characteristic of "
#             "diabetic retinopathy were identified on the retinal image."
#         ),
#         "flag": None,
#     },

#     "Mild": {
#         "short": "Mild",
#         "full": "Mild Non-Proliferative DR",
#         "guidance": "Early changes detected. Regular monitoring is advised.",
#         "meaning": (
#             "This result corresponds to mild non-proliferative diabetic retinopathy. "
#             "According to the International Clinical DR Severity Scale, this stage is "
#             "defined by the presence of microaneurysms, which are among the earliest "
#             "signs of diabetic retinal microvascular damage."
#         ),
#         "flag": (
#             "Images in this category are commonly associated with early retinal vascular "
#             "changes, particularly small microaneurysms. The classification indicates the "
#             "image pattern is more consistent with early diabetic retinopathy than with a "
#             "normal retinal image."
#         ),
#     },

#     "Moderate": {
#         "short": "Moderate",
#         "full": "Moderate Non-Proliferative DR",
#         "guidance": "Follow-up with an eye specialist is recommended.",
#         "meaning": (
#             "This result corresponds to moderate non-proliferative diabetic retinopathy. "
#             "This stage indicates retinal abnormalities beyond microaneurysms alone, but "
#             "not yet meeting the criteria for severe disease."
#         ),
#         "flag": (
#             "Images in this category are commonly associated with a greater extent of "
#             "retinal microvascular abnormalities than early-stage disease. This suggests "
#             "progression beyond mild retinopathy."
#         ),
#     },

#     "Severe": {
#         "short": "Severe",
#         "full": "Severe Non-Proliferative DR",
#         "guidance": "Urgent ophthalmology referral is recommended.",
#         "meaning": (
#             "This result corresponds to severe non-proliferative diabetic retinopathy. "
#             "In the International Clinical DR Severity Scale, this stage reflects extensive "
#             "retinal abnormalities and a high risk of progression to proliferative disease."
#         ),
#         "flag": (
#             "Images in this category are commonly associated with widespread retinal "
#             "abnormalities, such as extensive hemorrhages or vascular changes, indicating "
#             "significant retinal involvement."
#         ),
#     },

#     "Proliferative DR": {
#         "short": "Prolif. DR",
#         "full": "Proliferative Diabetic Retinopathy",
#         "guidance": "Immediate specialist review is required.",
#         "meaning": (
#             "This result corresponds to proliferative diabetic retinopathy. "
#             "This stage is defined by neovascularization and/or retinal hemorrhage, "
#             "reflecting advanced disease with a high risk to vision."
#         ),
#         "flag": (
#             "Images in this category are commonly associated with advanced retinal changes, "
#             "including abnormal blood vessel growth and severe vascular damage, which are "
#             "indicative of proliferative disease."
#         ),
#     },
# }

SEVERITY = {
    "No DR": {
        "sev_class": "sev-none",
        "short": "No DR",
        "full": "No Diabetic Retinopathy Detected",
        "guidance": "No DR features identified. Routine annual screening recommended.",
        "meaning": (
            "This result falls into the 'no apparent diabetic retinopathy' category. "
            "In the International Clinical Diabetic Retinopathy Disease Severity Scale "
            "(Wilkinson et al., 2003), this means no abnormalities characteristic of "
            "diabetic retinopathy were identified on the retinal image."
        ),
        "flag": None,
    },
    "Mild": {
        "sev_class": "sev-mild",
        "short": "Mild",
        "full": "Mild Non-Proliferative DR",
        "guidance": "Early changes detected. Regular monitoring is advised.",
        "meaning": (
            "This result corresponds to mild non-proliferative diabetic retinopathy. "
            "According to the International Clinical DR Severity Scale, this stage is "
            "defined by the presence of microaneurysms, which are among the earliest "
            "signs of diabetic retinal microvascular damage."
        ),
        "flag": (
            "Images in this category are commonly associated with early retinal vascular "
            "changes, particularly small microaneurysms. The classification indicates the "
            "image pattern is more consistent with early diabetic retinopathy than with a "
            "normal retinal image."
        ),
    },
    "Moderate": {
        "sev_class": "sev-mod",
        "short": "Moderate",
        "full": "Moderate Non-Proliferative DR",
        "guidance": "Follow-up with an eye specialist is recommended.",
        "meaning": (
            "This result corresponds to moderate non-proliferative diabetic retinopathy. "
            "This stage indicates retinal abnormalities beyond microaneurysms alone, but "
            "not yet meeting the criteria for severe disease."
        ),
        "flag": (
            "Images in this category are commonly associated with a greater extent of "
            "retinal microvascular abnormalities than early-stage disease. This suggests "
            "progression beyond mild retinopathy."
        ),
    },
    "Severe": {
        "sev_class": "sev-sev",
        "short": "Severe",
        "full": "Severe Non-Proliferative DR",
        "guidance": "Urgent ophthalmology referral is recommended.",
        "meaning": (
            "This result corresponds to severe non-proliferative diabetic retinopathy. "
            "In the International Clinical DR Severity Scale, this stage reflects extensive "
            "retinal abnormalities and a high risk of progression to proliferative disease."
        ),
        "flag": (
            "Images in this category are commonly associated with widespread retinal "
            "abnormalities, such as extensive hemorrhages or vascular changes, indicating "
            "significant retinal involvement."
        ),
    },
    "Proliferative DR": {
        "sev_class": "sev-prolif",
        "short": "Prolif. DR",
        "full": "Proliferative Diabetic Retinopathy",
        "guidance": "Immediate specialist review is required.",
        "meaning": (
            "This result corresponds to proliferative diabetic retinopathy. "
            "This stage is defined by neovascularization and/or retinal hemorrhage, "
            "reflecting advanced disease with a high risk to vision."
        ),
        "flag": (
            "Images in this category are commonly associated with advanced retinal changes, "
            "including abnormal blood vessel growth and severe vascular damage, which are "
            "indicative of proliferative disease."
        ),
    },
}
MODEL_OPTIONS = {
    "Qwen3-VL - GenAI Vision-Language Model": "qwen3vl",
    "RetiZero - Contrastive Vision-Language Model": "retizero",
}


def reset_loading_state() -> None:
    st.session_state.load_state = "idle"
    st.session_state.load_progress = 0
    st.session_state.load_message = "Not loaded"
    st.session_state.load_started_at = None
    st.session_state.loading_model_key = None
    st.session_state.load_error = None


def request_model_load(model_key: str) -> bool:
    try:
        resp = requests.post(
            BACKEND_PREPARE_URL,
            data={"model": model_key},
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as exc:
        st.session_state.load_state = "error"
        st.session_state.load_error = str(exc)
        st.session_state.load_message = f"Failed to start model loading: {exc}"
        return False

    st.session_state.load_state = "requested"
    st.session_state.load_progress = 1
    st.session_state.load_message = "Load request accepted"
    st.session_state.load_started_at = time.time()
    st.session_state.loading_model_key = model_key
    st.session_state.load_error = None

    if st.session_state.loaded_model_key != model_key:
        st.session_state.loaded_model_key = None

    return True


def refresh_model_status() -> None:
    loading_model_key = st.session_state.loading_model_key
    if not loading_model_key:
        return

    started_at = st.session_state.load_started_at
    if started_at is not None and (time.time() - started_at) > LOAD_TIMEOUT_SECONDS:
        st.session_state.load_state = "error"
        st.session_state.load_error = "Timed out while waiting for the model to finish loading."
        st.session_state.load_message = st.session_state.load_error
        return

    try:
        resp = requests.get(BACKEND_STATUS_URL, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        st.session_state.load_state = "error"
        st.session_state.load_error = f"Could not read model status from backend: {exc}"
        st.session_state.load_message = st.session_state.load_error
        return

    models = payload.get("models", {})
    model_status = models.get(loading_model_key)
    if not model_status:
        st.session_state.load_state = "error"
        st.session_state.load_error = f"No status found for model '{loading_model_key}'."
        st.session_state.load_message = st.session_state.load_error
        return

    state = model_status.get("state", "loading")
    progress = int(model_status.get("progress", 0))
    message = model_status.get("message", "Loading...")
    last_error = model_status.get("last_error")

    st.session_state.load_state = state
    st.session_state.load_progress = progress
    st.session_state.load_message = message
    st.session_state.load_error = last_error

    if state == "ready":
        st.session_state.loaded_model_key = loading_model_key
    elif state == "error":
        st.session_state.loaded_model_key = None


def maybe_autorefresh_loading() -> None:
    if st.session_state.load_state in {"requested", "loading"}:
        time.sleep(STATUS_POLL_INTERVAL_SECONDS)
        st.rerun()


def show_loading_widgets(current_model_key: str) -> None:
    is_current_target = st.session_state.loading_model_key == current_model_key
    should_show = (
        is_current_target
        and st.session_state.load_state in {"requested", "loading", "ready", "error"}
    )

    if not should_show:
        return

    progress = int(st.session_state.load_progress)
    message = st.session_state.load_message or "Loading..."
    state = st.session_state.load_state

    st.progress(progress, text=message)
    st.markdown(
        (
            "<div style='font-size:0.85rem;color:var(--text-muted);margin-top:0.35rem;'>"
            f"<strong>Status:</strong> {state} | "
            f"<strong>Progress:</strong> {progress}%"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if state == "ready":
        success_box("Selected model is loaded and ready for analysis.")
    elif state == "error":
        error_box(st.session_state.load_error or message or "Model failed to load.")
    else:
        info_box("Model loading is in progress. Please wait until it reaches ready state.")


def page_upload():
    if st.session_state.load_state in {"requested", "loading"}:
        refresh_model_status()

    render("upload.html", logo_tag=logo_tag())

    if st.session_state.is_inferencing:
        show_inference_overlay("The selected model is analysing the retinal image.")

        if st.session_state.pending_inference and st.session_state.selected_model_key:
            run_pending_inference(st.session_state.selected_model_key)
            return

    st.markdown('<div class="af-section">Select Model</div>', unsafe_allow_html=True)

    model_options = list(MODEL_OPTIONS.keys())

    selected_display = st.session_state.get("selected_model_display")
    if not selected_display or selected_display not in model_options:
        selected_display = model_options[0]
        st.session_state.selected_model_display = selected_display

    try:
        selected_index = model_options.index(selected_display)
    except ValueError:
        selected_index = 0
        st.session_state.selected_model_display = model_options[0]

    model_display = st.selectbox(
        label="model_select",
        options=model_options,
        index=selected_index,
        label_visibility="collapsed",
        key="model_selectbox",
    )

    st.session_state.selected_model_display = model_display
    model_key = MODEL_OPTIONS[model_display]
    st.session_state.selected_model_key = model_key

    descriptions = {
        "retizero": "A fine-tuned contrastive vision-language model trained for DR grading. Preprocessing is applied before inference.",
        "qwen3vl": "A fine-tuned GenAI Vision-Language Model with LoRA adapters. Preprocessing is applied before inference.",
    }
    st.markdown(
        f'<p style="font-size:0.78rem;color:var(--text-muted);margin:0.5rem 0 0;">{descriptions[model_key]}</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.loaded_model_key == model_key and st.session_state.load_state == "ready":
        st.markdown(
            '<p style="font-size:0.78rem;color:#1d6f4d;margin:0.45rem 0 0;font-weight:600;">Currently loaded</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="af-section">Load Selected Model</div>', unsafe_allow_html=True)

    loading_other_model = (
        st.session_state.load_state in {"requested", "loading"}
        and st.session_state.loading_model_key is not None
        and st.session_state.loading_model_key != model_key
    )

    confirm_disabled = loading_other_model

    if st.button("Confirm and Load Model", use_container_width=True, disabled=confirm_disabled):
        reset_loading_state()
        started = request_model_load(model_key)
        if started:
            refresh_model_status()
            st.rerun()

    if loading_other_model:
        info_box("Another model is currently loading. Wait for it to finish or fail before switching.")

    show_loading_widgets(model_key)

    if (
        st.session_state.loaded_model_key
        and st.session_state.loaded_model_key != model_key
        and st.session_state.load_state not in {"requested", "loading"}
    ):
        info_box("A different model is loaded. Press Confirm and Load Model for the selected one.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="af-section">Upload Image</div>', unsafe_allow_html=True)

    model_ready_for_upload = st.session_state.loaded_model_key == model_key

    if not model_ready_for_upload:
        disabled_upload_box("Upload is disabled until the selected model is loaded and ready.")
        maybe_autorefresh_loading()
        return

    uploaded = st.file_uploader(
        label="fundus_upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if not uploaded:
        maybe_autorefresh_loading()
        return

    image = Image.open(uploaded)
    w, h = image.size

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

    st.markdown("<br>", unsafe_allow_html=True)

    
    # disabled = st.session_state.loaded_model_key != model_key or st.session_state.is_inferencing

    # if st.button("Run Analysis", use_container_width=True, disabled=disabled):
    #     uploaded.seek(0)
    #     raw = uploaded.read()

    #     st.session_state.is_inferencing = True
    #     show_inference_overlay("The selected model is analysing the retinal image.")

    #     try:
    #         resp = requests.post(
    #             BACKEND_PREDICT_URL,
    #             files={"file": (uploaded.name, raw, uploaded.type)},
    #             data={"model": model_key},
    #             timeout=180,
    #         )
    #     except requests.exceptions.ConnectionError:
    #         st.session_state.is_inferencing = False
    #         error_box("Cannot reach the backend. Make sure uvicorn is running on port 9000.")
    #         maybe_autorefresh_loading()
    #         return
    #     except requests.exceptions.Timeout:
    #         st.session_state.is_inferencing = False
    #         error_box("Request timed out. The model took too long to respond.")
    #         maybe_autorefresh_loading()
    #         return
    #     except Exception as exc:
    #         st.session_state.is_inferencing = False
    #         error_box(f"Inference failed: {exc}")
    #         maybe_autorefresh_loading()
    #         return

    #     st.session_state.is_inferencing = False

    #     if resp.status_code != 200:
    #         try:
    #             msg = resp.json().get("error", resp.text)
    #         except Exception:
    #             msg = resp.text
    #         error_box(f"Error {resp.status_code} - {msg}")
    #         maybe_autorefresh_loading()
    #         return

    #     data = resp.json()

    #     st.session_state.label = data["label"]
    #     st.session_state.grade = data["grade"]
    #     st.session_state.model_name = data["model"]
    #     st.session_state.ts = datetime.now().strftime("%d %b %Y · %H:%M")
    #     st.session_state.img_bytes = raw
    #     st.session_state.img_name = uploaded.name
    #     st.session_state.img_size = (w, h)
    #     st.session_state.img_kb = round(len(raw) / 1024, 1)
    #     go("results")

    # maybe_autorefresh_loading()

    disabled = st.session_state.loaded_model_key != model_key or st.session_state.is_inferencing

    if st.button("Run Analysis", use_container_width=True, disabled=disabled):
        uploaded.seek(0)
        raw = uploaded.read()

        st.session_state.pending_file_bytes = raw
        st.session_state.pending_file_name = uploaded.name
        st.session_state.pending_file_type = uploaded.type
        st.session_state.pending_img_size = (w, h)
        st.session_state.pending_img_kb = round(len(raw) / 1024, 1)

        st.session_state.is_inferencing = True
        st.session_state.pending_inference = True
        st.rerun()

    maybe_autorefresh_loading()


# def page_results():
#     label = st.session_state.label
#     ts = st.session_state.ts

#     # sev_class, short, full, guidance, grade_label = SEVERITY.get(
#     #     label,
#     #     ("sev-none", label, label, "", "–"),
#     # )


#     entry = SEVERITY.get(label)

#     short = entry["short"]
#     full = entry["full"]
#     guidance = entry["guidance"]
#     meaning = entry["meaning"]
#     flag = entry["flag"]

#     # render(
#     #     "results.html",
#     #     logo_tag=logo_tag(),
#     #     sev_class=sev_class,
#     #     short=short,
#     #     full=full,
#     #     guidance=guidance,
#     #     grade=grade_label,
#     #     ts=ts,
#     #     model_name=st.session_state.model_name,
#     # )

#     render(
#         "results.html",
#         logo_tag=logo_tag(),
#         sev_class=sev_class,
#         short=short,
#         full=full,
#         guidance=guidance,
#         meaning=meaning,
#         flag_block=(
#             f"""
#             <div class="af-flag">
#                 <p class="af-explanation-title">Why this may have been flagged</p>
#                 <p class="af-explanation-text">{flag}</p>
#             </div>
#             """ if flag else ""
#         ),
#         grade=grade_label,
#         ts=ts,
#         model_name=st.session_state.model_name,
#     )

#     if st.session_state.img_bytes:
#         name = st.session_state.img_name
#         w, h = st.session_state.img_size
#         kb = st.session_state.img_kb
#         fmt = name.rsplit(".", 1)[-1].upper() if "." in name else "–"

#         st.markdown("<br>", unsafe_allow_html=True)
#         st.markdown('<div class="af-section">Analysed Image</div>', unsafe_allow_html=True)

#         st.markdown(
#             f"""<div class="af-image-card">
#                 <div class="af-image-card-header">
#                     <span class="af-image-card-title">{name}</span>
#                     <span class="af-image-card-meta">Analysed {ts}</span>
#                 </div>""",
#             unsafe_allow_html=True,
#         )
#         st.image(Image.open(io.BytesIO(st.session_state.img_bytes)), use_container_width=True)
#         st.markdown(
#             f"""    <div class="af-image-card-footer">
#                     <span><strong>Resolution</strong> {w} × {h} px</span>
#                     <span><strong>Format</strong> {fmt}</span>
#                     <span><strong>Size</strong> {kb} KB</span>
#                 </div>
#             </div>""",
#             unsafe_allow_html=True,
#         )

#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown('<div class="af-btn-ghost">', unsafe_allow_html=True)
#     if st.button("← Analyse Another Image", use_container_width=True):
#         keep_loaded = {
#             "loaded_model_key": st.session_state.loaded_model_key,
#             "load_state": st.session_state.load_state,
#             "load_progress": st.session_state.load_progress,
#             "load_message": st.session_state.load_message,
#             "load_started_at": st.session_state.load_started_at,
#             "loading_model_key": st.session_state.loading_model_key,
#             "load_error": st.session_state.load_error,
#             "selected_model_key": st.session_state.selected_model_key,
#             "selected_model_display": st.session_state.selected_model_display,
#         }
#         for k, v in DEFAULTS.items():
#             st.session_state[k] = v
#         for k, v in keep_loaded.items():
#             st.session_state[k] = v
#         go("upload")
#     st.markdown("</div>", unsafe_allow_html=True)


def page_results():
    label = st.session_state.label
    ts = st.session_state.ts

    entry = SEVERITY.get(label, {
        "sev_class": "sev-none",
        "short": label,
        "full": label,
        "guidance": "",
        "meaning": "",
        "flag": None,
    })

    sev_class = entry["sev_class"]
    short = entry["short"]
    full = entry["full"]
    guidance = entry["guidance"]
    meaning = entry["meaning"]
    flag = entry["flag"]

    grade_label = f"Grade {st.session_state.grade}" if st.session_state.grade is not None else "-"

    flag_block = (
        '<div class="af-flag">'
        '<p class="af-explanation-title">Why this may have been flagged</p>'
        f'<p class="af-explanation-text">{flag}</p>'
        '</div>'
        if flag else ""
    )

    render(
        "results.html",
        logo_tag=logo_tag(),
        sev_class=sev_class,
        short=short,
        full=full,
        guidance=guidance,
        meaning=meaning,
        flag_block=flag_block,
        grade=grade_label,
        ts=ts,
        model_name=st.session_state.model_name,
    )

    if st.session_state.img_bytes:
        name = st.session_state.img_name
        w, h = st.session_state.img_size
        kb = st.session_state.img_kb
        fmt = name.rsplit(".", 1)[-1].upper() if "." in name else "-"

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
            f"""<div class="af-image-card-footer">
<span><strong>Resolution</strong> {w} x {h} px</span>
<span><strong>Format</strong> {fmt}</span>
<span><strong>Size</strong> {kb} KB</span>
</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="af-btn-ghost">', unsafe_allow_html=True)

    if st.button("<- Analyse Another Image", use_container_width=True):
        keep_loaded = {
            "loaded_model_key": st.session_state.loaded_model_key,
            "load_state": st.session_state.load_state,
            "load_progress": st.session_state.load_progress,
            "load_message": st.session_state.load_message,
            "load_started_at": st.session_state.load_started_at,
            "loading_model_key": st.session_state.loading_model_key,
            "load_error": st.session_state.load_error,
            "selected_model_key": st.session_state.selected_model_key,
            "selected_model_display": st.session_state.selected_model_display,
        }

        for k, v in DEFAULTS.items():
            st.session_state[k] = v

        for k, v in keep_loaded.items():
            st.session_state[k] = v

        go("upload")

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.page == "results" and st.session_state.label:
    page_results()
else:
    page_upload()