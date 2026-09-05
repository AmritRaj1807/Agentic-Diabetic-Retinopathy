from __future__ import annotations

import base64
import hashlib
import html
import time
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from dashboard.services import CONFUSION_MATRIX_PATH
from dashboard.services import DR_LABELS
from dashboard.services import MODEL_PERFORMANCE
from dashboard.services import DashboardInputError
from dashboard.services import analyze_image_quality
from dashboard.services import get_checkpoint_status
from dashboard.services import get_demo_images
from dashboard.services import image_metadata
from dashboard.services import load_dashboard_bundle
from dashboard.services import open_image_from_bytes
from dashboard.services import predict_demo_image
from dashboard.services import predict_uploaded_image


st.set_page_config(
    page_title="RetinaAI | Diabetic Retinopathy Screening",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_ITEMS = [
    "Home",
    "Analyze Image",
    "Demo Images",
    "Model Performance",
    "About",
    "FAQ",
]


@st.cache_resource(show_spinner="Loading trained model checkpoint...")
def get_cached_bundle():
    return load_dashboard_bundle()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #eef5fb;
            --panel: #ffffff;
            --ink: #0f1f33;
            --muted: #617085;
            --line: #d7e4f2;
            --blue: #126bff;
            --blue-dark: #0756df;
            --blue-soft: #e7f1ff;
            --teal: #18a999;
            --teal-soft: #e4f8f5;
            --green: #2fb463;
            --green-soft: #e7f8ee;
            --amber-soft: #fff6e6;
            --danger: #b91c1c;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(135deg, #f8fbff 0%, var(--bg) 48%, #f3faf9 100%);
            color: var(--ink);
        }

        .main .block-container {
            max-width: 1240px;
            padding: 1.15rem 1.5rem 2.25rem;
        }

        h1, h2, h3, p {
            letter-spacing: 0;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #edf6ff 100%);
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding: 1.15rem 0.85rem;
        }

        [data-testid="stSidebar"] label p {
            font-size: 0.86rem;
            font-weight: 680;
        }

        div[role="radiogroup"] label {
            border-radius: 8px;
            padding: 0.52rem 0.66rem;
            margin-bottom: 0.2rem;
            border: 1px solid transparent;
        }

        div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        div[role="radiogroup"] label,
        div[role="radiogroup"] label p {
            color: #33465c !important;
            font-weight: 640;
        }

        div[role="radiogroup"] label:hover {
            background: #f0f6ff;
        }

        div[role="radiogroup"] label:has(input:checked),
        div[role="radiogroup"] label:has(input:checked) p {
            color: var(--blue-dark) !important;
            font-weight: 760;
        }

        div[role="radiogroup"] label:has(input:checked) {
            background: var(--blue-soft);
            border-color: #cfe3ff;
            box-shadow: inset 3px 0 0 var(--blue);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.62rem;
            margin-bottom: 1.1rem;
        }

        .brand-mark {
            width: 2.2rem;
            height: 2.2rem;
            border-radius: 999px;
            display: grid;
            place-items: center;
            background: radial-gradient(circle, #ffffff 31%, #126bff 33%, #126bff 58%, #dff0ff 60%);
            box-shadow: 0 8px 18px rgba(18, 107, 255, 0.22);
        }

        .brand-mark::after {
            content: "";
            width: 0.62rem;
            height: 0.62rem;
            border-radius: 999px;
            background: #102033;
        }

        .brand-name {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 820;
            line-height: 1;
        }

        .brand-sub {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 620;
            margin-top: 0.16rem;
        }

        .rail-note {
            color: #456077;
            background: linear-gradient(180deg, #eef7ff 0%, #e9f5f2 100%);
            border: 1px solid #d5e8fb;
            border-radius: 8px;
            padding: 0.65rem;
            font-size: 0.72rem;
            line-height: 1.35;
            margin-top: 2rem;
        }

        .hero {
            min-height: 500px;
            border: 1px solid #0e3158;
            border-radius: 8px;
            overflow: hidden;
            padding: 3rem 2.2rem 1.55rem;
            background:
                radial-gradient(circle at 74% 42%, rgba(26, 169, 153, 0.28), transparent 23rem),
                linear-gradient(90deg, rgba(1, 15, 31, 0.98) 0%, rgba(2, 33, 62, 0.92) 48%, rgba(2, 22, 43, 0.74) 100%);
            color: white;
            box-shadow: 0 24px 70px rgba(10, 36, 68, 0.17);
        }

        .hero h1 {
            font-size: clamp(2.2rem, 5vw, 4rem);
            line-height: 1.03;
            max-width: 690px;
            margin: 0 0 1rem;
            font-weight: 840;
        }

        .hero-lead {
            color: #b9dfff;
            font-size: 1.34rem;
            margin-bottom: 0.78rem;
        }

        .hero-copy {
            color: #93c8f4;
            font-size: 1.02rem;
            max-width: 560px;
        }

        .hero-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1.5rem;
        }

        .hero-actions a {
            border-radius: 7px;
            padding: 0.86rem 1.2rem;
            text-decoration: none;
            font-weight: 780;
            color: white;
            border: 1px solid #1f7cff;
        }

        .hero-actions .primary {
            background: linear-gradient(180deg, #2e83ff 0%, #126bff 100%);
        }

        .hero-actions .secondary {
            background: rgba(3, 26, 51, 0.58);
            border-color: #3aa0ff;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.08fr) minmax(300px, 1fr);
            gap: 1.45rem;
            align-items: center;
        }

        .hero-image-wrap {
            display: grid;
            place-items: center;
            min-height: 405px;
        }

        .hero-image-wrap img {
            width: min(430px, 100%);
            aspect-ratio: 1 / 1;
            border-radius: 999px;
            object-fit: cover;
            box-shadow: 0 0 0 1px rgba(87, 172, 255, 0.25), 0 0 58px rgba(15, 107, 255, 0.42);
        }

        .feature-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.35rem;
        }

        .feature-tile {
            border-left: 1px solid rgba(116, 180, 234, 0.18);
            padding: 0.75rem;
            color: #d9ecff;
            min-height: 6.75rem;
        }

        .feature-icon {
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 999px;
            display: grid;
            place-items: center;
            color: #80c5ff;
            border: 1px solid rgba(128, 197, 255, 0.48);
            margin-bottom: 0.52rem;
            font-weight: 820;
        }

        .feature-title {
            font-weight: 790;
            margin-bottom: 0.2rem;
        }

        .feature-copy {
            color: #86c7ff;
            font-size: 0.82rem;
            line-height: 1.35;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line) !important;
            border-radius: 8px !important;
            background: rgba(255, 255, 255, 0.94) !important;
            box-shadow: 0 18px 48px rgba(31, 70, 112, 0.075);
        }

        .page-title {
            font-size: 1.82rem;
            color: var(--ink);
            font-weight: 820;
            line-height: 1.08;
            margin-bottom: 0.25rem;
        }

        .page-subtitle {
            color: var(--muted);
            font-size: 0.96rem;
            margin-bottom: 1.1rem;
        }

        .stepper {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.4rem;
            margin-bottom: 1.35rem;
            border-bottom: 1px solid #e2ebf5;
            padding-bottom: 0.95rem;
        }

        .step {
            text-align: center;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
        }

        .step-dot {
            width: 2rem;
            height: 2rem;
            display: grid;
            place-items: center;
            border-radius: 999px;
            margin: 0 auto 0.32rem;
            background: #d9e5f2;
            color: #516276;
            font-weight: 820;
        }

        .step.done .step-dot {
            color: white;
            background: var(--green);
        }

        .step.active .step-dot {
            color: white;
            background: var(--blue);
        }

        .upload-panel {
            border: 1px dashed #9cc7f7;
            border-radius: 8px;
            background: #fbfdff;
            padding: 0.9rem 1rem 0.35rem;
            margin-bottom: 0.9rem;
        }

        .kv-table {
            display: grid;
            gap: 0.58rem;
            margin: 0.6rem 0 1rem;
        }

        .kv-row {
            display: grid;
            grid-template-columns: 8.5rem 1fr;
            gap: 0.75rem;
            align-items: center;
            font-size: 0.86rem;
        }

        .kv-key {
            color: var(--muted);
            font-weight: 700;
        }

        .kv-value {
            color: var(--ink);
            font-weight: 650;
        }

        .quality-row {
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid #edf2f7;
            font-size: 0.86rem;
        }

        .quality-row:last-child {
            border-bottom: 0;
        }

        .status-pill {
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.74rem;
            font-weight: 780;
            white-space: nowrap;
        }

        .status-good {
            background: var(--green-soft);
            color: #157044;
        }

        .status-review {
            background: var(--amber-soft);
            color: #8a5207;
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%);
            padding: 0.92rem;
            min-height: 5.2rem;
            box-shadow: 0 8px 20px rgba(31, 70, 112, 0.045);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 720;
            margin-bottom: 0.18rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 820;
        }

        .result-card {
            border: 1px solid #bce8cb;
            border-radius: 8px;
            padding: 1rem;
            background: linear-gradient(180deg, #f0fff6 0%, var(--green-soft) 100%);
            min-height: 8rem;
        }

        .result-grade {
            color: var(--danger);
            font-size: 2.3rem;
            font-weight: 860;
            line-height: 1.05;
            margin: 0.16rem 0 0.35rem;
        }

        .important-card {
            border: 1px solid #f0d7a6;
            border-radius: 8px;
            padding: 1rem;
            background: var(--amber-soft);
            min-height: 8rem;
            color: #683a0c;
        }

        .severity-list {
            display: grid;
            gap: 0.42rem;
        }

        .severity-row {
            display: grid;
            grid-template-columns: 1.3rem 2rem 1fr;
            align-items: center;
            gap: 0.5rem;
            border-radius: 7px;
            padding: 0.47rem 0.6rem;
            color: var(--ink);
            border: 1px solid transparent;
        }

        .severity-row.active {
            background: #d9ebff;
            color: #073b80;
            font-weight: 820;
            border-color: #bfdcff;
        }

        .severity-pin {
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 999px;
            background: var(--blue);
        }

        .score-row {
            display: grid;
            grid-template-columns: minmax(9rem, 1fr) minmax(7rem, 2fr) 3.5rem;
            gap: 0.75rem;
            align-items: center;
            margin: 0.7rem 0;
            color: var(--ink);
            font-size: 0.9rem;
        }

        .score-track {
            height: 0.92rem;
            border-radius: 999px;
            background: #e6edf5;
            overflow: hidden;
        }

        .score-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #8db8ff 0%, #126bff 100%);
        }

        .score-fill.muted {
            background: linear-gradient(90deg, #bdc9d5 0%, #8da0b3 100%);
        }

        .info-box {
            border: 1px solid #bfdefd;
            border-radius: 8px;
            background: #e8f4ff;
            padding: 0.85rem;
            color: #244967;
            font-size: 0.86rem;
        }

        .future-box {
            border: 1px dashed #c7d6e5;
            border-radius: 8px;
            background: #fbfdff;
            padding: 1rem;
            color: var(--muted);
        }

        .decision-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin-bottom: 0.85rem;
        }

        .decision-item {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #f8fbff;
            padding: 0.85rem;
        }

        .stButton > button {
            border-radius: 7px;
            border: 1px solid #0f68ed;
            background: var(--blue);
            color: white;
            min-height: 2.65rem;
            font-weight: 760;
        }

        .stButton > button:hover {
            border-color: var(--blue-dark);
            background: var(--blue-dark);
            color: white;
        }

        [data-testid="stFileUploader"] section {
            border: 1px dashed #9cc7f7;
            border-radius: 8px;
            background: #fbfdff;
            min-height: 9rem;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem;
        }

        @media (max-width: 820px) {
            .feature-strip, .stepper, .hero-grid, .decision-row, .kv-row {
                grid-template-columns: 1fr;
            }

            .hero {
                padding: 2rem 1.1rem 1.1rem;
                min-height: auto;
            }

            .score-row {
                grid-template-columns: 1fr;
                gap: 0.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand_html() -> str:
    return """
    <div class="brand">
        <div class="brand-mark"></div>
        <div>
            <div class="brand-name">RetinaAI</div>
            <div class="brand-sub">Diabetic Retinopathy Screening</div>
        </div>
    </div>
    """


def source_token(source_kind: str, name: str, data: bytes | None = None) -> str:
    digest = hashlib.sha256(data or name.encode("utf-8")).hexdigest()[:16]
    return f"{source_kind}:{name}:{digest}"


def page_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-title">{html.escape(title)}</div>
        <div class="page-subtitle">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def section_card():
    return st.container(border=True)


def html_text(value: Any) -> str:
    return html.escape(str(value))


def compact_html(markup: str) -> str:
    """Collapse per-line leading whitespace so Streamlit does not treat
    indented HTML fragments as markdown code blocks."""

    return "".join(line.strip() for line in markup.splitlines())


def format_file_size(file_size: int | None) -> str:
    if not file_size:
        return "Unavailable"
    if file_size >= 1024 * 1024:
        return f"{file_size / (1024 * 1024):.2f} MB"
    return f"{file_size / 1024:.1f} KB"


def card_metric(label: str, value: str, caption: str | None = None) -> None:
    caption_html = f'<div class="brand-sub">{html_text(caption)}</div>' if caption else ""
    st.markdown(
        compact_html(
            f"""
        <div class="metric-card">
            <div class="metric-label">{html_text(label)}</div>
            <div class="metric-value">{html_text(value)}</div>
            {caption_html}
        </div>
        """
        ),
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    checkpoint = get_checkpoint_status()

    with st.sidebar:
        st.markdown(brand_html(), unsafe_allow_html=True)
        if "nav" not in st.session_state:
            st.session_state.nav = "Home"

        query_page = st.query_params.get("page")
        if (
            query_page in NAV_ITEMS
            and st.session_state.get("last_query_page") != query_page
        ):
            st.session_state.nav = query_page
            st.session_state.last_query_page = query_page

        selected = st.radio(
            "Navigation",
            NAV_ITEMS,
            key="nav",
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div class="rail-note">
                Research/education use only.<br>
                Not a substitute for professional medical evaluation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if checkpoint["exists"]:
            st.caption("Checkpoint ready")
        else:
            st.caption("Checkpoint missing")

    return selected


def get_hero_image_path() -> Path | None:
    demos = get_demo_images()
    for demo in demos:
        if demo["filename"] == "grade2_337_r2.jpg":
            return Path(demo["path"])
    return Path(demos[0]["path"]) if demos else None


def image_data_uri(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_home() -> None:
    hero_image_path = get_hero_image_path()
    hero_src = image_data_uri(hero_image_path)
    image_html = (
        f'<div class="hero-image-wrap"><img src="{hero_src}" alt="Fundus image"></div>'
        if hero_src
        else ""
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <h1>AI-Assisted<br>Diabetic Retinopathy<br>Screening</h1>
                    <div class="hero-lead">Fast. Reliable. Research-Driven.</div>
                    <div class="hero-copy">
                        Upload a fundus image to get an AI-assisted screening prediction for diabetic retinopathy.
                    </div>
                    <div class="hero-actions">
                        <a class="primary" href="?page=Analyze%20Image">Analyze an Image &rarr;</a>
                        <a class="secondary" href="?page=About">Learn More</a>
                    </div>
                </div>
                {image_html}
            </div>
            <div class="feature-strip">
                <div class="feature-tile">
                    <div class="feature-icon">AI</div>
                    <div class="feature-title">AI Powered</div>
                    <div class="feature-copy">EfficientNet-B4 + Swin Transformer</div>
                </div>
                <div class="feature-tile">
                    <div class="feature-icon">5</div>
                    <div class="feature-title">5-Class Grading</div>
                    <div class="feature-copy">No DR to Proliferative DR</div>
                </div>
                <div class="feature-tile">
                    <div class="feature-icon">R</div>
                    <div class="feature-title">Research Focus</div>
                    <div class="feature-copy">Built on DeepDRiD v1.1</div>
                </div>
                <div class="feature-tile">
                    <div class="feature-icon">ED</div>
                    <div class="feature-title">Educational Use</div>
                    <div class="feature-copy">Not a substitute for professional evaluation</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(active: int) -> None:
    labels = ["Upload", "Quality Check", "Model Analysis", "Results"]
    steps = []
    for index, label in enumerate(labels, start=1):
        state = "done" if index < active else "active" if index == active else ""
        marker = "OK" if index < active else str(index)
        steps.append(
            f"""
            <div class="step {state}">
                <div class="step-dot">{marker}</div>
                <div>{html_text(label)}</div>
            </div>
            """
        )
    st.markdown(compact_html(f'<div class="stepper">{"".join(steps)}</div>'), unsafe_allow_html=True)


def render_demo_tiles(demos: list[dict[str, Any]]) -> None:
    columns = st.columns(5)
    for col, demo in zip(columns, demos):
        with col:
            st.image(str(demo["path"]), use_container_width=True)
            st.markdown(
                compact_html(
                    f"""
                <div class="feature-title" style="color:#0f1f33;text-align:center;font-size:0.82rem;">
                    {html_text(demo.get("reference_label", demo["filename"]))}
                </div>
                <div class="brand-sub" style="text-align:center;">Grade {html_text(demo.get("reference_class", "-"))}</div>
                """
                ),
                unsafe_allow_html=True,
            )


def build_source_from_upload(uploaded) -> dict[str, Any] | None:
    if uploaded is None:
        return None

    file_bytes = uploaded.getvalue()
    try:
        image = open_image_from_bytes(file_bytes)
    except DashboardInputError as exc:
        st.error(str(exc))
        return None

    metadata = image_metadata(image, uploaded.name, uploaded.size)
    return {
        "kind": "upload",
        "name": uploaded.name,
        "bytes": file_bytes,
        "image": image,
        "metadata": metadata,
        "token": source_token("upload", uploaded.name, file_bytes),
    }


def build_source_from_demo(selected_demo_name: str) -> dict[str, Any] | None:
    if selected_demo_name == "None":
        return None

    demos = get_demo_images()
    demo = next(item for item in demos if item["filename"] == selected_demo_name)
    image = Image.open(demo["path"]).convert("RGB")
    metadata = image_metadata(
        image,
        demo["filename"],
        Path(demo["path"]).stat().st_size,
    )
    return {
        "kind": "demo",
        "name": demo["filename"],
        "path": Path(demo["path"]),
        "image": image,
        "metadata": metadata,
        "demo": demo,
        "token": source_token("demo", str(demo["path"])),
    }


def render_image_input() -> dict[str, Any] | None:
    page_title(
        "Analyze a Fundus Image",
        "Upload a retinal fundus image (JPG, JPEG, or PNG) to get an AI-based screening prediction.",
    )

    st.markdown(
        """
        <div class="upload-panel">
            <div class="metric-label">Image Input</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drag and drop an image here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Supported formats: JPG, JPEG, PNG. Recommended maximum size: 10 MB.",
    )
    st.caption("Supported formats: JPG, JPEG, PNG | Recommended max size: 10 MB")

    demos = get_demo_images()
    st.markdown("**Or try a demo image**")
    render_demo_tiles(demos)
    selected_demo_name = st.selectbox(
        "Demo image selector",
        ["None"] + [demo["filename"] for demo in demos],
        label_visibility="collapsed",
    )

    source = build_source_from_upload(uploaded)
    if source is not None:
        return source

    return build_source_from_demo(selected_demo_name)


def render_preview(source: dict[str, Any]) -> None:
    image = source["image"]
    metadata = source["metadata"]
    file_size = metadata["file_size_bytes"]
    size_text = format_file_size(file_size)
    quality = analyze_image_quality(image)

    with section_card():
        render_stepper(2)
        left, right = st.columns([1, 1.15])
        with left:
            st.markdown("**Uploaded Image**")
            st.image(image, use_container_width=True)
        with right:
            st.markdown("**Image Information**")
            rows = {
                "File name": metadata["filename"],
                "Dimensions": f"{metadata['width']} x {metadata['height']}",
                "File size": size_text,
                "Format": source["name"].split(".")[-1].upper(),
            }
            row_html = "".join(
                f"""
                <div class="kv-row">
                    <div class="kv-key">{html_text(key)}</div>
                    <div class="kv-value">{html_text(value)}</div>
                </div>
                """
                for key, value in rows.items()
            )
            st.markdown(f'<div class="kv-table">{row_html}</div>', unsafe_allow_html=True)

            st.markdown("**Preliminary Image Quality Checks**")
            quality_rows = []
            for metric in quality["metrics"]:
                status = "Good" if metric["status"] == "No issue flagged" else "Review"
                status_class = "status-good" if status == "Good" else "status-review"
                quality_rows.append(
                    f"""
                    <div class="quality-row">
                        <div>{html_text(metric["name"])}</div>
                        <div class="status-pill {status_class}">{status}</div>
                    </div>
                    """
                )
            st.markdown("".join(quality_rows), unsafe_allow_html=True)

            if quality["issue_count"]:
                st.warning("Potential quality issue detected before model analysis.")
            else:
                st.success("Image appears suitable for model analysis.")

            if source["kind"] == "demo" and "reference_class" in source["demo"]:
                with st.expander("Reference label - evaluation/demo metadata"):
                    demo = source["demo"]
                    st.write(f"Class {demo['reference_class']} - {demo['reference_label']}")
                    st.caption("This metadata is not supplied to the model.")


def run_analysis(source: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    quality = analyze_image_quality(source["image"])
    bundle = get_cached_bundle()

    if source["kind"] == "upload":
        prediction = predict_uploaded_image(
            source["bytes"],
            source["name"],
            bundle=bundle,
        )
    else:
        prediction = predict_demo_image(source["path"], bundle=bundle)

    elapsed = time.perf_counter() - started
    prediction["metadata"]["inference_time_seconds"] = elapsed

    return {
        "source_token": source["token"],
        "quality": quality,
        "prediction": prediction,
    }


def analyze_button(source: dict[str, Any]) -> None:
    should_reset = st.session_state.get("analysis_source_token") != source["token"]
    if should_reset:
        st.session_state.pop("analysis", None)

    if st.button("Run Model Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Running preprocessing and model inference..."):
                st.session_state.analysis = run_analysis(source)
                st.session_state.analysis_source_token = source["token"]
        except DashboardInputError as exc:
            st.error(str(exc))
        except FileNotFoundError as exc:
            st.error("The model checkpoint or selected image could not be found.")
            with st.expander("Technical details"):
                st.code(str(exc))
        except RuntimeError as exc:
            st.error("Model loading or inference failed. Check dependencies, checkpoint availability, and device memory.")
            with st.expander("Technical details"):
                st.code(str(exc))
        except Exception as exc:
            st.error("Unexpected dashboard error while analyzing the image.")
            with st.expander("Technical details"):
                st.code(str(exc))


def score_bar(label: str, value: float, active: bool = False) -> None:
    percent = max(0.0, min(value, 1.0)) * 100.0
    muted = "" if active else " muted"
    st.markdown(
        f"""
        <div class="score-row">
            <div>{html_text(label)}</div>
            <div class="score-track"><div class="score-fill{muted}" style="width: {percent:.1f}%"></div></div>
            <div>{value:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_severity_scale(predicted_class: int) -> None:
    colors = ["#16a085", "#d35400", "#126bff", "#f39c12", "#ff5a2d"]
    rows = []
    for class_id, label in DR_LABELS.items():
        active = " active" if class_id == predicted_class else ""
        rows.append(
            f"""
            <div class="severity-row{active}">
                <span class="severity-pin" style="background: {colors[class_id]}"></span>
                <strong>{class_id}</strong>
                <span>{html_text(label)}</span>
            </div>
            """
        )
    st.markdown(f'<div class="severity-list">{"".join(rows)}</div>', unsafe_allow_html=True)


def render_results(source: dict[str, Any], analysis: dict[str, Any]) -> None:
    prediction = analysis["prediction"]
    quality = analysis["quality"]
    predicted_class = int(prediction["predicted_class"])
    label = prediction["predicted_label"]
    metadata = prediction.get("metadata", {})

    with section_card():
        render_stepper(4)
        top_left, top_right = st.columns([1.45, 1])
        with top_left:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="metric-label">Model-Predicted Grade</div>
                    <div class="result-grade">{predicted_class} - {html_text(label)}</div>
                    <div>AI-assisted screening prediction</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top_right:
            st.markdown(
                """
                <div class="important-card">
                    <div class="metric-label">Important</div>
                    Research/education use only. This output is not a clinical diagnosis or a substitute for professional medical evaluation.
                </div>
                """,
                unsafe_allow_html=True,
            )

        image_col, scale_col = st.columns([1, 1.1])
        with image_col:
            st.markdown("**Uploaded Image**")
            st.image(source["image"], use_container_width=True)
        with scale_col:
            st.markdown("**DR Severity Scale**")
            render_severity_scale(predicted_class)

        st.markdown("**Prediction Output**")
        cols = st.columns(4)
        with cols[0]:
            card_metric("Grade", str(predicted_class))
        with cols[1]:
            card_metric("Label", label)
        with cols[2]:
            card_metric("Confidence", "-", "No calibrated confidence in this version")
        with cols[3]:
            elapsed = metadata.get("inference_time_seconds")
            card_metric("Inference Time", f"{elapsed:.2f} s" if elapsed else "-")

        st.caption("Ordinal prediction scores are used internally. Confidence is intentionally not shown as calibrated clinical confidence.")

    render_analysis_details(prediction)
    render_decision(prediction, quality)


def render_analysis_details(prediction: dict[str, Any]) -> None:
    with section_card():
        page_title(
            "Model Analysis Details",
            "EfficientNet-B4 + Swin Transformer Base 384 + CORN",
        )

        tabs = st.tabs(["Class Scores", "Technical Details", "Preprocessing", "Explainability Future"])
        predicted_class = int(prediction["predicted_class"])
        with tabs[0]:
            st.markdown("**Ordinal Class Scores**")
            st.caption("These are ordinal prediction scores from the CORN head, not calibrated clinical probabilities.")
            class_scores = prediction.get("class_scores", {})
            for class_id, label in DR_LABELS.items():
                score = float(class_scores.get(str(class_id), 0.0))
                score_bar(f"Class {class_id} - {label}", score, active=class_id == predicted_class)
            st.markdown(
                """
                <div class="info-box">
                    The model uses a CORN ordinal classification head. Scores reflect ordinal threshold-derived class scores rather than direct softmax probabilities.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tabs[1]:
            metadata = prediction.get("metadata", {})
            st.write(f"Model: {metadata.get('architecture', 'Unavailable')}")
            st.write("Checkpoint: Epoch 15, internal validation QWK 0.8943")
            st.write(f"Checkpoint path: `{metadata.get('checkpoint', 'Unavailable')}`")
            st.write(f"Input: `{metadata.get('image_size', 384)} x {metadata.get('image_size', 384)}`")
            st.write(f"Device: `{metadata.get('device', 'Unavailable').upper()}`")
            st.write("Task: 5-class ordinal DR grading.")
            st.write("Dataset: DeepDRiD v1.1 regular fundus images.")
            with st.expander("Raw model output"):
                st.write("Ordinal threshold scores")
                st.json(prediction.get("ordinal_probabilities", {}))
                st.write("Raw CORN logits")
                st.json(prediction.get("logits", []))

        with tabs[2]:
            st.write(prediction.get("metadata", {}).get("preprocessing", "Unavailable"))
            st.caption("Training augmentations are not used during dashboard inference.")

        with tabs[3]:
            st.markdown(
                """
                <div class="future-box">
                    EfficientNet and Swin individual predictions, model disagreement, uncertainty estimation, calibration, Grad-CAM, and transformer attention will be available only after the inference layer exposes real outputs for them.
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_decision(prediction: dict[str, Any], quality: dict[str, Any]) -> None:
    with section_card():
        st.markdown("**Screening Summary**")
        st.markdown(
            f"""
            <div class="decision-row">
                <div class="decision-item">
                    <div class="metric-label">Model output</div>
                    <div class="metric-value">{html_text(prediction['predicted_class'])} - {html_text(prediction['predicted_label'])}</div>
                </div>
                <div class="decision-item">
                    <div class="metric-label">Image quality layer</div>
                    <div class="metric-value">{html_text(quality['overall'])}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Screening status: prediction generated by the existing model pipeline.")
        human_review = st.checkbox(
            "Flag this case for optional human review",
            help="Manual flag only. No automated agentic policy is active yet.",
        )
        if human_review:
            st.warning("Manual human-review flag set for this dashboard session.")
        st.caption("Future policy can combine quality, calibration, disagreement, uncertainty, and evidence consistency.")


def render_analyze_page() -> None:
    with section_card():
        active = 4 if st.session_state.get("analysis") else 1
        render_stepper(active)
        source = render_image_input()

    if source is None:
        return

    render_preview(source)
    analyze_button(source)

    analysis = st.session_state.get("analysis")
    if analysis and analysis.get("source_token") == source["token"]:
        render_results(source, analysis)


def render_demo_page() -> None:
    with section_card():
        page_title("Demo Images", "Try one of the project demo images without exposing its reference label to the model.")
        demos = get_demo_images()
        render_demo_tiles(demos)
        selected = st.selectbox("Choose demo image", [demo["filename"] for demo in demos])
        source = build_source_from_demo(selected)

    if source is None:
        return

    render_preview(source)
    analyze_button(source)
    analysis = st.session_state.get("analysis")
    if analysis and analysis.get("source_token") == source["token"]:
        render_results(source, analysis)


def render_performance_page() -> None:
    with section_card():
        page_title(
            "Model Performance",
            "Results on the held-out DeepDRiD v1.1 validation set of 400 images.",
        )

        held_out = MODEL_PERFORMANCE["held_out"]
        cols = st.columns(4)
        with cols[0]:
            card_metric("Accuracy", held_out["Accuracy"])
        with cols[1]:
            card_metric("QWK", held_out["QWK"])
        with cols[2]:
            card_metric("Macro F1", held_out["Macro F1"])
        with cols[3]:
            card_metric("Weighted F1", held_out["Weighted F1"])

        st.caption("Held-out evaluation QWK is 0.7813. The 0.8943 checkpoint QWK is an internal validation-selection score, not final held-out performance.")

    left, right = st.columns([1.15, 1])
    with left:
        with section_card():
            tabs = st.tabs(["Per-class F1", "Per-class Recall"])
            with tabs[0]:
                for label, value in MODEL_PERFORMANCE["per_class_f1"].items():
                    score_bar(label, float(value))
            with tabs[1]:
                for label, value in MODEL_PERFORMANCE["per_class_recall"].items():
                    score_bar(label, float(value))
    with right:
        with section_card():
            st.markdown("**Dataset Information**")
            dataset_rows = {
                "Dataset": "DeepDRiD v1.1",
                "Training images": "1,200",
                "Validation images": "400 held-out",
                "Image type": "Regular fundus",
                "UWF data": "Not used",
                "Online challenge data": "Not used",
            }
            row_html = "".join(
                f"""
                <div class="kv-row">
                    <div class="kv-key">{html_text(key)}</div>
                    <div class="kv-value">{html_text(value)}</div>
                </div>
                """
                for key, value in dataset_rows.items()
            )
            st.markdown(f'<div class="kv-table">{row_html}</div>', unsafe_allow_html=True)

    if CONFUSION_MATRIX_PATH.is_file():
        with section_card():
            st.markdown("**Confusion Matrix**")
            st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)


def render_about_page() -> None:
    with section_card():
        page_title("About RetinaAI", "A research dashboard for AI-assisted diabetic retinopathy screening.")
        st.write(
            "This dashboard wraps the existing single-image inference pipeline for a dual-backbone ordinal model: EfficientNet-B4 plus Swin Transformer Base 384 with a CORN fusion head."
        )
        st.write(
            "The app is designed so future inference fields can be added in the service layer without rewriting the user interface."
        )
        st.warning("Research/education use only. This model is not a substitute for professional medical evaluation.")


def render_faq_page() -> None:
    with section_card():
        page_title("FAQ", "Operational notes for the dashboard.")
        with st.expander("Does the dashboard run the real model?", expanded=True):
            st.write("Yes. The Analyze action calls `predict.py` through the dashboard service layer.")
        with st.expander("Are the displayed scores calibrated probabilities?"):
            st.write("No. They are CORN-derived ordinal class scores and are labeled as such.")
        with st.expander("Are Grad-CAM or attention maps implemented?"):
            st.write("No. Those sections are extension points only until real explainability outputs are produced by the inference layer.")
        with st.expander("Is this a diagnosis?"):
            st.write("No. The dashboard displays AI-assisted screening predictions for research and education use only.")


def main() -> None:
    inject_css()
    selected = render_sidebar()

    if selected == "Home":
        render_home()
    elif selected == "Analyze Image":
        render_analyze_page()
    elif selected == "Demo Images":
        render_demo_page()
    elif selected == "Model Performance":
        render_performance_page()
    elif selected == "About":
        render_about_page()
    elif selected == "FAQ":
        render_faq_page()


if __name__ == "__main__":
    main()
