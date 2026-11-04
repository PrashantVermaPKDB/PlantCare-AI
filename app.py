import json
from pathlib import Path
from html import escape

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================
# APP CONFIG
# =====================================================
st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_v2.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names_v2.json"
IMAGE_SIZE = (160, 160)

if "page_nav" not in st.session_state:
    st.session_state["page_nav"] = "Home"

# =====================================================
# PROFESSIONAL GREEN THEME
# =====================================================
st.markdown(
    """
    <style>
    :root {
        --green: #16803c;
        --dark-green: #105c2b;
        --light-green: #edf7ef;
        --text: #193b28;
    }

    .stApp {
        background: #f7faf7;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(247, 250, 247, 0.96);
    }

    .block-container {
        max-width: 1400px;
        padding: 1.5rem 2rem 2rem;
    }

    .navbar-label {
        color: #105c2b;
        font-size: 24px;
        font-weight: 800;
        white-space: nowrap;
        padding: 8px 0;
    }

    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    div.stButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 10px;
        border: 1px solid #d5e8d9;
        background: #ffffff;
        color: #105c2b;
        font-size: 14px;
        font-weight: 600;
        white-space: normal;
        line-height: 1.35;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #e5f4e9;
        border-color: #16803c;
        color: #105c2b;
    }

    div.stButton > button[kind="primary"] {
        background: #16803c;
        border-color: #16803c;
        color: white;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #105c2b;
        border-color: #105c2b;
        color: white;
    }

    .hero {
        background: linear-gradient(120deg, #105c2b, #279b50);
        padding: 36px;
        border-radius: 20px;
        color: white;
        margin: 12px 0 24px;
    }

    .hero h1 {
        color: white;
        font-size: 36px;
        margin: 0 0 12px;
    }

    .hero p {
        color: #e5f5e9;
        font-size: 16px;
çi        line-height: 1.7;
        margin-bottom: 0;
    }

    .section-title {
        color: #105c2b;
        font-size: 25px;
        font-weight: 750;
        margin: 15px 0;
    }

    .panel {
        background: white;
        border: 1px solid #dceade;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 14px;
        min-height: 120px;
        overflow-wrap: anywhere;
    }

    .panel h3 {
        color: #105c2b;
        font-size: 19px;
        margin-top: 0;
    }

    .muted {
        color: #607467;
        line-height: 1.7;
        font-size: 14px;
    }

    .metric {
        background: white;
        border: 1px solid #dceade;
        border-radius: 14px;
        padding: 18px;
        min-height: 100px;
        overflow-wrap: anywhere;
    }

    .metric-label {
        color: #637669;
        font-size: 13px;
    }

    .metric-value {
        color: #105c2b;
        font-size: 25px;
        font-weight: 800;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }

    .result-name {
        color: #105c2b;
        font-size: 22px;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #9bcaaa;
        border-radius: 14px;
        padding: 12px;
    }

    .footer {
        text-align: center;
        color: #718075;
        font-size: 12px;
        padding: 22px 0 8px;
        line-height: 1.8;
    }

    hr {
        border-color: #dceade;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 24px;
        }

        .hero h1 {
            font-size: 28px;
        }

        .navbar-label {
            font-size: 21px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# MODEL LOADING
# =====================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def format_name(name):
    return name.replace("___", " - ").replace("__", " ").replace("_", " ")


def is_healthy(name):
    return "healthy" in name.lower()


def class_description(name):
    if is_healthy(name):
        return (
            "The model classified this image in a healthy-leaf category. "
            "Inspect the whole plant for other symptoms too."
        )

    return (
        "The model identified this possible disease category. "
        "Compare visible symptoms and seek expert advice if needed."
    )


# =====================================================
# NAVBAR - NO RADIO WIDGET
# =====================================================
current_page = st.session_state["page_nav"]

nav_logo, nav_home, nav_detect, nav_library, nav_about = st.columns(
    [1.8, 0.8, 1.35, 1.4, 1.3],
    gap="small",
    vertical_alignment="center",
)

with nav_logo:
    st.markdown(
        '<div class="navbar-label">🌿 PlantCare AI</div>',
        unsafe_allow_html=True,
    )

with nav_home:
    if st.button(
        "Home",
        key="nav_home",
        type="primary" if current_page == "Home" else "secondary",
        use_container_width=True,
    ):
        st.session_state["page_nav"] = "Home"
        st.rerun()

with nav_detect:
    if st.button(
        "Detect Disease",
        key="nav_detect",
        type="primary" if current_page == "Detect Disease" else "secondary",
        use_container_width=True,
    ):
        st.session_state["page_nav"] = "Detect Disease"
        st.rerun()

with nav_library:
    if st.button(
        "Disease Library",
        key="nav_library",
        type="primary" if current_page == "Disease Library" else "secondary",
        use_container_width=True,
    ):
        st.session_state["page_nav"] = "Disease Library"
        st.rerun()

with nav_about:
    if st.button(
        "About Project",
        key="nav_about",
        type="primary" if current_page == "About Project" else "secondary",
        use_container_width=True,
    ):
        st.session_state["page_nav"] = "About Project"
        st.rerun()

page = st.session_state["page_nav"]
st.divider()

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Smarter Plant Care Starts Here 🌱</h1>
            <p>
                Explore AI-powered plant leaf classification.
                Upload a leaf photo, review the prediction and learn
                about the categories supported by the model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    for col, label, value in [
        (a, "AI Architecture", "MobileNetV2"),
        (b, "Supported Classes", "15"),
        (c, "Validation Accuracy", "94.21%"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Explore PlantCare AI</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown(
            """
            <div class="panel">
                <h3>🔍 Disease Detection</h3>
                <p class="muted">
                    Upload a leaf image and view the predicted category,
                    prediction score and top three results.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Open Disease Detection", key="home_detect",
                     use_container_width=True):
            st.session_state["page_nav"] = "Detect Disease"
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="panel">
                <h3>📚 Disease Library</h3>
                <p class="muted">
                    Search supported categories and explore
                    information about the model's predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Explore Disease Library", key="home_library",
                     use_container_width=True):
            st.session_state["page_nav"] = "Disease Library"
            st.rerun()

    st.markdown(
        """
        <div class="panel">
            <h3>⚙️ How it works</h3>
            <p class="muted">
                Upload a leaf image → Resize image → MobileNetV2
                classification → Prediction results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =====================================================
# DISEASE DETECTION
# =====================================================
elif page == "Detect Disease":
    st.markdown(
        '<div class="section-title">🔬 Plant Disease Detection</div>',
        unsafe_allow_html=True,
    )
    st.write("Upload a clear Pepper, Potato or Tomato leaf image.")

    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"],
        key="detection_upload",
    )

    if uploaded_file is None:
        st.info("Upload a leaf image above to get started.")

    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            left, right = st.columns([1, 1.15], gap="large")

            with left:
                st.markdown("#### 📷 Uploaded Leaf")
                st.image(image, use_container_width=True)
                st.caption(
                    f"Image dimensions: {image.width} × {image.height}"
                )

            with right:
                st.markdown("#### 🧪 Prediction Result")

                if not MODEL_PATH.exists() or not LABELS_PATH.exists():
                    st.error(
                        "Model or class labels missing. "
                        "Check the files inside the models folder."
                    )
                else:
                    with st.spinner("Analyzing leaf image..."):
                        model = load_model()
                        class_names = load_class_names()

                        resized = image.resize(IMAGE_SIZE)
                        array = np.asarray(resized, dtype=np.float32)
                        array = np.expand_dims(array, axis=0)

                        probabilities = model.predict(array, verbose=0)[0]

                    if len(probabilities) != len(class_names):
                        st.error(
                            "Model output and class labels do not match."
                        )
                    else:
                        top_indices = np.argsort(probabilities)[-3:][::-1]
                        best_idx = int(top_indices[0])
                        best_name = class_names[best_idx]
                        confidence = float(probabilities[best_idx]) * 100

                        if is_healthy(best_name):
                            st.success(
                                f"Healthy category: {format_name(best_name)}"
                            )
                        else:
                            st.warning(
                                f"Possible disease: {format_name(best_name)}"
                            )

                        st.markdown(
                            f'<div class="result-name">'
                            f'{escape(format_name(best_name))}</div>',
                            unsafe_allow_html=True,
                        )
                        st.metric("Prediction score", f"{confidence:.2f}%")
                        st.progress(float(np.clip(confidence / 100, 0, 1)))

                        st.markdown("**About this category**")
                        st.write(class_description(best_name))

            if (
                "probabilities" in locals()
                and len(probabilities) == len(class_names)
            ):
                st.divider()
                st.markdown(
                    '<div class="section-title">📊 Top 3 Predictions</div>',
                    unsafe_allow_html=True,
                )

                cols = st.columns(3)

                for rank, idx in enumerate(top_indices):
                    idx = int(idx)
                    label = format_name(class_names[idx])
                    score = float(probabilities[idx]) * 100

                    with cols[rank]:
                        st.markdown(
                            f"""
                            <div class="panel">
                                <div class="metric-label">Rank {rank + 1}</div>
                                <div class="result-name">{escape(label)}</div>
                                <div class="metric-value">{score:.2f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.progress(float(np.clip(score / 100, 0, 1)))

                st.caption(
                    "These scores are model estimates, not guarantees. "
                    "Confirm disease symptoms before taking action."
                )

        except Exception as error:
            st.error("Unable to process this image. Try another image.")
            st.exception(error)

# =====================================================
# DISEASE LIBRARY
# =====================================================
elif page == "Disease Library":
    st.markdown(
        '<div class="section-title">📚 Plant Category Library</div>',
        unsafe_allow_html=True,
    )
    st.write("Browse the 15 categories supported by the trained model.")

    if LABELS_PATH.exists():
        class_names = load_class_names()
        search = st.text_input(
            "Search categories",
            placeholder="Search Tomato, Potato, healthy...",
        )

        filtered = [
            name for name in class_names
            if search.lower() in format_name(name).lower()
        ]

        st.caption(
            f"Showing {len(filtered)} of {len(class_names)} categories"
        )

        for start in range(0, len(filtered), 2):
            cols = st.columns(2)

            for offset, name in enumerate(filtered[start:start + 2]):
                with cols[offset]:
                    category = (
                        "Healthy category"
                        if is_healthy(name)
                        else "Disease category"
                    )
                    st.markdown(
                        f"""
                        <div class="panel">
                            <h3>{escape(format_name(name))}</h3>
                            <p class="muted"><b>{category}</b></p>
                            <p class="muted">
                                {escape(class_description(name))}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if not filtered:
            st.warning("No matching categories found.")
    else:
        st.error("Class labels file not found.")

# =====================================================
# ABOUT PROJECT
# =====================================================
elif page == "About Project":
    st.markdown(
        '<div class="section-title">ℹ️ About PlantCare AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>AI for Plant Health 🌱</h1>
            <p>
                An educational deep learning project that classifies
                plant leaf images into supported categories.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project objective")
    st.write(
        "Demonstrate how computer vision and deep learning can assist "
        "with preliminary classification of plant leaf images."
    )

    st.markdown("### Technology stack")
    t1, t2, t3 = st.columns(3)

    for col, title, description in [
        (t1, "🐍 Python", "Application programming language"),
        (t2, "🧠 TensorFlow", "Deep learning framework"),
        (t3, "🌐 Streamlit", "Interactive web application"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="panel">
                    <h3>{title}</h3>
                    <p class="muted">{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Model details")
    st.write("- Architecture: MobileNetV2 with transfer learning")
    st.write("- Input image size: 160 × 160 RGB")
    st.write("- Supported classes: 15")
    st.write("- Dataset images: 20,638")
    st.write("- Best validation accuracy during training: 94.21%")

    st.warning(
        "Validation accuracy is not a guarantee of real-world accuracy. "
        "This educational app does not replace expert agricultural advice."
    )

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown(
    """
    <div class="footer">
        🌿 PlantCare AI · MobileNetV2 · TensorFlow · Streamlit<br>
        Educational project | Verify AI predictions independently
    </div>
    """,
    unsafe_allow_html=True,
)
