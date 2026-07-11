import streamlit as st
from PIL import Image
from pathlib import Path
import tempfile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEVICE
from src.data.transforms import get_val_transforms
from src.inference.embeddings import (
    build_feature_extractor,
    extract_embedding,
)
from src.inference.change_detector import (
    cosine_similarity,
    detect_change,
)
from src.visualization.plots import (
    plot_change_heatmap,
)
from app.utils.inference import load_model


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Temporal Change Detection",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Temporal Change Detection")

st.markdown(
    """
Upload two satellite images captured at different times and detect whether a significant land-use change has occurred.
"""
)

st.divider()


# --------------------------------------------------
# Load Feature Extractor
# --------------------------------------------------

@st.cache_resource
def get_feature_extractor():

    model = load_model(
        "models/checkpoints/resnet18_finetuned.pth"
    )

    feature_extractor = build_feature_extractor(model)

    feature_extractor.to(DEVICE)

    feature_extractor.eval()

    return feature_extractor


feature_extractor = get_feature_extractor()

transform = get_val_transforms()


# --------------------------------------------------
# Upload Images
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    image_a = st.file_uploader(
        "Image A (Earlier)",
        type=["jpg", "jpeg", "png"],
        key="image_a",
    )

with col2:

    image_b = st.file_uploader(
        "Image B (Later)",
        type=["jpg", "jpeg", "png"],
        key="image_b",
    )


# --------------------------------------------------
# Change Detection
# --------------------------------------------------

if image_a is not None and image_b is not None:

    img1 = Image.open(image_a).convert("RGB")
    img2 = Image.open(image_b).convert("RGB")

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False,
    ) as file1:

        img1.save(file1.name)
        image1_path = file1.name

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False,
    ) as file2:

        img2.save(file2.name)
        image2_path = file2.name

    with st.spinner("Detecting changes..."):

        embedding1 = extract_embedding(
            image1_path,
            feature_extractor,
            transform,
            DEVICE,
        )

        embedding2 = extract_embedding(
            image2_path,
            feature_extractor,
            transform,
            DEVICE,
        )

        similarity = cosine_similarity(
            embedding1,
            embedding2,
        )

        decision = detect_change(
            similarity,
        )

        heatmap = plot_change_heatmap(
            image1_path,
            image2_path,
        )

    st.success("Change Detection Complete")

    st.divider()

    left, middle, right = st.columns([1, 1, 2])

    # -------------------------
    # Image A
    # -------------------------

    with left:

        st.image(
            img1,
            caption="Image A",
            width=280,
        )

    # -------------------------
    # Image B
    # -------------------------

    with middle:

        st.image(
            img2,
            caption="Image B",
            width=280,
        )

    # -------------------------
    # Detection Result
    # -------------------------

    with right:

        st.subheader("Detection Result")

        st.metric(
            "Cosine Similarity",
            f"{similarity:.4f}",
        )

        st.metric(
            "Decision",
            decision,
        )

    st.divider()

    # -------------------------
    # Heatmap
    # -------------------------

    st.subheader("Change Heatmap")

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.pyplot(heatmap)

    st.divider()

    # -------------------------
    # Final Decision
    # -------------------------

    if decision == "Significant Change":

        st.error(
            "Significant land-use change detected between the uploaded images."
        )

    else:

        st.success(
            "No significant land-use change detected."
        )