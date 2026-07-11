import streamlit as st
from PIL import Image
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEVICE
from src.data.transforms import get_val_transforms
from src.models.resnet18_transfer import build_resnet18

from app.utils.inference import (
    load_model,
    predict_image,
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Land Use Classification",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Land Use Classification")

st.markdown(
    """
Upload a satellite image and classify its land-use category using the fine-tuned **ResNet18** model.
"""
)

# --------------------------------------------------
# Classes
# --------------------------------------------------

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def get_model():

    return load_model(
        "models/checkpoints/resnet18_finetuned.pth"
    )


model = get_model()

transform = get_val_transforms()

# --------------------------------------------------
# Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload satellite image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Classifying image..."):

        result = predict_image(
            model=model,
            image=image,
            transform=transform,
            class_names=CLASS_NAMES,
        )

    st.success("Prediction Complete")

    left, right = st.columns([1, 2])

    # ----------------------------
    # LEFT SIDE
    # ----------------------------

    with left:

        st.image(
            image,
            caption="Uploaded Image",
            width=350,
        )

    # ----------------------------
    # RIGHT SIDE
    # ----------------------------
    with right:

        st.subheader("Prediction Result")

        st.metric(
        "Predicted Class",
        result["prediction"],
    )

    st.metric(
        "Confidence",
        f"{result['confidence']*100:.2f}%"
    )

    st.metric(
        "Inference Time",
        f"{result['inference_time']:.4f} sec"
    )

    st.divider()

    st.subheader("Top-3 Predictions")

    for cls, score in result["top3"]:

        st.progress(score)

        st.write(
            f"**{cls}** : {score*100:.2f}%"
        )

    st.divider()
