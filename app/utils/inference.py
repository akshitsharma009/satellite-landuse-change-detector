"""
inference.py
------------
Utilities for model inference in the Streamlit application.
"""

import time

import torch
from PIL import Image

from src.config import DEVICE


def predict_image(
    model,
    image,
    transform,
    class_names,
):
    """
    Predict the land-use class for a single image.
    """

    model.eval()

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    start_time = time.time()

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

    inference_time = time.time() - start_time

    confidence, prediction = torch.max(
        probabilities,
        dim=1,
    )

    prediction = prediction.item()

    confidence = confidence.item()

    top3_probabilities, top3_indices = torch.topk(
        probabilities,
        k=3,
    )

    top3 = []

    for score, idx in zip(
        top3_probabilities[0],
        top3_indices[0],
    ):

        top3.append(
            (
                class_names[idx],
                score.item(),
            )
        )

    return {
        "prediction": class_names[prediction],
        "confidence": confidence,
        "top3": top3,
        "inference_time": inference_time,
    }
from torchvision.models import resnet18

from src.models.resnet18_transfer import build_resnet18


def load_model(checkpoint_path):
    """
    Load the fine-tuned ResNet18 model.
    """

    model = build_resnet18()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model