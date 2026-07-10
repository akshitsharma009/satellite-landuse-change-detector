"""
embeddings.py
-------------
Utilities for extracting feature embeddings using a fine-tuned ResNet18 model.
"""

import torch
import torch.nn as nn


def build_feature_extractor(model):
    """
    Remove the classifier head and return a feature extractor.
    """

    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.eval()

    return feature_extractor
from PIL import Image


def extract_embedding(
    image_path,
    feature_extractor,
    transform,
    device,
):
    """
    Extract a 512-dimensional embedding from a satellite image.
    """

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():

        embedding = feature_extractor(image)

        embedding = embedding.flatten(start_dim=1)

    return embedding