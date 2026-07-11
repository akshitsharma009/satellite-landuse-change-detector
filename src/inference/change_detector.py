"""
change_detector.py
"""

import torch
import torch.nn.functional as F

from src.config import CHANGE_THRESHOLD


def cosine_similarity(
    embedding1,
    embedding2,
):

    similarity = F.cosine_similarity(
        embedding1,
        embedding2,
    )

    return similarity.item()


def detect_change(
    similarity_score,
    threshold=CHANGE_THRESHOLD,
):

    if similarity_score >= threshold:

        return "No Significant Change"

    return "Significant Change"