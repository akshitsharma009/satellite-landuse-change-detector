"""
change_detector.py
------------------
Utilities for embedding-based temporal change detection.
"""

from src.config import CHANGE_THRESHOLD


def detect_change(similarity_score, threshold=CHANGE_THRESHOLD):
    """
    Decide whether significant land-use change has occurred.
    """

    if similarity_score >= threshold:
        return "No Significant Change"

    return "Significant Change"