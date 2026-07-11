"""
plots.py
--------
Visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_change_heatmap(
    image_a_path,
    image_b_path,
):
    """
    Generate a pixel-wise change heatmap.
    """

    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")

    image_a = np.asarray(image_a).astype(np.float32)
    image_b = np.asarray(image_b).astype(np.float32)

    difference = np.abs(image_a - image_b)

    heatmap = difference.mean(axis=2)

    return heatmap


def plot_change_heatmap(
    image_a_path,
    image_b_path,
):
    """
    Create a clean heatmap figure for Streamlit.
    """

    heatmap = generate_change_heatmap(
        image_a_path,
        image_b_path,
    )

    fig, ax = plt.subplots(
        figsize=(4.5, 4.5),   # Smaller figure
        dpi=200,
    )

    ax.imshow(
        heatmap,
        cmap="hot",
        interpolation="nearest",
    )

    ax.set_title(
        "Change Heatmap",
        fontsize=14,
        fontweight="bold",
        pad=8,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # Remove border
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.2)

    return fig