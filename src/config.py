"""
Project Configuration
---------------------
Central configuration file for the Satellite Image Land-Use Classifier
and Temporal Change Detector project.
"""

from pathlib import Path
import torch


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

EUROSAT_DIR = RAW_DATA_DIR / "EuroSAT" / "2750"
UC_MERCED_DIR = RAW_DATA_DIR / "uc_merced"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"


# ==========================================================
# Dataset Configuration
# ==========================================================

NUM_CLASSES = 10
IMAGE_SIZE = 224

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# ==========================================================
# Training Configuration
# ==========================================================

BATCH_SIZE = 32
NUM_WORKERS = 4

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42


# ==========================================================
# Device Configuration
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Utility
# ==========================================================

def print_config():
    """Print important project configuration."""

    print("=" * 50)
    print("Project Configuration")
    print("=" * 50)

    print(f"Project Root : {PROJECT_ROOT}")
    print(f"EuroSAT Path : {EUROSAT_DIR}")
    print(f"UC Merced    : {UC_MERCED_DIR}")
    print(f"Device        : {DEVICE}")
    print(f"Image Size    : {IMAGE_SIZE}")
    print(f"Batch Size    : {BATCH_SIZE}")
    print("=" * 50)
    # ----------------------------------------------------
# Change Detection
# ----------------------------------------------------

CHANGE_THRESHOLD = 0.6345