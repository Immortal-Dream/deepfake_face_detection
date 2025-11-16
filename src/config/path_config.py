from pathlib import Path

"""
Path Configuration for Deepfake Face Detection Project
This module contains all path constants for easy access to data directories.
"""

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_ROOT = SRC_ROOT / "config"

# DL Models Layer
MODEL_FOLDER = PROJECT_ROOT / "models"

# Dataset root
RVF10K_ROOT = DATA_ROOT / "rvf10k"

# Training data paths
TRAIN_ROOT = RVF10K_ROOT / "train"
TRAIN_FAKE = TRAIN_ROOT / "fake"
TRAIN_REAL = TRAIN_ROOT / "real"

# Validation data paths
VALID_ROOT = RVF10K_ROOT / "valid"
VALID_FAKE = VALID_ROOT / "fake"
VALID_REAL = VALID_ROOT / "real"

# CSV files
TRAIN_CSV = RVF10K_ROOT / "train.csv"
VALID_CSV = RVF10K_ROOT / "valid.csv"

# Experiment Output Folder
OUTPUT_FOLDER = DATA_ROOT / "output"

# Leader board result path
OVERALL_FOLDER = OUTPUT_FOLDER / "overall"
