# src/config.py
"""
Project configuration constants for New-Paradigm VISION VAE project.
Central source for sizes, device selection, and small runtime limits.
"""

import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Model sizes (tunable)
IMG_H = 64
IMG_W = 64
IMG_C = 3
LATENT_DIM = 64

# LSTM predictor sizes
LSTM_HIDDEN = 256
LSTM_LAYERS = 3
LSTM_DROPOUT = 0.1

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
