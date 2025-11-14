# src/core/pipeline.py
"""
High-level pipeline orchestrator (New-Paradigm wrapper for original AI.py).
Registers module, enforces light contract, and coordinates VAE <-> LSTM flows.
"""

import os
import json
import numpy as np

from src.meta.ontology import ModuleDescriptor, ontology
from src.core.vae_wrapper import encode, decode
from src.core.lstm_wrapper import predict

MODULE_NAME = "pipeline_orchestrator"
desc = ModuleDescriptor(
    name=MODULE_NAME,
    role="Orchestrator: converts frames -> latents -> predicts next latent -> decodes to frames",
    inputs={"frames": {"dtype":"uint8","shape":["B","H","W","C"]}},
    outputs={"predicted_frames": {"dtype":"uint8","shape":["B","H","W","C"]}},
    invariants=["latents shape stable across pipeline"],
    version="v1.0-pipeline"
)
ontology.register_module(desc)

def step_predict(frames):
    """
    frames: list or numpy array (B,H,W,C)
    returns: predicted frames (B,H,W,C), metrics dict (empty)
    """
    latents = encode(frames)            # (B, L) or similar
    # For prediction we create a sequence from previous frames' latents.
    # naively we will use whole batch as sequence: (T,B,L) -> do per-batch
    if latents is None:
        raise RuntimeError("VAE encode returned None")
    # If latents shape is (B,L), create a small temporal window of length 3 by stacking
    if latents.ndim == 2:
        # create pseudo-sequence: repeat last N times
        seq = np.stack([latents for _ in range(3)], axis=0)  # (T,B,L)
        seq = seq.transpose(1,0,2)  # (B,T,L)
        # pick first batch element
        seq0 = seq[0]
        pred_latent = predict(seq0)
        pred_frame = decode(pred_latent.reshape(1, -1))[0]
        metrics = {}
        return [pred_frame], metrics
    else:
        # latents already temporal -> pick last sequence
        seq = latents
        pred_latent = predict(seq)
        pred_frame = decode(pred_latent.reshape(1,-1))[0]
        return [pred_frame], {}
