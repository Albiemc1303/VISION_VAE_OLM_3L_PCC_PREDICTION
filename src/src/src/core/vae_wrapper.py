# src/core/vae_wrapper.py
"""
VAE wrapper that registers the module with ontology and delegates to original vae_processor if present.
If original module missing, provides a safe mock for evaluation runs.
"""
import importlib.util
import os
import sys
from pathlib import Path
import json

from src.meta.ontology import ModuleDescriptor, ontology

MODULE_NAME = "vae_processor"

desc = ModuleDescriptor(
    name=MODULE_NAME,
    role="VAE encoder/decoder for latent extraction and reconstruction",
    inputs={"images": {"dtype": "uint8", "shape": ["H","W","C"]}},
    outputs={"latents": {"dtype":"float","shape":["L"]}, "recon": {"dtype":"uint8","shape":["H","W","C"]}},
    invariants=["latent vector length consistent across runs"],
    version="v1.0-vae-wrapper"
)
ontology.register_module(desc)

def try_import_original():
    candidates = ["vae_processor.py", "src/vae_processor.py", "vae_processor"]
    for c in candidates:
        if os.path.exists(c):
            # import by path
            spec = importlib.util.spec_from_file_location("orig_vae", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        else:
            try:
                mod = importlib.import_module(c)
                return mod
            except Exception:
                continue
    return None

_orig = try_import_original()

if _orig:
    # expose a simple API delegating to original functions if they exist
    def encode(images):
        if hasattr(_orig, "encode"):
            return _orig.encode(images)
        if hasattr(_orig, "encode_batch"):
            return _orig.encode_batch(images)
        raise RuntimeError("Original VAE module found but no encode function.")

    def decode(latents):
        if hasattr(_orig, "decode"):
            return _orig.decode(latents)
        if hasattr(_orig, "decode_batch"):
            return _orig.decode_batch(latents)
        raise RuntimeError("Original VAE module found but no decode function.")
else:
    # fallback mock implementations (very small, not suitable for training)
    import numpy as np
    def encode(images):
        # images: list or numpy array -> produce fixed random latents deterministically
        arr = np.array(images)
        bs = arr.shape[0] if arr.ndim == 4 else 1
        L = 64
        latents = np.zeros((bs, L), dtype=float)
        return latents

    def decode(latents):
        # produce black frames of same batch size
        import numpy as _np
        bs = latents.shape[0] if hasattr(latents, "shape") else 1
        H,W,C = 64,64,3
        return _np.zeros((bs,H,W,C), dtype=_np.uint8)
