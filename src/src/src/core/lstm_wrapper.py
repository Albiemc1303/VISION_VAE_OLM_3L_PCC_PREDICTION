# src/core/lstm_wrapper.py
"""
LSTM predictor wrapper that registers with ontology and delegates to repo's lstm_models if present.
Provides predict(sequence_of_latents) -> next_latent
"""

import importlib.util
import os
from src.meta.ontology import ModuleDescriptor, ontology

MODULE_NAME = "lstm_predictor"
desc = ModuleDescriptor(
    name=MODULE_NAME,
    role="3-layer LSTM predictor for temporal prediction on latents",
    inputs={"latent_sequence": {"dtype":"float","shape":["T","L"]}},
    outputs={"predicted_latent": {"dtype":"float","shape":["L"]}},
    invariants=["predictor returns shape (L,) for single-step prediction"],
    version="v1.0-lstm-wrapper"
)
ontology.register_module(desc)

def try_import_original():
    candidates = ["lstm_models.py", "src/lstm_models.py", "lstm_models"]
    for c in candidates:
        if os.path.exists(c):
            spec = importlib.util.spec_from_file_location("orig_lstm", c)
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
    def predict(seq):
        if hasattr(_orig, "predict"):
            return _orig.predict(seq)
        if hasattr(_orig, "predict_next"):
            return _orig.predict_next(seq)
        raise RuntimeError("Original LSTM module found but no predict function.")
else:
    import numpy as np
    def predict(seq):
        # naive last-step copy
        arr = np.array(seq)
        return arr[-1]
