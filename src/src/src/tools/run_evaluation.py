# src/tools/run_evaluation.py
"""
High-level evaluation runner that uses the new pipeline wrappers.
This runs a small deterministic evaluation (using sample frames in data/ or dummy frames),
produces metrics and prints JSON to stdout for capture by extract_metrics.py.
"""

import json
import numpy as np
from pathlib import Path
from src.core.pipeline import step_predict

def load_sample_frames(n=1, H=64, W=64, C=3):
    # Try to load real frames from data/ if available
    p = Path("data")
    if p.exists():
        files = list(p.glob("**/*.png")) + list(p.glob("**/*.jpg"))
        if files:
            # naive load first n using PIL if available
            try:
                from PIL import Image
                frames = []
                for f in files[:n]:
                    im = Image.open(f).convert("RGB")
                    im = im.resize((W,H))
                    arr = np.array(im).astype(np.uint8)
                    frames.append(arr)
                return frames
            except Exception:
                pass
    # fallback: black frames
    return [np.zeros((H,W,C), dtype=np.uint8) for _ in range(n)]

def compute_simple_metrics(gt_frames, pred_frames):
    # compute naive MSE, PSNR approximation
    import numpy as _np
    gt = _np.array(gt_frames).astype(float)
    pr = _np.array(pred_frames).astype(float)
    mse = float(_np.mean((gt - pr)**2))
    psnr = None
    if mse > 0:
        psnr = 20 * _np.log10(255.0 / _np.sqrt(mse))
    else:
        psnr = 100.0
    return {"mse": mse, "psnr": float(psnr)}

def main():
    frames = load_sample_frames(n=1)
    preds, _ = step_predict(frames)
    metrics = compute_simple_metrics(frames, preds)
    print(json.dumps(metrics))
    # also write to file for auditing
    Path("metrics").mkdir(parents=True, exist_ok=True)
    with open("metrics/latest_metrics.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
