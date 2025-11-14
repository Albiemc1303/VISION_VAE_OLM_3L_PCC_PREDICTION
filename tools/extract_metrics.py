#!/usr/bin/env python
"""
tools/extract_metrics.py

Purpose:
  - Run the repository's evaluation / visualization script if available
  - Capture standard metrics (PSNR, SSIM, MSE, loss, FPS, latent_mse, etc.)
  - Save to JSON file suitable for audit/report_audit.py

Strategy:
  1. Try to import create_visualizations and call a main() or evaluate() function.
  2. If import fails, try to run common evaluation scripts with subprocess and capture stdout.
  3. Parse stdout for JSON blobs or regex metric lines.
  4. Save metrics to provided output path.

Usage:
  python3 tools/extract_metrics.py --run-cmd "python3 create_visualizations.py" --out before_metrics.json
  or simply:
  python3 tools/extract_metrics.py --out before_metrics.json

Notes:
  This script is intentionally resilient: it won't change model weights or checkpoints.
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_METRICS = {
    "psnr": None,
    "ssim": None,
    "mse": None,
    "latent_mse": None,
    "val_loss": None,
    "train_loss": None,
    "fps": None
}

def try_import_and_run(module_path_candidates):
    """
    Try to import candidate modules and call evaluate() or main().
    Returns: (metrics_dict or None, textual_output or None)
    """
    for mp in module_path_candidates:
        if not mp:
            continue
        try:
            # Convert path-like to module import if possible
            if os.path.exists(mp):
                # add repo root to sys.path and import by filename without .py
                sys.path.insert(0, os.path.abspath("."))
                mod_name = Path(mp).stem
                spec = importlib.util.spec_from_file_location(mod_name, mp)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:
                # try to import by module name
                mod = importlib.import_module(mp)
            # prefer evaluate() -> run() -> main()
            for fn in ("evaluate", "run", "main"):
                if hasattr(mod, fn):
                    try:
                        res = getattr(mod, fn)()
                        # If result is a dict, assume it's metrics
                        if isinstance(res, dict):
                            return res, None
                        # If function printed JSON to stdout, we may catch that in subprocess path instead
                    except TypeError:
                        # call without args failed; continue
                        pass
            # else continue to next candidate
        except Exception as e:
            # swallow and continue
            # print("import attempt failed for", mp, ":", e)
            continue
    return None, None

def run_subprocess(cmd):
    """
    Runs a shell command and returns stdout (string).
    """
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=600)
        return out.decode("utf-8", errors="ignore")
    except subprocess.CalledProcessError as e:
        return e.output.decode("utf-8", errors="ignore")
    except Exception as e:
        return ""

def parse_metrics_from_text(text):
    """
    Attempts to parse common metrics from text:
      - JSON blob
      - lines like "PSNR: 30.23" or "psnr=30.23"
    Returns dict (possibly partial).
    """
    if not text:
        return {}

    # Try to extract JSON blob first
    json_blob = None
    # naive find braces JSON
    m = re.search(r'(\{[\s\S]*\})', text)
    if m:
        try:
            json_blob = json.loads(m.group(1))
            if isinstance(json_blob, dict):
                return json_blob
        except Exception:
            pass

    metrics = {}
    # common regexes
    patterns = {
        "psnr": r"psnr[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "ssim": r"ssim[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "mse": r"mse[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "latent_mse": r"latent[_\s]?mse[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "val_loss": r"val[_\s]?loss[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "train_loss": r"train[_\s]?loss[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "fps": r"fps[:=]\s*([0-9]+\.[0-9]+|[0-9]+)"
    }
    lower = text.lower()
    for k, pat in patterns.items():
        mm = re.search(pat, lower)
        if mm:
            try:
                val = float(mm.group(1))
                metrics[k] = val
            except Exception:
                metrics[k] = mm.group(1)
    return metrics

def locate_default_candidates():
    """
    Heuristic list of candidate modules / files to try to import/run.
    """
    candidates = []
    # common file names in this repo (based on prior scan)
    for name in ("create_visualizations.py", "create_visualization.py", "evaluate.py", "eval.py", "AI.py", "vae_processor.py", "run_eval.py"):
        if os.path.exists(name):
            candidates.append(name)
    # also attempt module-style names
    for mod in ("create_visualizations", "vae_processor", "AI", "eval", "evaluate"):
        candidates.append(mod)
    return candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-cmd", type=str, default=None, help="Command to run to produce metrics (shell).")
    parser.add_argument("--out", type=str, default="before_metrics.json", help="Output JSON file path.")
    parser.add_argument("--candidates", type=str, nargs="*", default=None, help="Candidate module paths to try importing.")
    args = parser.parse_args()

    out_path = args.out
    candidates = args.candidates or locate_default_candidates()

    metrics = {}

    # 1) Try import-and-run
    m, txt = try_import_and_run(candidates)
    if m:
        metrics.update(m)
    else:
        # 2) If run-cmd provided, try running it
        if args.run_cmd:
            print("Running user-provided command:", args.run_cmd)
            txt = run_subprocess(args.run_cmd)
            m2 = parse_metrics_from_text(txt)
            metrics.update(m2)
        else:
            # 3) try default candidates via subprocess (python file)
            for c in candidates:
                if os.path.exists(c):
                    cmd = f"python3 {c}"
                    print("Running candidate:", cmd)
                    txt = run_subprocess(cmd)
                    parsed = parse_metrics_from_text(txt)
                    if parsed:
                        metrics.update(parsed)
                        break

    # 4) ensure we have at least default keys with None
    final = DEFAULT_METRICS.copy()
    final.update(metrics)

    # Save to out_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print("Wrote metrics to", out_path)
    print(json.dumps(final, indent=2))

if __name__ == "__main__":
    main()
