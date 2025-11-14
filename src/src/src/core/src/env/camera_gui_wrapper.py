# src/env/camera_gui_wrapper.py
"""
Camera GUI wrapper that registers module and tries to delegate to repo's camera_gui.py.
If absent, provides a no-op stub.
"""
import importlib.util
import os
import sys
from src.meta.ontology import ModuleDescriptor, ontology

MODULE_NAME = "camera_gui"
desc = ModuleDescriptor(
    name=MODULE_NAME,
    role="Camera interface / visualization bridge",
    inputs={"camera_device": {"dtype":"int"}},
    outputs={"display": {"dtype":"frame"}}
)
ontology.register_module(desc)

def try_import_original():
    candidates = ["camera_gui.py", "src/camera_gui.py", "camera_gui"]
    for c in candidates:
        if os.path.exists(c):
            spec = importlib.util.spec_from_file_location("orig_cam", c)
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
    def launch_ui(*args, **kwargs):
        if hasattr(_orig, "main"):
            return _orig.main(*args, **kwargs)
        if hasattr(_orig, "launch_ui"):
            return _orig.launch_ui(*args, **kwargs)
        raise RuntimeError("Original camera_gui found but no main/launch function.")
else:
    def launch_ui(*args, **kwargs):
        print("camera_gui not found. Running stub UI (no-op).")
        return
