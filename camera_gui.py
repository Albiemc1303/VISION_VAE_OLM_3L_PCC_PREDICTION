import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as exc:  # pragma: no cover - environment-specific
    print("Failed to import tkinter:", exc)
    sys.exit(1)

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

try:
    from AI import AIPipeline
except Exception:
    AIPipeline = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "mirror_preview": True,
    "title": "Camera Preview with AI",
    "enable_ai": True,
    "vae_path": "vae",
    "ai_config_path": "ai_config.json",
}


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        config: Dict[str, Any] = DEFAULT_CONFIG.copy()
        if isinstance(data, dict):
            config.update({k: data[k] for k in DEFAULT_CONFIG.keys() if k in data})
        return config
    except Exception:
        return DEFAULT_CONFIG.copy()


class CameraApp:
    def __init__(self, root: tk.Tk, config: Dict[str, Any]):
        if cv2 is None:
            messagebox.showerror("Missing Dependency", "opencv-python is required (pip install opencv-python)")
            root.destroy()
            return
        if Image is None or ImageTk is None:
            messagebox.showerror("Missing Dependency", "Pillow is required (pip install Pillow)")
            root.destroy()
            return

        self.root = root
        self.config = config
        self.root.title(str(config.get("title", DEFAULT_CONFIG["title"])))

        # State
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.is_streaming: bool = False
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.ai_photo_image: Optional[ImageTk.PhotoImage] = None
        self.after_job: Optional[str] = None

        # AI Pipeline
        self.ai_pipeline: Optional[AIPipeline] = None
        self.ai_enabled = config.get("enable_ai", True)
        self._initialize_ai()

        # UI
        self._build_ui()

        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _initialize_ai(self) -> None:
        """Initialize AI pipeline if enabled"""
        if not self.ai_enabled or AIPipeline is None:
            print("AI Pipeline disabled or not available")
            return
        
        try:
            vae_path = Path(self.config.get("vae_path", "vae"))
            ai_config_path = Path(self.config.get("ai_config_path", "ai_config.json"))
            
            if not vae_path.exists():
                print(f"VAE path {vae_path} does not exist. AI Pipeline disabled.")
                self.ai_enabled = False
                return
            
            print("Initializing AI Pipeline...")
            self.ai_pipeline = AIPipeline(vae_path, ai_config_path)
            print("AI Pipeline initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize AI Pipeline: {e}")
            self.ai_enabled = False
            self.ai_pipeline = None

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=8)
        container.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Video display areas
        if self.ai_enabled:
            # Split view: Camera and AI prediction
            video_frame = ttk.Frame(container)
            video_frame.grid(row=0, column=0, columnspan=4, sticky="nsew")
            
            # Camera feed
            camera_frame = ttk.LabelFrame(video_frame, text="Camera Feed")
            camera_frame.grid(row=0, column=0, padx=(0, 4), sticky="nsew")
            self.video_label = ttk.Label(camera_frame)
            self.video_label.grid(row=0, column=0, sticky="nsew")
            
            # AI prediction
            ai_frame = ttk.LabelFrame(video_frame, text="AI Prediction")
            ai_frame.grid(row=0, column=1, padx=(4, 0), sticky="nsew")
            self.ai_video_label = ttk.Label(ai_frame)
            self.ai_video_label.grid(row=0, column=0, sticky="nsew")

            # Debug vectors/heatmaps
            debug_frame = ttk.LabelFrame(video_frame, text="AI Debug Vectors")
            debug_frame.grid(row=1, column=0, columnspan=2, pady=(6, 0), sticky="nsew")
            self.pattern_canvas = tk.Canvas(debug_frame, width=256, height=40, bg="#111")
            self.pattern_canvas.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
            self.compressed_canvas = tk.Canvas(debug_frame, width=256, height=40, bg="#111")
            self.compressed_canvas.grid(row=1, column=0, padx=4, pady=4, sticky="ew")
            self.central_canvas = tk.Canvas(debug_frame, width=256, height=40, bg="#111")
            self.central_canvas.grid(row=2, column=0, padx=4, pady=4, sticky="ew")
            
            # Labels
            ttk.Label(debug_frame, text="Pattern").grid(row=0, column=1, sticky="w")
            ttk.Label(debug_frame, text="Compressed").grid(row=1, column=1, sticky="w")
            ttk.Label(debug_frame, text="Central").grid(row=2, column=1, sticky="w")
            
            # Configure grid weights for split view
            video_frame.columnconfigure(0, weight=1)
            video_frame.columnconfigure(1, weight=1)
            video_frame.rowconfigure(0, weight=1)
            video_frame.rowconfigure(1, weight=0)
            camera_frame.columnconfigure(0, weight=1)
            camera_frame.rowconfigure(0, weight=1)
            ai_frame.columnconfigure(0, weight=1)
            ai_frame.rowconfigure(0, weight=1)
            
            # Status label for AI metrics
            self.status_label = ttk.Label(container, text="AI Status: Initializing...")
            self.status_label.grid(row=1, column=0, columnspan=4, pady=(4, 0))
        else:
            # Single camera view
            self.video_label = ttk.Label(container)
            self.video_label.grid(row=0, column=0, columnspan=3, sticky="nsew")

        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        # Controls
        self.start_button = ttk.Button(container, text="Start", command=self.start_stream)
        self.stop_button = ttk.Button(container, text="Stop", command=self.stop_stream, state=tk.DISABLED)
        self.reset_button = ttk.Button(container, text="Reset AI", command=self.reset_ai)
        self.quit_button = ttk.Button(container, text="Quit", command=self._on_close)

        start_col = 0 if not self.ai_enabled else 0
        stop_col = 1 if not self.ai_enabled else 1
        reset_col = 2 if not self.ai_enabled else 2
        quit_col = 2 if not self.ai_enabled else 3

        self.start_button.grid(row=2 if self.ai_enabled else 1, column=start_col, pady=(8, 0), sticky="ew")
        self.stop_button.grid(row=2 if self.ai_enabled else 1, column=stop_col, pady=(8, 0), sticky="ew")
        self.reset_button.grid(row=2 if self.ai_enabled else 1, column=reset_col, pady=(8, 0), sticky="ew")
        self.quit_button.grid(row=2 if self.ai_enabled else 1, column=quit_col, pady=(8, 0), sticky="ew")

    def reset_ai(self) -> None:
        """Reset AI pipeline state"""
        if self.ai_pipeline is not None:
            self.ai_pipeline.reset_state()
            print("AI Pipeline state reset")
            if hasattr(self, 'status_label'):
                self.status_label.config(text="AI Status: Reset")

    def _open_capture(self) -> bool:
        if self.video_capture is not None:
            return True
        camera_index = int(self.config.get("camera_index", 0))
        width = int(self.config.get("frame_width", 640))
        height = int(self.config.get("frame_height", 480))

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Unable to open camera index {camera_index}.")
            return False

        # Try to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.video_capture = cap
        return True

    def start_stream(self) -> None:
        if self.is_streaming:
            return
        if not self._open_capture():
            return
        self.is_streaming = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self._update_frame()

    def stop_stream(self) -> None:
        if not self.is_streaming and self.video_capture is None:
            return
        self.is_streaming = False
        if self.after_job is not None:
            try:
                self.root.after_cancel(self.after_job)
            except Exception:
                pass
            self.after_job = None

        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None

        # Clear display
        self.video_label.configure(image="")
        self.photo_image = None

        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def _update_frame(self) -> None:
        if not self.is_streaming or self.video_capture is None:
            return
        ret, frame = self.video_capture.read()
        if not ret or frame is None:
            # Stop on failure
            self.stop_stream()
            messagebox.showwarning("Camera", "Failed to read frame. Stopped.")
            return

        # Mirror if requested
        if bool(self.config.get("mirror_preview", True)):
            frame = cv2.flip(frame, 1)

        # Display camera frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.photo_image = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self.photo_image)

        # Process with AI if enabled
        if self.ai_enabled and self.ai_pipeline is not None:
            try:
                ai_frame, metrics = self.ai_pipeline.process_frame(frame)
                
                if ai_frame is not None:
                    # Display AI prediction
                    ai_frame_rgb = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
                    ai_img = Image.fromarray(ai_frame_rgb)
                    self.ai_photo_image = ImageTk.PhotoImage(image=ai_img)
                    self.ai_video_label.configure(image=self.ai_photo_image)
                
                # Update status
                if hasattr(self, 'status_label') and metrics.get("status") == "processed":
                    fps = metrics.get("fps", 0)
                    frame_count = metrics.get("frame_count", 0)
                    comp_loss = metrics.get("compression_loss", 0)
                    central_loss = metrics.get("central_loss", 0) or 0
                    status_text = f"AI Status: FPS={fps:.1f}, Frames={frame_count}, CompLoss={comp_loss:.4f}, CentLoss={central_loss:.4f}"
                    self.status_label.config(text=status_text)

                # Draw debug vectors as heatmaps
                dbg = metrics.get("debug_vectors", {}) if isinstance(metrics, dict) else {}
                self._draw_vector_heatmap(self.pattern_canvas, dbg.get("pattern"))
                self._draw_vector_heatmap(self.compressed_canvas, dbg.get("compressed"))
                self._draw_vector_heatmap(self.central_canvas, dbg.get("central"))
                    
            except Exception as e:
                print(f"AI processing error: {e}")
                if hasattr(self, 'status_label'):
                    self.status_label.config(text=f"AI Error: {str(e)[:50]}...")

        # Schedule next frame
        self.after_job = self.root.after(10, self._update_frame)  # ~100 FPS cap, actual depends on camera

    def _draw_vector_heatmap(self, canvas: tk.Canvas, vec: Optional[Any]) -> None:
        if canvas is None or vec is None:
            return
        try:
            import numpy as np  # local import to avoid hard dep at import time
            canvas.delete("all")
            data = np.array(vec).flatten()
            if data.size == 0:
                return
            # Normalize to [0,1]
            vmin, vmax = float(data.min()), float(data.max())
            rng = max(vmax - vmin, 1e-6)
            norm = (data - vmin) / rng
            # Draw bars across width
            width = int(canvas["width"]) if str(canvas["width"]).isdigit() else 256
            height = int(canvas["height"]) if str(canvas["height"]).isdigit() else 40
            n = data.size
            bar_w = max(width // n, 1)
            for i in range(min(n, width // bar_w)):
                val = float(norm[i])
                # Simple colormap from dark to bright green
                g = int(50 + 205 * val)
                color = f"#{0:02x}{g:02x}{0:02x}"
                x0 = i * bar_w
                x1 = x0 + bar_w
                canvas.create_rectangle(x0, 0, x1, height, fill=color, width=0)
        except Exception:
            pass

    def _on_close(self) -> None:
        try:
            self.stop_stream()
            # Cleanup AI pipeline
            if self.ai_pipeline is not None:
                self.ai_pipeline.cleanup()
        finally:
            self.root.destroy()


def main() -> None:
    config_path = Path(__file__).with_name("camera_config.json")
    config = load_config(config_path)

    root = tk.Tk()
    app = CameraApp(root, config)
    # If dependencies were missing, the app would destroy root in __init__
    if not isinstance(app, CameraApp):  # type: ignore[unreachable]
        return
    root.mainloop()


if __name__ == "__main__":
    main()


