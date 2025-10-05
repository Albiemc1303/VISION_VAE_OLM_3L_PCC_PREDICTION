import json
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Deque

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as exc:  # pragma: no cover - environment-specific
    print("Failed to import tkinter:", exc)
    sys.exit(1)

try:
    import cv2  # type: ignore
    import numpy as np
except Exception:
    cv2 = None  # type: ignore
    np = None  # type: ignore

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = None  # type inore
    ImageTk = None  # type ignore

# Assuming AIPipeline is importable from the AI module
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
    # Added configuration for heatmap dynamic range tracking
    "heatmap_ema_alpha": 0.01,
}


def load_config(config_path: Path) -> Dict[str, Any]:
    # ... (Configuration loading logic remains the same)
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        config: Dict[str, Any] = DEFAULT_CONFIG.copy()
        if isinstance(data, dict):
            # Only update keys present in DEFAULT_CONFIG
            config.update({k: data[k] for k in DEFAULT_CONFIG.keys() if k in data})
        return config
    except Exception:
        return DEFAULT_CONFIG.copy()


class ProcessingThread(threading.Thread):
    """
    Worker thread to handle the heavy AI processing and training, 
    decoupling it from the tkinter GUI thread.
    """
    def __init__(self, ai_pipeline: Any, frame_queue: 'queue.Queue', result_queue: 'queue.Queue'):
        super().__init__()
        self.ai_pipeline = ai_pipeline
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self._stop_event = threading.Event()
        self.daemon = True # Allows the thread to exit when the main program exits

    def run(self):
        while not self._stop_event.is_set():
            try:
                # Wait for a new frame (non-blocking if queue is empty after timeout)
                # Use a small timeout to check the stop event periodically
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Heavy lifting: process, predict, and train
                ai_frame, metrics = self.ai_pipeline.process_frame(frame)
                
                # Safely put results onto the result queue for the main thread
                # We put the frame and the metrics dictionary
                self.result_queue.put((ai_frame, metrics))

            except Exception as e:
                # Pass error back to the main thread
                self.result_queue.put({"error": str(e)})
            finally:
                self.frame_queue.task_done()

    def stop(self):
        self._stop_event.set()


class CameraApp:
    def __init__(self, root: tk.Tk, config: Dict[str, Any]):
        if cv2 is None or np is None:
            messagebox.showerror("Missing Dependency", "opencv-python and numpy are required")
            root.destroy()
            return
        if Image is None or ImageTk is None:
            messagebox.showerror("Missing Dependency", "Pillow is required")
            root.destroy()
            return

        self.root = root
        self.config = config
        self.root.title(str(config.get("title", DEFAULT_CONFIG["title"])))

        # State
        self.video_capture: 'Optional[Any]' = None
        self.is_streaming: bool = False
        self.photo_image: 'Optional[Any]' = None
        self.ai_photo_image: 'Optional[Any]' = None
        self.after_job: Optional[str] = None
        
        # Multithreading setup
        self.frame_queue: 'queue.Queue' = queue.Queue(maxsize=1) # Queue for fresh frames to AI
        self.result_queue: 'queue.Queue' = queue.Queue(maxsize=1) # Queue for AI results
        self.ai_thread: 'Optional[ProcessingThread]' = None

        # AI Pipeline
        self.ai_pipeline: 'Optional[Any]' = None
        self.ai_enabled = config.get("enable_ai", True)
        self._initialize_ai()
        
        # State for dynamic heatmap scaling (EMA tracking)
        # Tracks min/max for pattern, compressed, central vectors
        self._ema_max: Dict[str, float] = {'pattern': 1e-4, 'compressed': 1e-4, 'central': 1e-4}
        self._ema_min: Dict[str, float] = {'pattern': -1e-4, 'compressed': -1e-4, 'central': -1e-4}
        self._ema_alpha = self.config.get("heatmap_ema_alpha", 0.01)

        # UI
        self._build_ui()

        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _initialize_ai(self) -> None:
        """Initialize AI pipeline and worker thread"""
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
            
            # Start the dedicated worker thread
            self.ai_thread = ProcessingThread(self.ai_pipeline, self.frame_queue, self.result_queue)
            self.ai_thread.start()
            
            print("AI Pipeline and Processing Thread initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize AI Pipeline: {e}")
            self.ai_enabled = False
            self.ai_pipeline = None
            self.ai_thread = None

    def _build_ui(self) -> None:
        # ... (UI building logic remains largely the same, but with slight grid refactoring for clarity)
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
            debug_frame = ttk.LabelFrame(container, text="AI Debug Vectors")
            debug_frame.grid(row=1, column=0, columnspan=4, pady=(6, 0), sticky="ew")
            
            # Configure canvas sizes based on expected dimensions (assuming 512 for Pattern/Central, 256 for Compressed)
            # These are now *hints*; they will resize horizontally based on grid weight
            self.pattern_canvas = tk.Canvas(debug_frame, height=20, bg="#111")
            self.pattern_canvas.grid(row=0, column=0, padx=4, pady=2, sticky="ew")
            self.compressed_canvas = tk.Canvas(debug_frame, height=20, bg="#111")
            self.compressed_canvas.grid(row=1, column=0, padx=4, pady=2, sticky="ew")
            self.central_canvas = tk.Canvas(debug_frame, height=20, bg="#111")
            self.central_canvas.grid(row=2, column=0, padx=4, pady=2, sticky="ew")
            
            # Labels
            ttk.Label(debug_frame, text="Pattern").grid(row=0, column=1, sticky="w", padx=4)
            ttk.Label(debug_frame, text="Compressed").grid(row=1, column=1, sticky="w", padx=4)
            ttk.Label(debug_frame, text="Central").grid(row=2, column=1, sticky="w", padx=4)
            
            debug_frame.columnconfigure(0, weight=1) # Canvas gets the weight

            # Configure grid weights for split view
            video_frame.columnconfigure(0, weight=1)
            video_frame.columnconfigure(1, weight=1)
            video_frame.rowconfigure(0, weight=1)
            camera_frame.columnconfigure(0, weight=1)
            camera_frame.rowconfigure(0, weight=1)
            ai_frame.columnconfigure(0, weight=1)
            ai_frame.rowconfigure(0, weight=1)
            
            # Status label for AI metrics
            self.status_label = ttk.Label(container, text="AI Status: Initializing...")
            self.status_label.grid(row=2, column=0, columnspan=4, pady=(4, 0), sticky="w")
        else:
            # Single camera view
            self.video_label = ttk.Label(container)
            self.video_label.grid(row=0, column=0, columnspan=3, sticky="nsew")

        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=1)
        container.columnconfigure(3, weight=1)
        container.rowconfigure(0, weight=1)
        
        # Controls (moved to row 3 if AI is enabled)
        row_control = 3 if self.ai_enabled else 1
        
        self.start_button = ttk.Button(container, text="Start", command=self.start_stream)
        self.stop_button = ttk.Button(container, text="Stop", command=self.stop_stream, state=tk.DISABLED)
        self.reset_button = ttk.Button(container, text="Reset AI", command=self.reset_ai, state=tk.DISABLED if not self.ai_enabled else tk.NORMAL)
        self.quit_button = ttk.Button(container, text="Quit", command=self._on_close)

        self.start_button.grid(row=row_control, column=0, pady=(8, 0), sticky="ew")
        self.stop_button.grid(row=row_control, column=1, pady=(8, 0), sticky="ew")
        self.reset_button.grid(row=row_control, column=2, pady=(8, 0), sticky="ew")
        self.quit_button.grid(row=row_control, column=3, pady=(8, 0), sticky="ew")


    def reset_ai(self) -> None:
        """Reset AI pipeline state"""
        if self.ai_pipeline is not None:
            self.ai_pipeline.reset_state()
            # Reset EMA tracking for heatmaps too
            self._ema_max = {'pattern': 1e-4, 'compressed': 1e-4, 'central': 1e-4}
            self._ema_min = {'pattern': -1e-4, 'compressed': -1e-4, 'central': -1e-4}
            print("AI Pipeline state reset")
            if hasattr(self, 'status_label'):
                self.status_label.config(text="AI Status: State Reset")

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
        
        # Verify actual resolution
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != width or actual_h != height:
            print(f"Warning: Requested resolution {width}x{height} not supported. Using {actual_w}x{actual_h}.")

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
        # Start the frame update loop
        self._update_frame()
        # Start checking for AI results
        if self.ai_enabled:
            self._check_ai_results()

    def stop_stream(self) -> None:
        # ... (Stop logic remains the same)
        if not self.is_streaming and self.video_capture is None:
            return
        self.is_streaming = False
        
        # Cancel main frame update loop
        if self.after_job is not None:
            try:
                self.root.after_cancel(self.after_job)
            except Exception:
                pass
            self.after_job = None
        
        # Clear result checking loop (if running)
        if hasattr(self, '_ai_check_job') and self._ai_check_job is not None:
            try:
                self.root.after_cancel(self._ai_check_job)
            except Exception:
                pass
            del self._ai_check_job

        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None

        # Clear display
        self.video_label.configure(image="")
        self.photo_image = None
        if hasattr(self, 'ai_video_label'):
            self.ai_video_label.configure(image="")
            self.ai_photo_image = None

        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        
        if self.ai_enabled and hasattr(self, 'status_label'):
            self.status_label.config(text="AI Status: Stream Stopped")

    def _update_frame(self) -> None:
        """Reads frame, displays it, and pushes a copy to the AI thread."""
        if not self.is_streaming or self.video_capture is None:
            return

        ret, frame = self.video_capture.read()
        if not ret or frame is None:
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

        # Push frame copy to the AI thread (non-blocking)
        if self.ai_enabled and self.ai_thread and self.ai_thread.is_alive():
            # Only put if the queue is not full to avoid blocking the main thread
            if self.frame_queue.empty():
                try:
                    # Pass a copy to avoid threading issues if cv2 reuses buffers
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    # Occurs if the AI thread is too slow; just drop the frame
                    pass

        # Schedule next frame (fast update for responsive camera feed)
        self.after_job = self.root.after(10, self._update_frame)


    def _check_ai_results(self) -> None:
        """Pulls results from the AI thread's result queue and updates UI/metrics."""
        if not self.is_streaming:
            return

        has_new_result = False
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                self.result_queue.task_done()
                has_new_result = True
            except queue.Empty:
                break
            
            if isinstance(result, dict) and 'error' in result:
                # AI error handling
                self.ai_enabled = False # Disable AI processing
                if hasattr(self, 'status_label'):
                    self.status_label.config(text=f"AI Error (Disabled): {result['error'][:50]}...")
                print(f"AI Processing Halted due to Error: {result['error']}")
                continue

            # Unpack results: (ai_frame, metrics)
            ai_frame, metrics = result
            
            # --- UI Updates ---
            if ai_frame is not None:
                # Resize the predicted frame to match the camera frame size for display consistency
                display_w = self.video_label.winfo_width() or self.config.get("frame_width", 640)
                display_h = self.video_label.winfo_height() or self.config.get("frame_height", 480)
                
                ai_frame_resized = cv2.resize(ai_frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
                
                ai_frame_rgb = cv2.cvtColor(ai_frame_resized, cv2.COLOR_BGR2RGB)
                ai_img = Image.fromarray(ai_frame_rgb)
                self.ai_photo_image = ImageTk.PhotoImage(image=ai_img)
                self.ai_video_label.configure(image=self.ai_photo_image)
            
            # Update status
            if hasattr(self, 'status_label') and metrics.get("status") == "processed":
                fps = metrics.get("fps", 0)
                frame_count = metrics.get("frame_count", 0)
                loss_ema_long = metrics.get("loss_ema_long", 0) or 0
                grad_norm_ema = metrics.get("grad_norm_ema", 0) or 0
                
                status_text = (
                    f"AI Status: FPS={fps:.1f}, Frames={frame_count}, "
                    f"Loss(L-EMA)={loss_ema_long:.4f}, GradNorm(EMA)={grad_norm_ema:.2e}"
                )
                self.status_label.config(text=status_text)

            # Draw debug vectors as heatmaps
            dbg = metrics.get("debug_vectors", {})
            self._draw_vector_heatmap(self.pattern_canvas, dbg.get("pattern"), 'pattern')
            self._draw_vector_heatmap(self.compressed_canvas, dbg.get("compressed"), 'compressed')
            self._draw_vector_heatmap(self.central_canvas, dbg.get("central"), 'central')
        
        # Schedule the next check (fast check)
        self._ai_check_job = self.root.after(100, self._check_ai_results) # 10 times per second check


    def _draw_vector_heatmap(self, canvas: tk.Canvas, vec: Optional[Any], key: str) -> None:
        """
        Draws a heatmap using EMA-based (absolute) normalization for stable visualization.
        """
        if canvas is None or vec is None or np is None:
            return
        
        try:
            canvas.delete("all")
            data = np.array(vec).flatten().astype(np.float32)
            if data.size == 0:
                return

            # --- EMA Tracking for Absolute Normalization ---
            # Update global min/max for dynamic range
            current_max = data.max()
            current_min = data.min()
            alpha = self._ema_alpha
            
            # Update EMA
            self._ema_max[key] = (1 - alpha) * self._ema_max[key] + alpha * current_max
            self._ema_min[key] = (1 - alpha) * self._ema_min[key] + alpha * current_min
            
            # Use the EMA values for normalization
            vmin = self._ema_min[key]
            vmax = self._ema_max[key]
            rng = max(vmax - vmin, 1e-6)
            
            # Normalize to [0,1], clamping values outside the EMA range
            norm = np.clip((data - vmin) / rng, 0.0, 1.0)
            
            # --- Drawing ---
            width = canvas.winfo_width() # Use actual widget width
            height = canvas.winfo_height() or 20
            
            n = data.size
            bar_w = max(width // n, 1) # Ensure bar_w is at least 1 pixel

            for i in range(min(n, width // bar_w)):
                val = float(norm[i])
                # Simple colormap from dark to bright green
                g = int(50 + 205 * val)
                color = f"#{0:02x}{g:02x}{0:02x}"
                x0 = i * bar_w
                x1 = x0 + bar_w
                canvas.create_rectangle(x0, 0, x1, height, fill=color, width=0)
        except Exception as e:
            # Handle drawing errors gracefully (e.g., if canvas size is 0)
            print(f"Error drawing heatmap for {key}: {e}")
            pass


    def _on_close(self) -> None:
        try:
            self.stop_stream()
            # Stop the AI worker thread gracefully
            if self.ai_thread is not None:
                self.ai_thread.stop()
                self.ai_thread.join(timeout=2) # Wait for thread to finish
                if self.ai_thread.is_alive():
                    print("Warning: AI thread failed to terminate gracefully.")

            # Cleanup AI pipeline resources
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