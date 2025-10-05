import sys
import os
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception as exc:
    print("Failed to import tkinter:", exc)
    sys.exit(1)

try:
    import cv2
except Exception as exc:
    print("Failed to import OpenCV (cv2):", exc)
    sys.exit(1)


# --- Define OLM Target Rate for Scientific Consistency ---
# This matches the target FPS defined in the VAEProcessor component.
TARGET_FPS = 24


def clear_directory(dir_path: Path) -> None:
    """Recursively deletes all files and subdirectories within the given path."""
    for entry in dir_path.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                entry.unlink(missing_ok=True)
            elif entry.is_dir():
                # Remove nested directories recursively
                clear_directory(entry)
                entry.rmdir()
        except Exception:
            # Silently ignore errors during cleanup
            pass


def extract_frames(video_path: Path, output_dir: Path) -> int:
    """
    Extracts frames from a video and resamples them to TARGET_FPS for OLM consistency.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    # 1. Determine resampling ratio
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
         # Fallback if FPS cannot be read (e.g., some containers)
         native_fps = 30.0 
         print(f"Warning: Could not read video FPS. Assuming 30.0 FPS for extraction.")
         
    frame_skip_ratio = native_fps / TARGET_FPS
    
    if frame_skip_ratio < 1.0:
        # If native FPS is less than target FPS, we must upsample (interpolate), 
        # which is usually undesirable for LSTMs. For simplicity, we sample every frame.
        frame_skip_ratio = 1.0
        print(f"Warning: Native FPS ({native_fps:.1f}) is less than target ({TARGET_FPS}). Sampling every frame.")
    
    print(f"Video Native FPS: {native_fps:.2f}. Target FPS: {TARGET_FPS}. Skip Ratio: {frame_skip_ratio:.2f}")

    # 2. Resampling loop
    extracted_frame_count = 0
    save_index = 0
    current_frame_pos = 0.0 # Floating point position in time
    
    while True:
        # Move the video capture to the nearest frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame_pos))
        ret, frame = cap.read()
        
        if not ret or frame is None:
            break
            
        save_index += 1
        out_name = output_dir / f"frame_{save_index:06d}.png"
        ok = cv2.imwrite(str(out_name), frame)
        
        if not ok:
            raise RuntimeError(f"Failed to write frame to {out_name}")
            
        extracted_frame_count += 1
        
        # Increment to the next required frame position
        current_frame_pos += frame_skip_ratio
        
        if extracted_frame_count % 100 == 0:
            print(f"Saved {extracted_frame_count} frames...")

    cap.release()
    return extracted_frame_count


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    
    try:
        video_file = filedialog.askopenfilename(
            title="Select video file for OLM data generation",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.mpg *.mpeg"),
                ("All files", "*.*"),
            ],
        )
    except Exception:
        # Catch exception if dialog fails (e.g., no display)
        messagebox.showerror("Error", "Could not open file dialog. Check environment.")
        return
        
    root.update()

    if not video_file:
        print("No file selected. Exiting.")
        return

    video_path = Path(video_file)
    if not video_path.exists():
        messagebox.showerror("Error", f"Selected file does not exist:\n{video_path}")
        return

    # Output directory named 'source' next to this script
    script_path = Path(sys.argv[0] if getattr(sys, 'frozen', False) else __file__)
    output_dir = script_path.with_name("source")
    output_dir.mkdir(exist_ok=True)

    # If directory is non-empty, ask to clear
    non_empty = any(output_dir.iterdir())
    if non_empty:
        resp = messagebox.askyesno(
            "Directory not empty",
            f"The output directory '{output_dir.name}' already contains files.\n\n"
            f"Do you want to clear it before extracting frames?",
        )
        if resp:
            clear_directory(output_dir)
        else:
            # Warn user if files are kept
            messagebox.showinfo("Notice", "Existing files will be kept. New frames will be added, which may disrupt training data order.")

    try:
        total = extract_frames(video_path, output_dir)
    except Exception as exc:
        messagebox.showerror("Extraction Failed", str(exc))
        return

    messagebox.showinfo("Done", f"Successfully saved {total} frames (resampled to {TARGET_FPS} FPS) to:\n{output_dir}")


if __name__ == "__main__":
    main()