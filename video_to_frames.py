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


def clear_directory(dir_path: Path) -> None:
    for entry in dir_path.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                entry.unlink(missing_ok=True)
            elif entry.is_dir():
                # Remove nested directories recursively
                clear_directory(entry)
                entry.rmdir()
        except Exception:
            pass


def extract_frames(video_path: Path, output_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = 0
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        index += 1
        out_name = output_dir / f"frame_{index:06d}.png"
        ok = cv2.imwrite(str(out_name), frame)
        if not ok:
            raise RuntimeError(f"Failed to write frame to {out_name}")
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Saved {frame_count} frames...")

    cap.release()
    return frame_count


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    root.update()

    video_file = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[
            ("Video files", ".mp4 .avi .mov .mkv .webm .m4v .mpg .mpeg"),
            ("All files", "*.*"),
        ],
    )

    if not video_file:
        print("No file selected. Exiting.")
        return

    video_path = Path(video_file)
    if not video_path.exists():
        messagebox.showerror("Error", f"Selected file does not exist:\n{video_path}")
        return

    # Output directory named 'source' next to this script
    output_dir = Path(__file__).with_name("source")
    output_dir.mkdir(exist_ok=True)

    # If directory is non-empty, ask to clear
    non_empty = any(output_dir.iterdir())
    if non_empty:
        resp = messagebox.askyesno(
            "Directory not empty",
            f"The output directory '\\source' already contains files.\n\n"
            f"Do you want to clear it before extracting frames?",
        )
        if resp:
            clear_directory(output_dir)
        else:
            # Continue writing new frames; warn user
            messagebox.showinfo("Notice", "Existing files will be kept. New frames will be added.")

    try:
        total = extract_frames(video_path, output_dir)
    except Exception as exc:
        messagebox.showerror("Extraction Failed", str(exc))
        return

    messagebox.showinfo("Done", f"Saved {total} frames to:\n{output_dir}")


if __name__ == "__main__":
    main()


