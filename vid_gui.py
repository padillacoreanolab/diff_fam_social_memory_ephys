import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import subprocess
import os
import random
import threading


class VideoColorEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Color Correction Tool")
        self.root.geometry("1000x700")

        # Variables
        self.video_path = ""
        self.original_frames = []
        self.preview_frames = []
        self.photo_refs = []  # Keep references to prevent garbage collection

        # Color adjustment values
        self.brightness = tk.DoubleVar(value=1.0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.gamma = tk.DoubleVar(value=1.0)

        self.setup_gui()

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Video File", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))

        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Color Adjustments", padding="10")
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        controls_frame.columnconfigure(1, weight=1)

        # Brightness slider
        ttk.Label(controls_frame, text="Brightness:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        brightness_frame = ttk.Frame(controls_frame)
        brightness_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        brightness_frame.columnconfigure(0, weight=1)

        brightness_scale = tk.Scale(
            brightness_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.brightness,
            command=self.update_preview,
        )
        brightness_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.brightness_val = ttk.Label(brightness_frame, text="1.0")
        self.brightness_val.grid(row=0, column=1, padx=(5, 0))

        # Contrast slider
        ttk.Label(controls_frame, text="Contrast:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        contrast_frame = ttk.Frame(controls_frame)
        contrast_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        contrast_frame.columnconfigure(0, weight=1)

        contrast_scale = tk.Scale(
            contrast_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.contrast,
            command=self.update_preview,
        )
        contrast_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.contrast_val = ttk.Label(contrast_frame, text="1.0")
        self.contrast_val.grid(row=0, column=1, padx=(5, 0))

        # Gamma slider
        ttk.Label(controls_frame, text="Gamma:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        gamma_frame = ttk.Frame(controls_frame)
        gamma_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        gamma_frame.columnconfigure(0, weight=1)

        gamma_scale = tk.Scale(
            gamma_frame,
            from_=0.1,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.gamma,
            command=self.update_preview,
        )
        gamma_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.gamma_val = ttk.Label(gamma_frame, text="1.0")
        self.gamma_val.grid(row=0, column=1, padx=(5, 0))

        # Reset button
        ttk.Button(controls_frame, text="Reset", command=self.reset_values).grid(
            row=3, column=0, columnspan=2, pady=(10, 0)
        )

        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Preview (3 Random Frames)", padding="10")
        preview_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Frame for preview images
        self.preview_container = ttk.Frame(preview_frame)
        self.preview_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status and export frame
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(bottom_frame, text="Ready")
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        ttk.Button(bottom_frame, text="Export Video", command=self.export_video).grid(row=0, column=1, padx=(10, 0))

        # Configure preview container for 3 images
        for i in range(3):
            self.preview_container.columnconfigure(i, weight=1)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All files", "*.*")],
        )

        if file_path:
            self.video_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.load_sample_frames()

    def load_sample_frames(self):
        if not self.video_path:
            return

        self.status_label.config(text="Loading sample frames...")
        self.root.update()

        try:
            # Open video and get basic info
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                messagebox.showerror("Error", "Could not read video file")
                return

            # Select 3 random frame positions
            frame_positions = random.sample(range(0, total_frames), min(3, total_frames))

            self.original_frames = []

            for pos in frame_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    # Resize for preview (maintain aspect ratio)
                    pil_image.thumbnail((200, 150), Image.Resampling.LANCZOS)
                    self.original_frames.append(pil_image)

            cap.release()

            if self.original_frames:
                self.update_preview()
                self.status_label.config(text=f"Loaded {len(self.original_frames)} sample frames")
            else:
                messagebox.showerror("Error", "Could not extract sample frames")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")
            self.status_label.config(text="Error loading video")

    def apply_color_corrections(self, image):
        """Apply brightness, contrast, and gamma corrections to a PIL image"""
        # Apply brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.brightness.get())

        # Apply contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.contrast.get())

        # Apply gamma correction
        gamma_val = self.gamma.get()
        if gamma_val != 1.0:
            # Convert to numpy array for gamma correction
            img_array = np.array(image)
            # Apply gamma correction
            img_array = np.power(img_array / 255.0, 1.0 / gamma_val) * 255.0
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)

        return image

    def update_preview(self, *args):
        """Update the preview frames with current color adjustments"""
        if not self.original_frames:
            return

        # Update value labels
        self.brightness_val.config(text=f"{self.brightness.get():.1f}")
        self.contrast_val.config(text=f"{self.contrast.get():.1f}")
        self.gamma_val.config(text=f"{self.gamma.get():.1f}")

        # Clear existing preview images
        for widget in self.preview_container.winfo_children():
            widget.destroy()
        self.photo_refs.clear()

        # Apply corrections and display
        for i, original_frame in enumerate(self.original_frames):
            # Apply color corrections
            corrected_frame = self.apply_color_corrections(original_frame.copy())

            # Convert to PhotoImage for display
            photo = ImageTk.PhotoImage(corrected_frame)
            self.photo_refs.append(photo)  # Keep reference

            # Create label to display image
            label = ttk.Label(self.preview_container, image=photo)
            label.grid(row=0, column=i, padx=5, pady=5)

    def reset_values(self):
        """Reset all values to defaults"""
        self.brightness.set(1.0)
        self.contrast.set(1.0)
        self.gamma.set(1.0)

    def export_video(self):
        """Export the video with applied color corrections using ffmpeg"""
        if not self.video_path:
            messagebox.showerror("Error", "No video file selected")
            return

        # Get output file path
        output_path = filedialog.asksaveasfilename(
            title="Save Processed Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")],
        )

        if not output_path:
            return

        # Create ffmpeg command
        self.run_ffmpeg_export(output_path)

    def run_ffmpeg_export(self, output_path):
        """Run ffmpeg export in a separate thread"""

        def export_thread():
            try:
                self.status_label.config(text="Exporting video...")
                self.root.update()

                # Build ffmpeg filter string
                brightness_val = self.brightness.get()
                contrast_val = self.contrast.get()
                gamma_val = self.gamma.get()

                # Convert values to ffmpeg format
                # FFmpeg brightness: -1.0 to 1.0 (we need to convert from 0.1-3.0 to -1.0-1.0)
                ffmpeg_brightness = brightness_val - 1.0
                # FFmpeg contrast: 0.0 to 4.0 (direct mapping)
                ffmpeg_contrast = contrast_val
                # FFmpeg gamma: 0.1 to 10.0 (direct mapping)
                ffmpeg_gamma = gamma_val

                # Build filter chain
                filters = []
                if brightness_val != 1.0 or contrast_val != 1.0:
                    filters.append(f"eq=brightness={ffmpeg_brightness}:contrast={ffmpeg_contrast}")
                if gamma_val != 1.0:
                    filters.append(f"gamma={ffmpeg_gamma}")

                filter_string = ",".join(filters) if filters else "null"

                # FFmpeg command
                cmd = [
                    "ffmpeg",
                    "-i",
                    self.video_path,
                    "-vf",
                    filter_string,
                    "-c:a",
                    "copy",  # Copy audio without re-encoding
                    "-y",  # Overwrite output file
                    output_path,
                ]

                # Run ffmpeg
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    self.status_label.config(text=f"Export completed: {os.path.basename(output_path)}")
                    messagebox.showinfo("Success", f"Video exported successfully to:\n{output_path}")
                else:
                    error_msg = result.stderr if result.stderr else "Unknown error"
                    self.status_label.config(text="Export failed")
                    messagebox.showerror("Export Error", f"FFmpeg error:\n{error_msg}")

            except FileNotFoundError:
                messagebox.showerror("Error", "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
                self.status_label.config(text="FFmpeg not found")
            except Exception as e:
                messagebox.showerror("Error", f"Export error: {str(e)}")
                self.status_label.config(text="Export error")

        # Run in separate thread to prevent GUI freezing
        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()


def main():
    root = tk.Tk()
    app = VideoColorEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
