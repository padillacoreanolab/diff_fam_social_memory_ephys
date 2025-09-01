import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageEnhance
import subprocess
import os
import random
import threading
import locale


class VideoColorEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Video Color Correction Tool")
        self.root.geometry("1400x900")  # Increased window size for more controls and before/after preview

        # Variables
        self.video_path = ""
        self.original_frames = []
        self.preview_frames = []
        self.photo_refs = []  # Keep references to prevent garbage collection
        self.output_filename = tk.StringVar(value="processed_video.mp4")  # Output filename variable

        # Color adjustment values - using real FFmpeg ranges
        self.brightness = tk.DoubleVar(value=0.0)  # FFmpeg range: -1.0 to 1.0
        self.contrast = tk.DoubleVar(value=1.0)    # FFmpeg range: 0.0 to 4.0
        self.gamma = tk.DoubleVar(value=1.0)       # FFmpeg range: 0.1 to 10.0
        self.exposure = tk.DoubleVar(value=0.0)    # FFmpeg range: -3.0 to 3.0

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

        # Output filename
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="5")
        output_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 10))
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Output filename:").grid(row=0, column=0, padx=(0, 10))
        output_entry = ttk.Entry(output_frame, textvariable=self.output_filename, width=50)
        output_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Color Adjustments", padding="10")
        controls_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        controls_frame.columnconfigure(1, weight=1)

        # Basic Controls
        ttk.Label(controls_frame, text="Basic Controls", font=("Arial", 9, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

        # Brightness slider
        ttk.Label(controls_frame, text="Brightness:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        brightness_frame = ttk.Frame(controls_frame)
        brightness_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        brightness_frame.columnconfigure(0, weight=1)

        brightness_scale = tk.Scale(
            brightness_frame,
            from_=-1.0,
            to=1.0,
            resolution=0.02,
            orient=tk.HORIZONTAL,
            variable=self.brightness,
            command=self.update_preview,
            length=400,  # Make slider longer
        )
        brightness_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.brightness_val = ttk.Label(brightness_frame, text="0.00")
        self.brightness_val.grid(row=0, column=1, padx=(5, 0))

        # Contrast slider
        ttk.Label(controls_frame, text="Contrast:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        contrast_frame = ttk.Frame(controls_frame)
        contrast_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        contrast_frame.columnconfigure(0, weight=1)

        contrast_scale = tk.Scale(
            contrast_frame,
            from_=0.0,
            to=4.0,
            resolution=0.02,
            orient=tk.HORIZONTAL,
            variable=self.contrast,
            command=self.update_preview,
            length=400,  # Make slider longer
        )
        contrast_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.contrast_val = ttk.Label(contrast_frame, text="1.00")
        self.contrast_val.grid(row=0, column=1, padx=(5, 0))

        # Gamma slider
        ttk.Label(controls_frame, text="Gamma:").grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        gamma_frame = ttk.Frame(controls_frame)
        gamma_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        gamma_frame.columnconfigure(0, weight=1)

        gamma_scale = tk.Scale(
            gamma_frame,
            from_=0.1,
            to=10.0,
            resolution=0.02,
            orient=tk.HORIZONTAL,
            variable=self.gamma,
            command=self.update_preview,
            length=400,  # Make slider longer
        )
        gamma_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.gamma_val = ttk.Label(gamma_frame, text="1.00")
        self.gamma_val.grid(row=0, column=1, padx=(5, 0))

        # Exposure slider
        ttk.Label(controls_frame, text="Exposure:").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        exposure_frame = ttk.Frame(controls_frame)
        exposure_frame.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        exposure_frame.columnconfigure(0, weight=1)

        exposure_scale = tk.Scale(
            exposure_frame,
            from_=-3.0,
            to=3.0,
            resolution=0.02,
            orient=tk.HORIZONTAL,
            variable=self.exposure,
            command=self.update_preview,
            length=400,
        )
        exposure_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.exposure_val = ttk.Label(exposure_frame, text="0.00")
        self.exposure_val.grid(row=0, column=1, padx=(5, 0))

        # Reset button and shuffle button
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Reset", command=self.reset_values).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Shuffle Frames", command=self.shuffle_frames).pack(side=tk.LEFT)

        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Preview (3 Random Frames)", padding="10")
        preview_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Frame for preview images
        self.preview_container = ttk.Frame(preview_frame)
        self.preview_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status and export frame
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(bottom_frame, text="Ready")
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        # Export buttons
        export_button_frame = ttk.Frame(bottom_frame)
        export_button_frame.grid(row=0, column=1, padx=(10, 0))
        
        ttk.Button(export_button_frame, text="Test Export (5s)", command=self.test_export).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_button_frame, text="Export Video", command=self.export_video).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_button_frame, text="Export Command", command=self.export_command).pack(side=tk.LEFT)

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
                    # Resize for preview (2x larger: 400x300 instead of 200x150)
                    pil_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
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
        """Apply brightness, contrast, gamma, and exposure corrections using FFmpeg filters for accuracy"""
        # Get current adjustment values
        brightness_val = self.brightness.get()
        contrast_val = self.contrast.get()
        gamma_val = self.gamma.get()
        exposure_val = self.exposure.get()
        
        # Check if we need any adjustments (use real defaults: brightness=0.0, contrast=1.0, gamma=1.0, exposure=0.0)
        if brightness_val == 0.0 and contrast_val == 1.0 and gamma_val == 1.0 and exposure_val == 0.0:
            return image
        
        try:
            # Create temporary files
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_input = os.path.join(temp_dir, f"preview_input_{random.randint(1000,9999)}.png")
            temp_output = os.path.join(temp_dir, f"preview_output_{random.randint(1000,9999)}.png")
            
            # Save input image
            image.save(temp_input, "PNG")
            
            # Use FFmpeg values directly (no mapping needed)
            brightness_str = f"{brightness_val:.3f}".replace(',', '.')
            contrast_str = f"{contrast_val:.3f}".replace(',', '.')
            gamma_str = f"{gamma_val:.3f}".replace(',', '.')
            exposure_str = f"{exposure_val:.3f}".replace(',', '.')
            
            # Build filter chain
            filters = []
            
            # Add exposure filter if needed
            if exposure_val != 0.0:
                filters.append(f"exposure=exposure={exposure_str}")
            
            # Add eq filter if needed
            if brightness_val != 0.0 or contrast_val != 1.0 or gamma_val != 1.0:
                filters.append(f"eq=brightness={brightness_str}:contrast={contrast_str}:gamma={gamma_str}")
            
            # Combine filters
            filter_string = ",".join(filters) if filters else ""
            
            # Run FFmpeg with the filters
            cmd = [
                "ffmpeg",
                "-i", temp_input,
                "-vf", filter_string,
                "-y",  # Overwrite output
                temp_output
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode == 0 and os.path.exists(temp_output):
                # Load the processed image
                processed_image = Image.open(temp_output)
                processed_copy = processed_image.copy()  # Make a copy since we'll delete the file
                processed_image.close()
                
                # Clean up temporary files
                try:
                    os.remove(temp_input)
                    os.remove(temp_output)
                except:
                    pass
                
                return processed_copy
            else:
                # FFmpeg failed, fallback to original PIL method
                print(f"FFmpeg preview failed: {result.stderr}")
                return self.apply_color_corrections_pil(image)
                
        except Exception as e:
            print(f"FFmpeg preview error: {e}")
            # Fallback to original PIL method
            return self.apply_color_corrections_pil(image)
    
    def apply_color_corrections_pil(self, image):
        """Fallback PIL method (original implementation) for basic adjustments only"""
        # Apply brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.0 + self.brightness.get())

        # Apply contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.0 + self.contrast.get())

        # Apply gamma correction
        gamma_val = 1.0 + self.gamma.get()
        if gamma_val != 1.0:
            # Convert to numpy array for gamma correction
            img_array = np.array(image)
            # Apply gamma correction
            img_array = np.power(img_array / 255.0, 1.0 / gamma_val) * 255.0
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)

        return image

    def update_preview(self, *args):
        """Update the preview frames with current color adjustments - show before/after"""
        if not self.original_frames:
            return

        # Update all value labels with proper formatting
        self.brightness_val.config(text=f"{self.brightness.get():.2f}")
        self.contrast_val.config(text=f"{self.contrast.get():.2f}")
        self.gamma_val.config(text=f"{self.gamma.get():.2f}")
        self.exposure_val.config(text=f"{self.exposure.get():.2f}")

        # Clear existing preview images (except labels)
        for widget in self.preview_container.winfo_children():
            if isinstance(widget, ttk.Label) and widget.cget("text") in ["BEFORE (Original)", "AFTER (Edited)"]:
                continue  # Keep the labels
            widget.destroy()
        self.photo_refs.clear()

        # Re-add the labels
        ttk.Label(self.preview_container, text="BEFORE (Original)", font=("Arial", 9, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 5))
        ttk.Label(self.preview_container, text="AFTER (Edited)", font=("Arial", 9, "bold")).grid(row=2, column=0, columnspan=3, pady=(5, 5))

        # Display original frames (top row)
        for i, original_frame in enumerate(self.original_frames):
            # Convert to PhotoImage for display
            photo = ImageTk.PhotoImage(original_frame)
            self.photo_refs.append(photo)  # Keep reference

            # Create label to display original image
            label = ttk.Label(self.preview_container, image=photo)
            label.grid(row=1, column=i, padx=5, pady=5)

        # Apply corrections and display edited frames (bottom row)
        for i, original_frame in enumerate(self.original_frames):
            # Apply color corrections
            corrected_frame = self.apply_color_corrections(original_frame.copy())

            # Convert to PhotoImage for display
            photo = ImageTk.PhotoImage(corrected_frame)
            self.photo_refs.append(photo)  # Keep reference

            # Create label to display corrected image
            label = ttk.Label(self.preview_container, image=photo)
            label.grid(row=3, column=i, padx=5, pady=5)

    def shuffle_frames(self):
        """Load 3 new random frames from the current video"""
        if not self.video_path:
            messagebox.showwarning("Warning", "No video file selected")
            return
        
        self.load_sample_frames()

    def reset_values(self):
        """Reset all values to FFmpeg defaults"""
        self.brightness.set(0.0)  # FFmpeg default
        self.contrast.set(1.0)    # FFmpeg default
        self.gamma.set(1.0)       # FFmpeg default
        self.exposure.set(0.0)    # FFmpeg default

    def generate_ffmpeg_command(self, input_path, output_path, test_mode=False):
        """Generate the FFmpeg command based on current settings"""
        # Get current values
        brightness_val = self.brightness.get()
        contrast_val = self.contrast.get()
        gamma_val = self.gamma.get()
        exposure_val = self.exposure.get()

        # Check if we need any adjustments (use real FFmpeg defaults)
        needs_adjustment = not (brightness_val == 0.0 and contrast_val == 1.0 and gamma_val == 1.0 and exposure_val == 0.0)
        
        if not needs_adjustment:
            # If no adjustments needed, just copy the video without re-encoding
            cmd = [
                "ffmpeg",
                "-i", input_path,
            ]
            if test_mode:
                cmd.extend(["-t", "5"])  # Only 5 seconds for test
            cmd.extend([
                "-c", "copy",  # Copy both video and audio without re-encoding
                "-avoid_negative_ts", "make_zero",  # Handle timestamp issues
                "-y",  # Overwrite output file
                output_path,
            ])
        else:
            # Use FFmpeg values directly (no mapping needed)
            brightness_str = f"{brightness_val:.2f}".replace(',', '.')
            contrast_str = f"{contrast_val:.2f}".replace(',', '.')
            gamma_str = f"{gamma_val:.2f}".replace(',', '.')
            exposure_str = f"{exposure_val:.2f}".replace(',', '.')
            
            # Build filter chain
            filters = []
            
            # Add exposure filter if needed
            if exposure_val != 0.0:
                filters.append(f"exposure=exposure={exposure_str}")
            
            # Add eq filter if needed
            if brightness_val != 0.0 or contrast_val != 1.0 or gamma_val != 1.0:
                filters.append(f"eq=brightness={brightness_str}:contrast={contrast_str}:gamma={gamma_str}")
            
            filter_string = ",".join(filters)
            
            cmd = [
                "ffmpeg",
                "-i", input_path,
            ]
            if test_mode:
                cmd.extend(["-t", "5"])  # Only 5 seconds for test
            cmd.extend([
                "-vf", filter_string,
                "-c:v", "libx264",  # Explicitly set video codec
                "-pix_fmt", "yuv420p",  # Force standard pixel format
                "-color_range", "1",  # Preserve color range (1=tv/limited, 2=pc/full)
                "-colorspace", "bt709",  # Standard HD colorspace
                "-color_primaries", "bt709",  # Standard HD primaries
                "-color_trc", "bt709",  # Standard HD transfer characteristics
                "-c:a", "copy",  # Copy audio without re-encoding
                "-preset", "fast" if not test_mode else "ultrafast",  # Use fast encoding preset
                "-crf", "18" if not test_mode else "23",  # High quality encoding
                "-y",  # Overwrite output file
                output_path,
            ])
        
        return cmd

    def test_export(self):
        """Export a 5-second test clip with current settings"""
        if not self.video_path:
            messagebox.showerror("Error", "No video file selected")
            return

        # Auto-generate test filename
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.dirname(self.video_path)
        test_output = os.path.join(output_dir, f"{base_name}_test_5s.mp4")
        
        # Create test ffmpeg command (5 seconds from start)
        self.run_test_export(test_output)

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

    def export_command(self):
        """Export ONLY the FFmpeg command as text with real file paths"""
        if not self.video_path:
            messagebox.showerror("Error", "No video file selected")
            return

        # Get the output filename from the text box
        output_filename = self.output_filename.get().strip()
        if not output_filename:
            messagebox.showerror("Error", "Please enter an output filename")
            return

        # Create the full output path (same directory as input + user-specified filename)
        input_dir = os.path.dirname(self.video_path)
        output_path = os.path.join(input_dir, output_filename)

        # Generate the FFmpeg command for full video (not test mode)
        cmd = self.generate_ffmpeg_command(self.video_path, output_path, test_mode=False)

        # Format the command as a single line with proper quoting
        def format_command_for_export(cmd):
            formatted_args = []
            for arg in cmd:
                if " " in arg or "'" in arg or '"' in arg or any(char in arg for char in ['&', '|', '(', ')', '<', '>', ';']):
                    formatted_args.append(f'"{arg}"')
                else:
                    formatted_args.append(arg)
            return " ".join(formatted_args)

        command_text = format_command_for_export(cmd)

        # Get save location for the command file
        save_path = filedialog.asksaveasfilename(
            title="Save FFmpeg Command",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if not save_path:
            return

        try:
            # Write ONLY the command to file - nothing else
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(command_text)
            
            self.status_label.config(text=f"Command exported: {os.path.basename(save_path)}")
            messagebox.showinfo("Success", f"FFmpeg command exported to:\n{save_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving command: {str(e)}")
            self.status_label.config(text="Error exporting command")

    def run_test_export(self, output_path):
        """Run a test export for 5 seconds"""
        def export_thread():
            try:
                # Force C locale for consistent decimal formatting
                old_locale = locale.getlocale(locale.LC_NUMERIC)
                try:
                    locale.setlocale(locale.LC_NUMERIC, 'C')
                except:
                    pass
                
                self.status_label.config(text="Testing export (5s)...")
                self.root.update()

                cmd = self.generate_ffmpeg_command(self.video_path, output_path, test_mode=True)

                print(f"Test command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    shell=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )

                def update_test_success():
                    self.status_label.config(text=f"Test completed: {os.path.basename(output_path)}")
                    messagebox.showinfo("Test Success", f"5-second test exported to:\n{output_path}\n\nCheck if it looks correct before doing full export.")

                def update_test_error(error_msg):
                    self.status_label.config(text="Test export failed")
                    messagebox.showerror("Test Export Error", f"Test failed:\n{error_msg}")

                if result.returncode == 0:
                    self.root.after(0, update_test_success)
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    if not error_msg:
                        error_msg = f"FFmpeg exited with code {result.returncode}"
                    print(f"Test export error: {error_msg}")
                    self.root.after(0, lambda: update_test_error(error_msg))

                # Restore locale
                try:
                    locale.setlocale(locale.LC_NUMERIC, old_locale)
                except:
                    pass

            except Exception as e:
                try:
                    locale.setlocale(locale.LC_NUMERIC, old_locale)
                except:
                    pass
                def update_general_error():
                    messagebox.showerror("Error", f"Test export error: {str(e)}")
                    self.status_label.config(text="Test error")
                self.root.after(0, update_general_error)

        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()

    def run_ffmpeg_export(self, output_path):
        """Run ffmpeg export in a separate thread"""

        def export_thread():
            try:
                # Force C locale for consistent decimal formatting
                old_locale = locale.getlocale(locale.LC_NUMERIC)
                try:
                    locale.setlocale(locale.LC_NUMERIC, 'C')
                except:
                    pass  # If setting locale fails, the .replace() method will handle it
                
                self.status_label.config(text="Exporting video...")
                self.root.update()

                cmd = self.generate_ffmpeg_command(self.video_path, output_path, test_mode=False)

                # Debug output - properly quote paths for display
                debug_cmd = []
                for arg in cmd:
                    if " " in arg or "'" in arg or '"' in arg:
                        debug_cmd.append(f'"{arg}"')
                    else:
                        debug_cmd.append(arg)
                print(f"Running command: {' '.join(debug_cmd)}")

                # Run ffmpeg - using list format handles spaces in paths automatically
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    shell=False,  # Explicitly use shell=False for security and proper path handling
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )

                # Update status in main thread
                def update_status_success():
                    self.status_label.config(text=f"Export completed: {os.path.basename(output_path)}")
                    messagebox.showinfo("Success", f"Video exported successfully to:\n{output_path}")

                def update_status_error(error_msg):
                    self.status_label.config(text="Export failed")
                    messagebox.showerror("Export Error", f"FFmpeg error:\n{error_msg}")

                if result.returncode == 0:
                    self.root.after(0, update_status_success)
                else:
                    error_msg = result.stderr if result.stderr else result.stdout
                    if not error_msg:
                        error_msg = f"FFmpeg exited with code {result.returncode}"
                    print(f"FFmpeg error: {error_msg}")  # Debug output
                    self.root.after(0, lambda: update_status_error(error_msg))

                # Restore original locale
                try:
                    locale.setlocale(locale.LC_NUMERIC, old_locale)
                except:
                    pass

            except FileNotFoundError:
                # Restore locale even on error
                try:
                    locale.setlocale(locale.LC_NUMERIC, old_locale)
                except:
                    pass
                    
                def update_ffmpeg_error():
                    messagebox.showerror("Error", "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
                    self.status_label.config(text="FFmpeg not found")
                self.root.after(0, update_ffmpeg_error)
            except Exception as e:
                # Restore locale even on error
                try:
                    locale.setlocale(locale.LC_NUMERIC, old_locale)
                except:
                    pass
                    
                def update_general_error():
                    messagebox.showerror("Error", f"Export error: {str(e)}")
                    self.status_label.config(text="Export error")
                print(f"Export exception: {str(e)}")  # Debug output
                self.root.after(0, update_general_error)

        # Run in separate thread to prevent GUI freezing
        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()


def main():
    root = tk.Tk()
    app = VideoColorEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()