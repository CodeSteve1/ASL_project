import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# --- Constants ---
DATASET_DIR = "dataset"
APP_BG_COLOR = "#1e1e1e"
WIDGET_BG_COLOR = "#2d2d2d"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR_GREEN = "#4caf50"
ACCENT_COLOR_BLUE = "#2196f3"
ACCENT_COLOR_RED = "#f44336"
VIDEO_CAPTURE_BORDER_COLOR = "#00ff00" # Green flash on capture

class ASLDatasetTool:
    """
    A GUI tool for creating an American Sign Language (ASL) image dataset.
    It captures frames from a webcam, allows dynamic label creation, and saves
    images to a structured directory.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced ASL Dataset Tool")
        self.root.geometry("1000x750")
        self.root.configure(bg=APP_BG_COLOR)

        # --- State Variables ---
        self.labels = []
        self.current_label = tk.StringVar()
        self.image_count = tk.IntVar(value=0)
        
        # --- Camera Setup ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            self.root.destroy()
            return

        # --- Initial Setup ---
        os.makedirs(DATASET_DIR, exist_ok=True)
        self.load_existing_labels()
        self.setup_styles()
        self.create_widgets()
        
        # --- Start Video Loop ---
        self.update_video()

    def setup_styles(self):
        """Configures ttk styles for a modern, dark theme."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure general widgets
        style.configure('.', background=APP_BG_COLOR, foreground=TEXT_COLOR, font=('Arial', 11))
        style.configure('TFrame', background=APP_BG_COLOR)
        style.configure('TLabel', background=APP_BG_COLOR, foreground=TEXT_COLOR, font=('Arial', 12))
        style.configure('TButton', background=WIDGET_BG_COLOR, foreground=TEXT_COLOR, font=('Arial', 12), padding=5)
        style.map('TButton', background=[('active', ACCENT_COLOR_BLUE)])
        style.configure('TEntry', fieldbackground=WIDGET_BG_COLOR, foreground=TEXT_COLOR, insertbackground=TEXT_COLOR)
        
        # Special button styles
        style.configure('Success.TButton', background=ACCENT_COLOR_GREEN, foreground='white')
        style.map('Success.TButton', background=[('active', '#388e3c')])
        style.configure('Primary.TButton', background=ACCENT_COLOR_BLUE, foreground='white')
        style.map('Primary.TButton', background=[('active', '#1976d2')])
        style.configure('Danger.TButton', background=ACCENT_COLOR_RED, foreground='white')
        style.map('Danger.TButton', background=[('active', '#d32f2f')])

    def load_existing_labels(self):
        """Scans the dataset directory and loads any existing labels."""
        print("[INFO] Loading existing labels...")
        try:
            for item in os.scandir(DATASET_DIR):
                if item.is_dir():
                    self.labels.append(item.name)
            self.labels.sort()
            print(f"[INFO] Found labels: {self.labels}")
        except FileNotFoundError:
            print("[WARN] Dataset directory not found. Starting fresh.")

    def create_widgets(self):
        """Creates and arranges all the GUI widgets."""
        # --- Main Layout Frames ---
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        video_frame = ttk.Frame(main_frame, padding=5)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        controls_frame = ttk.Frame(main_frame, padding=10)
        controls_frame.grid(row=0, column=1, sticky="ns")
        
        main_frame.grid_columnconfigure(0, weight=3) # Video takes more space
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # --- Video Feed ---
        self.video_label = ttk.Label(video_frame, borderwidth=2, relief="solid")
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # --- Control Panel Widgets ---
        # Label selection
        ttk.Label(controls_frame, text="1. Select or Add a Label", font=('Arial', 14, 'bold')).pack(fill=tk.X, pady=(0, 10))
        
        self.label_buttons_frame = ttk.Frame(controls_frame)
        self.label_buttons_frame.pack(fill=tk.X, pady=5)
        self.repopulate_label_buttons()

        # Add new label
        add_label_frame = ttk.Frame(controls_frame)
        add_label_frame.pack(fill=tk.X, pady=10)
        self.new_label_entry = ttk.Entry(add_label_frame, font=("Arial", 12))
        self.new_label_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, ipady=4)
        self.add_label_btn = ttk.Button(add_label_frame, text="Add", command=self.add_new_label, style='Primary.TButton')
        self.add_label_btn.pack(side=tk.LEFT, padx=(5, 0))

        # Stats display
        stats_frame = ttk.Frame(controls_frame, padding=10, relief="groove")
        stats_frame.pack(fill=tk.X, pady=20)
        ttk.Label(stats_frame, text="Current Label:").grid(row=0, column=0, sticky="w")
        self.current_label_display = ttk.Label(stats_frame, textvariable=self.current_label, font=('Arial', 12, 'bold'))
        self.current_label_display.grid(row=0, column=1, sticky="w")
        
        ttk.Label(stats_frame, text="Image Count:").grid(row=1, column=0, sticky="w")
        self.image_count_display = ttk.Label(stats_frame, textvariable=self.image_count, font=('Arial', 12, 'bold'))
        self.image_count_display.grid(row=1, column=1, sticky="w")
        
        # Action buttons
        ttk.Label(controls_frame, text="2. Capture or Delete", font=('Arial', 14, 'bold')).pack(fill=tk.X, pady=(20, 10))
        self.capture_button = ttk.Button(controls_frame, text="📸 Capture Image", command=self.capture_image, style='Success.TButton')
        self.capture_button.pack(fill=tk.X, ipady=10, pady=5)
        
        self.delete_button = ttk.Button(controls_frame, text="🗑️ Delete Last Image", command=self.delete_last_image, style='Danger.TButton')
        self.delete_button.pack(fill=tk.X, ipady=10, pady=5)

    def repopulate_label_buttons(self):
        """Clears and rebuilds the label selection buttons."""
        for widget in self.label_buttons_frame.winfo_children():
            widget.destroy()
            
        for label in self.labels:
            btn = ttk.Button(self.label_buttons_frame, text=label, command=lambda l=label: self.set_label(l))
            btn.pack(fill=tk.X, pady=2)

    def set_label(self, label):
        """Sets the current label and updates the UI."""
        self.current_label.set(label)
        label_dir = os.path.join(DATASET_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        count = len(os.listdir(label_dir))
        self.image_count.set(count)
        print(f"[INFO] Label set to '{label}' with {count} images.")

    def add_new_label(self):
        """Adds a new label to the list and UI."""
        label = self.new_label_entry.get().strip().lower().replace(" ", "_")
        if not label:
            messagebox.showwarning("Invalid Label", "Label cannot be empty.")
            return
        if label in self.labels:
            messagebox.showwarning("Invalid Label", "Label already exists.")
        else:
            self.labels.append(label)
            self.labels.sort()
            self.repopulate_label_buttons()
            self.set_label(label)
            self.new_label_entry.delete(0, tk.END)
            print(f"[INFO] New label added: {label}")

    def capture_image(self):
        """Captures a single frame, saves it, and gives feedback."""
        label = self.current_label.get()
        if not label:
            messagebox.showerror("Error", "Please select a label first.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Camera Error", "Failed to capture image from camera.")
            return

        frame = cv2.flip(frame, 1) # Mirror view for intuitive interaction
        label_dir = os.path.join(DATASET_DIR, label)
        
        # Find the next available image number
        existing_files = os.listdir(label_dir)
        next_num = len(existing_files) + 1
        
        filepath = os.path.join(label_dir, f"{next_num}.jpg")
        cv2.imwrite(filepath, frame)
        
        self.image_count.set(self.image_count.get() + 1)
        print(f"[INFO] Saved: {filepath}")

        # Visual feedback
        self.video_label.config(background=VIDEO_CAPTURE_BORDER_COLOR)
        self.root.after(200, lambda: self.video_label.config(background="")) # Reset color

    def delete_last_image(self):
        """Deletes the most recently added image for the current label."""
        label = self.current_label.get()
        if not label:
            messagebox.showerror("Error", "Please select a label first.")
            return

        label_dir = os.path.join(DATASET_DIR, label)
        files = os.listdir(label_dir)
        if not files:
            messagebox.showinfo("Info", "No images to delete for this label.")
            return

        # Find the file with the highest number in its name
        last_file = max(files, key=lambda f: int(os.path.splitext(f)[0]))
        filepath = os.path.join(label_dir, last_file)

        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete '{filepath}'?"):
            os.remove(filepath)
            self.image_count.set(self.image_count.get() - 1)
            print(f"[INFO] Deleted: {filepath}")

    def update_video(self):
        """Continuously reads frames from the camera and updates the video label."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for display to fit the window better
            h, w, _ = img_rgb.shape
            aspect_ratio = w / h
            new_h = 480
            new_w = int(new_h * aspect_ratio)
            
            img_resized = cv2.resize(img_rgb, (new_w, new_h))
            
            img_pil = Image.fromarray(img_resized)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.root.after(10, self.update_video)

    def on_close(self):
        """Releases resources and closes the application."""
        print("[INFO] Releasing camera and closing application.")
        self.cap.release()
        self.root.destroy()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ASLDatasetTool(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
