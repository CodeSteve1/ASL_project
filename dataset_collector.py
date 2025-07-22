

import cv2
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Base folder to save data
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

class ASLDatasetTool:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Dataset Tool (No MediaPipe)")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e1e")

        # Camera and variables
        self.cap = cv2.VideoCapture(0)
        self.labels = ["eat", "sleep", "help", "restroom"]
        self.current_label = tk.StringVar()

        # Build UI
        self.create_widgets()
        self.update_video()

    def create_widgets(self):
        # Video Feed
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        # Label buttons
        self.button_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.button_frame.pack(pady=5)

        for label in self.labels:
            self.add_label_button(label)

        # Current label
        self.current_label_display = tk.Label(self.root, text="Current Label: None", fg="white", bg="#1e1e1e", font=("Arial", 14))
        self.current_label_display.pack()

        # Capture button
        self.capture_button = tk.Button(self.root, text="Capture", command=self.capture_image, bg="#4caf50", fg="white", font=("Arial", 12))
        self.capture_button.pack(pady=10)

        # Add new label UI
        self.add_label_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.add_label_frame.pack(pady=10)

        self.new_label_entry = tk.Entry(self.add_label_frame, font=("Arial", 12))
        self.new_label_entry.pack(side=tk.LEFT, padx=5)

        self.add_label_btn = tk.Button(self.add_label_frame, text="Add Label", command=self.add_new_label, bg="#2196f3", fg="white", font=("Arial", 12))
        self.add_label_btn.pack(side=tk.LEFT)

    def add_label_button(self, label):
        btn = tk.Button(self.button_frame, text=label, command=lambda l=label: self.set_label(l), font=("Arial", 12), bg="#3e3e3e", fg="white", padx=10)
        btn.pack(side=tk.LEFT, padx=5)

    def set_label(self, label):
        self.current_label.set(label)
        self.current_label_display.config(text=f"Current Label: {label}")
        os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

    def add_new_label(self):
        label = self.new_label_entry.get().strip().lower()
        if label and label not in self.labels:
            self.labels.append(label)
            self.add_label_button(label)
            self.set_label(label)
            self.new_label_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Invalid", "Label is empty or already exists.")

    def capture_image(self):
        if not self.current_label.get():
            messagebox.showerror("Error", "Please select a label first.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Camera failed.")
            return

        frame = cv2.flip(frame, 1)  # Mirror view

        label_dir = os.path.join(DATASET_DIR, self.current_label.get())
        count = len(os.listdir(label_dir))
        filepath = os.path.join(label_dir, f"{count+1}.jpg")
        cv2.imwrite(filepath, frame)
        print(f"[INFO] Saved: {filepath}")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_video)

    def on_close(self):
        self.cap.release()
        self.root.destroy()

# Main
if __name__ == "__main__":
    root = tk.Tk()
    app = ASLDatasetTool(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
