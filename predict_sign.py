import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import os

class TrafficSignPredictor:
    def __init__(self, model_path="best_model.h5"):
        """Initialize with trained model"""
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = [
            "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", 
            "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
            "End speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
            "No passing", "No passing (trucks)", "Right-of-way at intersection",
            "Priority road", "Yield", "STOP", "No vehicles", 
            "No trucks", "No entry", "Caution", "Dangerous left curve",
            "Dangerous right curve", "Double curve", "Bumpy road", "Slippery road",
            "Road narrows", "Road work", "Traffic signals", "Pedestrians", 
            "Children crossing", "Bicycle crossing", "Ice/Snow",
            "Wild animals", "End restrictions", "Turn right", 
            "Turn left", "Ahead only", "Straight or right", "Straight or left",
            "Keep right", "Keep left", "Roundabout", "End no passing",
            "End no passing (trucks)"
        ]

    def predict(self, image_path):
        """Predict traffic sign from image"""
        try:
            # Preprocess image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (30, 30))
            img = np.expand_dims(img, axis=0) / 255.0

            # Make prediction
            pred = self.model.predict(img)
            class_id = np.argmax(pred)
            confidence = float(np.max(pred))
            return class_id, self.class_names[class_id], confidence
        
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

class TrafficSignGUI:
    def __init__(self, root):
        """Initialize GUI"""
        self.root = root
        self.predictor = TrafficSignPredictor()
        self.setup_gui()
        
    def setup_gui(self):
        """Configure GUI layout and widgets"""
        self.root.title("Traffic Sign Recognition")
        self.root.geometry("700x600")
        self.root.configure(bg="#f0f2f5")

        # Header
        tk.Label(
            self.root, 
            text="German Traffic Sign Recognition",
            font=("Helvetica", 18, "bold"),
            bg="#f0f2f5",
            pady=20
        ).pack()

        # Image display
        self.image_frame = tk.Frame(self.root, bg="white", bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.img_label = tk.Label(self.image_frame, bg="white")
        self.img_label.pack(pady=40)

        # Upload button
        tk.Button(
            self.root,
            text="Upload Image",
            command=self.load_image,
            font=("Helvetica", 12),
            bg="#4a6fa5",
            fg="white",
            padx=20,
            pady=10
        ).pack(pady=10)

        # Results display
        self.result_frame = tk.Frame(self.root, bg="#f0f2f5")
        self.result_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(
            self.result_frame,
            text="Prediction:",
            font=("Helvetica", 12, "bold"),
            bg="#f0f2f5"
        ).pack(anchor=tk.W)
        
        self.prediction_text = tk.StringVar(value="No image loaded")
        tk.Label(
            self.result_frame,
            textvariable=self.prediction_text,
            font=("Helvetica", 12),
            bg="#f0f2f5",
            wraplength=600
        ).pack(anchor=tk.W)

        # Confidence meter
        self.confidence_frame = tk.Frame(self.root, bg="#f0f2f5")
        self.confidence_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.confidence_text = tk.StringVar(value="Confidence: -")
        tk.Label(
            self.confidence_frame,
            textvariable=self.confidence_text,
            font=("Helvetica", 12),
            bg="#f0f2f5"
        ).pack(anchor=tk.W)
        
        self.canvas = tk.Canvas(
            self.confidence_frame,
            width=300,
            height=20,
            bg="#e0e0e0",
            highlightthickness=0
        )
        self.canvas.pack(anchor=tk.W)

    def load_image(self):
        """Handle image upload"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Traffic Sign Image",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                # Display image
                img = Image.open(filepath)
                img.thumbnail((400, 400))
                img_tk = ImageTk.PhotoImage(img)
                self.img_label.config(image=img_tk)
                self.img_label.image = img_tk

                # Make prediction
                class_id, class_name, confidence = self.predictor.predict(filepath)
                
                # Update results
                self.prediction_text.set(
                    f"Sign: {class_name}\n"
                    f"Category ID: {class_id}"
                )
                
                # Update confidence display
                self.confidence_text.set(f"Confidence: {confidence:.1%}")
                self.draw_confidence(confidence)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")

    def draw_confidence(self, confidence):
        """Draw confidence meter"""
        self.canvas.delete("all")
        width = 300 * confidence
        color = "#4CAF50" if confidence > 0.7 else "#FFC107" if confidence > 0.5 else "#F44336"
        self.canvas.create_rectangle(0, 0, width, 20, fill=color, outline="")

if __name__ == "__main__":
    # Verify model exists
    if not os.path.exists("best_model.h5"):
        messagebox.showerror(
            "Error", 
            "Model file 'best_model.h5' not found.\n"
            "Please train the model first using traffic.py"
        )
    else:
        root = tk.Tk()
        app = TrafficSignGUI(root)
        root.mainloop()