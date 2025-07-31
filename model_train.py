import torch
from ultralytics import YOLO
import os

# --- CONFIGURATION ---
# Path to your dataset configuration file
DATASET_YAML_PATH = 'skeletal_data.yaml'

# Model to use. 'yolov8s.pt' is the standard small model.
MODEL_NAME = 'yolo11s.pt' 

# A descriptive name for this specific training run
RUN_NAME = 'yolo11s_asl_custom_aug'

# Training parameters
EPOCHS = 75
IMAGE_SIZE = 224
BATCH_SIZE = 32 # You can increase this if your GPU has more memory

def main():
    """
    Main function to run YOLOv8 training with custom augmentations on the GPU.
    """
    # Check if a GPU is available
    if torch.cuda.is_available():
        print(f"✅ GPU found: {torch.cuda.get_device_name(0)}")
        device = 0 # Use the first available GPU
    else:
        print("⚠️ GPU not found, training will run on the CPU.")
        device = "cpu"

    # Load the base YOLOv8 model
    model = YOLO(MODEL_NAME)

    print("\nStarting training with custom augmentations...")
    # Train the model with specific augmentation settings
    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        name=RUN_NAME,
        save_period=5,   # Saves a checkpoint every 5 epochs.
        
        # --- Custom Augmentation Settings ---
        fliplr=1.0,      # Horizontally flip 100% of the images.
        
        # --- Enable Geometric Augmentations ---
        degrees=10.0,      # Random rotation between -10 and +10 degrees.
        scale=0.1,       # Random scaling between 90% and 110%.
        translate=0.1,   # Random translation up to 10% of image size.
        perspective=0.001, # Enables a slight "tilt" effect.
        
        # --- Disable Color Distortion ---
        hsv_h=0.0,     # Disable Hue distortion.
        hsv_s=0.0,     # Disable Saturation distortion.
        hsv_v=0.0,     # Disable Value/Brightness distortion.
        
        # --- Keep Other Complex Augmentations Disabled ---
        shear=0.0,       # Disable shearing.
        mosaic=0.0,      # Disable mosaic augmentation.
        mixup=0.0,       # Disable mixup augmentation.
        copy_paste=0.0   # Disable copy-paste augmentation.
    )

    print("\n✅ Training complete.")
    print(f"Model and results saved in the 'runs/detect/{RUN_NAME}' directory.")

    # Evaluate model performance on the validation set
    print("\nRunning validation on the best model...")
    metrics = model.val()
    print(f"Validation mAP50-95: {metrics.box.map}")
    print(f"Validation mAP50: {metrics.box.map50}")


if __name__ == '__main__':
    main()
