from ultralytics import YOLO

# --- CONFIG ---
# Path to your dataset configuration file
DATASET_YAML_PATH = 'asl_data.yaml'

# Model to use. 'yolov8s.pt' is a good starting point.
MODEL_NAME = 'best.pt'

# Training parameters
EPOCHS = 50
IMAGE_SIZE = 224
BATCH_SIZE = 16

def main():
    """
    Main function to run the YOLOv8 training.
    """
    print("Loading YOLOv8 model...")
    # Load a pre-trained YOLOv8s model
    model = YOLO(MODEL_NAME)

    print("Starting training...")
    # Train the model using the dataset specified in the YAML file
    # The results will be saved in a 'runs/detect/train' folder by default.
    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name='yolov8s_asl_skeletal' # Name for the output folder
    )

    print("\n✅ Training complete.")
    print(f"Model and results saved in the 'runs/detect/yolov8s_asl_skeletal' directory.")

    # (Optional) Evaluate model performance on the validation set
    print("\nRunning validation...")
    metrics = model.val()
    print("Validation metrics:", metrics.box.map50)

if __name__ == '__main__':
    main()
