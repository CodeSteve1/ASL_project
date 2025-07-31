import os
import shutil
import random
from tqdm import tqdm

# --- CONFIGURATION ---
# The skeletal dataset created by the previous script
SOURCE_DIR = "skeletal_dataset_from_new" 
# The new directory for the YOLO-formatted dataset
YOLO_DIR = "yolo_skeletal_dataset" 
# Define the split ratios
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
# TEST_RATIO will be the remainder

# --- Class mapping from letter to an integer ID ---
# Including 'del' and 'space' if they exist in your dataset
CLASS_NAMES = sorted([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))])
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# --- Main Script ---
if os.path.exists(YOLO_DIR):
    print(f"Directory '{YOLO_DIR}' already exists. Removing it to start fresh.")
    shutil.rmtree(YOLO_DIR)

print("Creating YOLO dataset structure...")
# Create directories for images and labels for each split
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(YOLO_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DIR, 'labels', split), exist_ok=True)

print("Processing and splitting files...")

# Iterate over each class folder ('A', 'B', etc.)
for class_name, class_id in tqdm(CLASS_MAP.items(), desc="Processing Classes"):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images) # Shuffle for random splitting

    # Calculate split indices
    train_end = int(len(images) * TRAIN_RATIO)
    valid_end = int(len(images) * (TRAIN_RATIO + VALID_RATIO))

    # Assign images to splits
    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]

    splits = {'train': train_images, 'valid': valid_images, 'test': test_images}

    for split_name, image_list in splits.items():
        for filename in image_list:
            # Copy image file
            shutil.copy(
                os.path.join(class_path, filename),
                os.path.join(YOLO_DIR, 'images', split_name, filename)
            )
            
            # Create YOLO label file (.txt)
            # The bounding box is the entire image (class_id center_x center_y width height)
            yolo_label_content = f"{class_id} 0.5 0.5 1 1"
            label_filename = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(YOLO_DIR, 'labels', split_name, label_filename), 'w') as f:
                f.write(yolo_label_content)

print("\n✅ YOLO dataset preparation complete.")
print(f"New dataset is located at: '{YOLO_DIR}'")
print(f"Found {len(CLASS_NAMES)} classes: {CLASS_NAMES}")
