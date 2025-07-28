import os
import shutil
from tqdm import tqdm

# CONFIG
# This should be the output directory from the previous script
SOURCE_DATASET_DIR = "cleaned_dataset_skeletal" 
# This is where the new YOLO-formatted dataset will be created
YOLO_DATASET_DIR = "yolo_dataset" 

# Class mapping from letter to an integer ID
CLASS_MAP = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
    'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
    'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25
}

# --- Main Script ---
if os.path.exists(YOLO_DATASET_DIR):
    print(f"Directory '{YOLO_DATASET_DIR}' already exists. Deleting it to start fresh.")
    shutil.rmtree(YOLO_DATASET_DIR)

print("Creating YOLO dataset structure...")

# Loop through train, valid, test splits
for split in ['train', 'valid', 'test']:
    source_split_dir = os.path.join(SOURCE_DATASET_DIR, split)
    if not os.path.isdir(source_split_dir):
        print(f"Warning: Source directory not found, skipping: {source_split_dir}")
        continue

    # Create YOLO directories
    yolo_images_dir = os.path.join(YOLO_DATASET_DIR, 'images', split)
    yolo_labels_dir = os.path.join(YOLO_DATASET_DIR, 'labels', split)
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    print(f"\nProcessing split: {split}")
    
    # Get all class folders (e.g., 'A', 'B', 'C')
    class_folders = [d for d in os.listdir(source_split_dir) if os.path.isdir(os.path.join(source_split_dir, d))]

    for class_name in tqdm(class_folders, desc=f"  Processing classes for {split}"):
        class_id = CLASS_MAP.get(class_name)
        if class_id is None:
            print(f"Warning: Class '{class_name}' not in CLASS_MAP. Skipping.")
            continue
            
        source_class_dir = os.path.join(source_split_dir, class_name)
        
        for filename in os.listdir(source_class_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                # Copy image file
                source_image_path = os.path.join(source_class_dir, filename)
                dest_image_path = os.path.join(yolo_images_dir, filename)
                shutil.copy(source_image_path, dest_image_path)
                
                # Create YOLO label file (.txt)
                # Since the skeleton fills the image, the bounding box is the whole frame.
                # Format: class_id center_x center_y width height (normalized)
                yolo_label_content = f"{class_id} 0.5 0.5 1 1"
                
                label_filename = os.path.splitext(filename)[0] + ".txt"
                dest_label_path = os.path.join(yolo_labels_dir, label_filename)
                
                with open(dest_label_path, 'w') as f:
                    f.write(yolo_label_content)

print("\n✅ YOLO dataset preparation complete.")
print(f"New dataset is located at: '{YOLO_DATASET_DIR}'")
