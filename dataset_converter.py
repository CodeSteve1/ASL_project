import os
import cv2
import numpy as np
import mediapipe as mp
import random

# --- CONFIGURATION ---
# The path to your new dataset folder (containing subfolders A, B, C...)
BASE_DIR = "dataset_new" 
# The directory where the new skeletal images will be saved
OUTPUT_DIR = "skeletal_dataset_from_new" 
# The size of the output images
TARGET_SIZE = 224

# --- AUGMENTATION CONFIG ---
# For each original image, create this many augmented versions.
# Set to 0 to disable augmentation.
AUGMENTATION_COUNT = 5
# Maximum random rotation in degrees (+/-)
MAX_ROTATION = 15
# Maximum random scaling (+/- this percentage)
MAX_SCALE_CHANGE = 0.15
# Maximum random translation as a fraction of the image size
MAX_TRANSLATION = 0.1

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Output directory '{OUTPUT_DIR}' is ready.")

# --- MediaPipe Hand Landmarker Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3 # Lower confidence to be less strict
)

# --- Helper Functions ---
def augment_landmarks(landmarks, target_size):
    """Applies random rotation, scaling, and translation to landmarks."""
    if not landmarks: return []
    center_x = sum(p[0] for p in landmarks) / len(landmarks)
    center_y = sum(p[1] for p in landmarks) / len(landmarks)
    center = np.array([center_x, center_y])
    angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    scale = random.uniform(1 - MAX_SCALE_CHANGE, 1 + MAX_SCALE_CHANGE)
    dx = random.uniform(-MAX_TRANSLATION, MAX_TRANSLATION) * target_size
    dy = random.uniform(-MAX_TRANSLATION, MAX_TRANSLATION) * target_size
    augmented_landmarks = []
    for point in landmarks:
        p_centered = np.array(point) - center
        p_scaled = p_centered * scale
        p_rotated = np.dot(rotation_matrix, p_scaled)
        p_recentered = p_rotated + center
        p_translated = p_recentered + np.array([dx, dy])
        augmented_landmarks.append(tuple(p_translated.astype(int)))
    return augmented_landmarks

def draw_skeleton(landmarks, target_size):
    """Draws a hand skeleton on a black canvas from a list of landmarks."""
    skeleton_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point, end_point = landmarks[start_idx], landmarks[end_idx]
            if 0 <= start_point[0] < target_size and 0 <= start_point[1] < target_size and \
               0 <= end_point[0] < target_size and 0 <= end_point[1] < target_size:
                cv2.line(skeleton_img, start_point, end_point, (255, 255, 255), 2)
    for point in landmarks:
        if 0 <= point[0] < target_size and 0 <= point[1] < target_size:
            cv2.circle(skeleton_img, point, 5, (0, 255, 0), -1)
    return skeleton_img

# --- Main Processing Loop ---
# Get the list of class folders (A, B, C...)
class_folders = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])

if not class_folders:
    print(f"❌ No class folders found in '{BASE_DIR}'. Please check the path.")
else:
    print(f"Found {len(class_folders)} classes. Starting processing...")

for class_name in class_folders:
    source_class_dir = os.path.join(BASE_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(source_class_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    total_files = len(image_files)
    
    print(f"\nProcessing class: '{class_name}' ({total_files} images)")
    
    # Replaced tqdm with a manual counter for better compatibility
    for i, filename in enumerate(image_files):
        # Print progress every 100 files
        if (i + 1) % 100 == 0 or (i + 1) == total_files:
            print(f"  -> Class '{class_name}': Processing file {i+1}/{total_files}")

        try:
            image_path = os.path.join(source_class_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                # Silently skip unreadable images to avoid cluttering the log
                continue
            
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, _ = image.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox_w, bbox_h = x_max - x_min, y_max - y_min
                
                if bbox_w <= 0 or bbox_h <= 0: continue

                margin = 0.20
                x_min -= bbox_w * margin
                x_max += bbox_w * margin
                y_min -= bbox_h * margin
                y_max += bbox_h * margin
                bbox_w, bbox_h = x_max - x_min, y_max - y_min

                scale = TARGET_SIZE / max(bbox_w, bbox_h)
                offset_x = (TARGET_SIZE - bbox_w * scale) / 2
                offset_y = (TARGET_SIZE - bbox_h * scale) / 2

                original_landmarks = []
                for landmark in hand_landmarks.landmark:
                    orig_x, orig_y = landmark.x * w, landmark.y * h
                    canvas_x = int(((orig_x - x_min) * scale) + offset_x)
                    canvas_y = int(((orig_y - y_min) * scale) + offset_y)
                    original_landmarks.append((canvas_x, canvas_y))

                base_name, ext = os.path.splitext(filename)
                
                # Save original image
                original_skeleton_img = draw_skeleton(original_landmarks, TARGET_SIZE)
                cv2.imwrite(os.path.join(output_class_dir, filename), original_skeleton_img)

                # Save augmented versions
                for j in range(AUGMENTATION_COUNT):
                    augmented_landmarks = augment_landmarks(original_landmarks, TARGET_SIZE)
                    augmented_img = draw_skeleton(augmented_landmarks, TARGET_SIZE)
                    aug_filename = f"{base_name}_aug_{j+1}{ext}"
                    cv2.imwrite(os.path.join(output_class_dir, aug_filename), augmented_img)
            else:
                # This is not an error, just an informational skip
                pass 

        except Exception as e:
            print(f"💥 An unexpected error occurred on {filename}: {e}")

# Clean up
hands.close()
print("\n✅ All processing complete.")
