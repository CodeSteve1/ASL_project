import os
import cv2
import numpy as np
import mediapipe as mp
import time
import random

# CONFIG
BASE_DIR = "test_dataset"
OUTPUT_DIR = "cleaned_dataset_skeletal"
SPLITS = ['train', 'valid', 'test']
TARGET_SIZE = 224

# --- AUGMENTATION CONFIG ---
# For each original image, we will create this many augmented versions.
# Set to 0 to disable augmentation.
AUGMENTATION_COUNT = 5
# Maximum random rotation in degrees (+/-)
MAX_ROTATION = 15
# Maximum random scaling (+/- this percentage)
MAX_SCALE_CHANGE = 0.15
# Maximum random translation as a fraction of the image size
MAX_TRANSLATION = 0.1

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Output directory '{OUTPUT_DIR}' is ready.")

# --- MediaPipe Hand Landmarker Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3 # Lowered confidence to be less strict
)

# --- Class ID to letter mapping ---
class_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# --- Augmentation Helper Functions ---
def augment_landmarks(landmarks, target_size):
    """Applies random rotation, scaling, and translation to landmarks."""
    
    # Calculate the center of the landmarks
    center_x = sum(p[0] for p in landmarks) / len(landmarks)
    center_y = sum(p[1] for p in landmarks) / len(landmarks)
    center = np.array([center_x, center_y])

    # 1. Rotation
    angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # 2. Scaling
    scale = random.uniform(1 - MAX_SCALE_CHANGE, 1 + MAX_SCALE_CHANGE)

    # 3. Translation
    dx = random.uniform(-MAX_TRANSLATION, MAX_TRANSLATION) * target_size
    dy = random.uniform(-MAX_TRANSLATION, MAX_TRANSLATION) * target_size

    augmented_landmarks = []
    for point in landmarks:
        # Translate point to origin, scale, rotate, then translate back
        p_centered = np.array(point) - center
        p_scaled = p_centered * scale
        p_rotated = np.dot(rotation_matrix, p_scaled)
        p_recentered = p_rotated + center
        # Apply final translation
        p_translated = p_recentered + np.array([dx, dy])
        augmented_landmarks.append(tuple(p_translated.astype(int)))
        
    return augmented_landmarks

def draw_skeleton(landmarks):
    """Draws a hand skeleton on a black canvas from a list of landmarks."""
    skeleton_img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    
    # Draw connections
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            # Ensure points are within canvas bounds
            if 0 <= start_point[0] < TARGET_SIZE and 0 <= start_point[1] < TARGET_SIZE and \
               0 <= end_point[0] < TARGET_SIZE and 0 <= end_point[1] < TARGET_SIZE:
                cv2.line(skeleton_img, start_point, end_point, (255, 255, 255), 2)
    
    # Draw landmark points
    for point in landmarks:
        if 0 <= point[0] < TARGET_SIZE and 0 <= point[1] < TARGET_SIZE:
            cv2.circle(skeleton_img, point, 5, (0, 255, 0), -1)
            
    return skeleton_img

# --- Main Processing Loop ---
for split in SPLITS:
    image_dir = os.path.join(BASE_DIR, split, "images")
    label_dir = os.path.join(BASE_DIR, split, "labels")

    if not os.path.isdir(image_dir):
        print(f"❌ FATAL: Image directory not found, skipping split: '{image_dir}'")
        continue

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print(f"🤷 No images found in '{image_dir}'.")
        continue

    total_files = len(image_files)
    print(f"\nProcessing {total_files} images in '{split}' split...")
    
    for i, filename in enumerate(image_files):
        if (i + 1) % 50 == 0 or (i + 1) == total_files:
             print(f"  -> Processing file {i+1}/{total_files} ({filename})")

        try:
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

            if not os.path.exists(label_path):
                label = "unknown"
            else:
                with open(label_path, 'r') as f:
                    line = f.readline()
                    if line and line.strip():
                        class_id = int(line.strip().split()[0])
                        label = class_map.get(class_id, "unknown")
                    else:
                        label = "unknown"

            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Error: Could not read image '{filename}'")
                continue
            
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, _ = image.shape

                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox_w, bbox_h = x_max - x_min, y_max - y_min
                
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                margin = 0.20
                x_min -= bbox_w * margin
                x_max += bbox_w * margin
                y_min -= bbox_h * margin
                y_max += bbox_h * margin
                bbox_w, bbox_h = x_max - x_min, y_max - y_min

                scale = TARGET_SIZE / max(bbox_w, bbox_h)
                offset_x = (TARGET_SIZE - bbox_w * scale) / 2
                offset_y = (TARGET_SIZE - bbox_h * scale) / 2

                # Calculate the original, clean landmarks
                original_landmarks = []
                for landmark in hand_landmarks.landmark:
                    orig_x, orig_y = landmark.x * w, landmark.y * h
                    canvas_x = int(((orig_x - x_min) * scale) + offset_x)
                    canvas_y = int(((orig_y - y_min) * scale) + offset_y)
                    mirrored_x = TARGET_SIZE - 1 - canvas_x
                    original_landmarks.append((mirrored_x, canvas_y))

                # --- Save Original and Augmented Images ---
                save_dir = os.path.join(OUTPUT_DIR, split, label)
                os.makedirs(save_dir, exist_ok=True)
                base_name, ext = os.path.splitext(filename)

                # 1. Save the original, non-augmented image
                original_skeleton_img = draw_skeleton(original_landmarks)
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, original_skeleton_img)

                # 2. Save augmented versions
                for j in range(AUGMENTATION_COUNT):
                    augmented_landmarks = augment_landmarks(original_landmarks, TARGET_SIZE)
                    augmented_img = draw_skeleton(augmented_landmarks)
                    aug_filename = f"{base_name}_aug_{j+1}{ext}"
                    save_path = os.path.join(save_dir, aug_filename)
                    cv2.imwrite(save_path, augmented_img)
                
            else:
                print(f"⚠️  Skipped {filename}: No hand detected by MediaPipe.")

        except Exception as e:
            print(f"💥 An unexpected error occurred on {filename}: {e}")

# Clean up MediaPipe resources
hands.close()
print("\n✅ Processing complete.")
