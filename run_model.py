import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# --- CONFIGURATION ---
# Path to your trained YOLOv8 model. UPDATE THIS PATH if needed.
MODEL_PATH = r'C:\Users\coder\Desktop\asl\runs\detect\yolo11s_asl_custom_aug3\weights\best.pt'

# The size of the image the model was trained on
TARGET_SIZE = 224

# --- Load YOLO Model ---
print("Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    print(f"👉 Please ensure the MODEL_PATH is correct: '{MODEL_PATH}'")
    exit()

# --- MediaPipe Hand Landmarker Setup ---
print("Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ MediaPipe Hands initialized.")

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

def main():
    cap = None
    try:
        print("\n--- Live ASL Inference ---")
        print("Attempting to open webcam...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(2.0)

        if not cap.isOpened():
            print("❌ FATAL ERROR: Could not open webcam.")
            return

        print("✅ Webcam opened successfully. Starting inference loop...")
        
        while True:
            success, frame = cap.read()
            if not success:
                print("❌ Failed to read frame from camera. Exiting loop.")
                break

            # Flip the frame for a selfie-view
            frame = cv2.flip(frame, 1)
            
            # Create a copy for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to find hands
            results = hands.process(rgb_frame)
            
            # Create a blank canvas for the skeletal view
            skeletal_view = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)

            # If a hand is detected, process it for YOLO
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                h, w, _ = frame.shape
                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                bbox_w, bbox_h = x_max - x_min, y_max - y_min
                
                if bbox_w > 0 and bbox_h > 0:
                    margin = 0.20
                    x_min_margin = int(x_min - bbox_w * margin)
                    x_max_margin = int(x_max + bbox_w * margin)
                    y_min_margin = int(y_min - bbox_h * margin)
                    y_max_margin = int(y_max + bbox_h * margin)
                    
                    bbox_w_margin = x_max_margin - x_min_margin
                    bbox_h_margin = y_max_margin - y_min_margin

                    if bbox_w_margin > 0 and bbox_h_margin > 0:
                        scale = TARGET_SIZE / max(bbox_w_margin, bbox_h_margin)
                        offset_x = (TARGET_SIZE - bbox_w_margin * scale) / 2
                        offset_y = (TARGET_SIZE - bbox_h_margin * scale) / 2

                        transformed_landmarks = []
                        for landmark in hand_landmarks.landmark:
                            orig_x, orig_y = landmark.x * w, landmark.y * h
                            canvas_x = int(((orig_x - x_min_margin) * scale) + offset_x)
                            canvas_y = int(((orig_y - y_min_margin) * scale) + offset_y)
                            transformed_landmarks.append((canvas_x, canvas_y))

                        skeletal_view = draw_skeleton(transformed_landmarks, TARGET_SIZE)

                        yolo_results = model.predict(skeletal_view, verbose=False)
                        
                        if yolo_results and yolo_results[0].boxes:
                            box = yolo_results[0].boxes[0]
                            class_id = int(box.cls)
                            confidence = float(box.conf)
                            label = model.names[class_id]
                            
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            text = f"{label.upper()} ({confidence:.2f})"
                            cv2.putText(frame, text, (x_min, y_min - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display both windows on every frame
            cv2.imshow('Webcam Feed', frame)
            cv2.imshow('Skeletal View (Model Input)', skeletal_view)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' key pressed. Exiting.")
                break
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        print("Exiting and cleaning up resources...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if 'hands' in locals() and hands:
            hands.close()
        print("✅ Done.")

if __name__ == '__main__':
    main()
