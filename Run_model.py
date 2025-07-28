import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time

# --- CONFIGURATION ---
MODEL_PATH = 'best.pt'
TARGET_SIZE = 224

# --- Load YOLO Model ---
try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    exit()

# --- MediaPipe Hand Landmarker Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def draw_skeleton(landmarks, target_size):
    skeleton_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            if 0 <= start_point[0] < target_size and 0 <= start_point[1] < target_size and \
               0 <= end_point[0] < target_size and 0 <= end_point[1] < target_size:
                cv2.line(skeleton_img, start_point, end_point, (255, 255, 255), 2)
    for point in landmarks:
        if 0 <= point[0] < target_size and 0 <= point[1] < target_size:
            cv2.circle(skeleton_img, point, 5, (0, 255, 0), -1)
    return skeleton_img

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("📷 Webcam started. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(rgb_frame)
        frame.flags.writeable = True

        if results.multi_hand_landmarks:
            print("🖐️ Hand Detected ✅")
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bbox_w, bbox_h = x_max - x_min, y_max - y_min

            if bbox_w > 0 and bbox_h > 0:
                margin = 0.2
                x_min -= bbox_w * margin
                x_max += bbox_w * margin
                y_min -= bbox_h * margin
                y_max += bbox_h * margin

                bbox_w, bbox_h = x_max - x_min, y_max - y_min
                scale = TARGET_SIZE / max(bbox_w, bbox_h)
                offset_x = (TARGET_SIZE - bbox_w * scale) / 2
                offset_y = (TARGET_SIZE - bbox_h * scale) / 2

                transformed_landmarks = []
                for landmark in hand_landmarks.landmark:
                    orig_x, orig_y = landmark.x * w, landmark.y * h
                    canvas_x = int(((orig_x - x_min) * scale) + offset_x)
                    canvas_y = int(((orig_y - y_min) * scale) + offset_y)
                    transformed_landmarks.append((canvas_x, canvas_y))

                skeleton_img = draw_skeleton(transformed_landmarks, TARGET_SIZE)
                cv2.imshow("🦴 Skeleton", skeleton_img)
                print("📏 Skeleton image shape:", skeleton_img.shape)

                try:
                    yolo_results = model.predict(skeleton_img, verbose=False)
                    print("✅ YOLO Prediction Done")

                    if yolo_results and hasattr(yolo_results[0], 'boxes') and yolo_results[0].boxes:
                        box = yolo_results[0].boxes[0]
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        label = model.names[class_id]

                        print(f"🎯 Prediction: {label} ({confidence:.2f})")

                        # Draw bounding box and label
                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                    (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        print("⚠️ No prediction made by YOLO.")
                except Exception as e:
                    print("🚨 Error during YOLO prediction:", e)
        else:
            print("❌ No hand detected.")

        cv2.imshow('📷 ASL Hand Sign Detection', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    main()
