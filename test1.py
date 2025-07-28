import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Heem2/sign-language-classification")
model = AutoModelForImageClassification.from_pretrained("Heem2/sign-language-classification")

# Get label dictionary
id2label = model.config.id2label

# Start webcam
cap = cv2.VideoCapture(0)

print("Press SPACE to classify, ESC to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the live frame
    cv2.imshow("Sign Language Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

    if key == 32:  # SPACE key
        # Convert frame to RGB and PIL Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Preprocess and predict
        inputs = processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            pred_class = torch.argmax(outputs.logits, dim=-1).item()

        # Get label name
        label = id2label[pred_class]
        print("Predicted Label:", label)

        # Show label on frame
        frame = cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Sign Language Detection", frame)
        cv2.waitKey(1000)  # Show result for 1 second

cap.release()
cv2.destroyAllWindows()
