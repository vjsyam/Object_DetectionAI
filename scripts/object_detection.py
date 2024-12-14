from ultralytics import YOLO
import torch
import cv2

# Load YOLOv8 model and ensure it's using GPU
model = YOLO('F:/AI Projects/Object_Detection_AI/models/yolov8n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Resize frame for faster processing
    frame = cv2.resize(frame, (1280, 720))

    # Perform detection on the frame with a lower confidence threshold for more detections
    results = model.predict(source=frame, conf=0.25, show=True)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
