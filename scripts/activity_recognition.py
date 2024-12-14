import tensorflow as tf
import cv2
import numpy as np

# Load your YOLOv8x object detection model (assume this is done)
# Example: Load pre-trained YOLOv8x model here
# yolo_model = some_yolov8x_loading_function()

# Placeholder: Load your Activity Recognition model (e.g., 3D CNN, LSTM, etc.)
# activity_model = some_activity_model_loading_function()

# Function to preprocess input frame for YOLOv8x and activity recognition
def preprocess_frame(frame):
    # Resize the frame to the required input size (e.g., 224x224 or YOLOv8x's input size)
    frame_resized = cv2.resize(frame, (224, 224))  # Example size, change based on your model
    
    # Convert the frame to float32 for YOLOv8x and normalize it
    frame_resized = tf.cast(frame_resized, tf.float32)  # Convert to float32
    frame_resized /= 255.0  # Normalize pixel values to range [0, 1]

    # Expand dimensions to match model input shape (e.g., batch size of 1)
    frame_resized = np.expand_dims(frame_resized, axis=0)

    return frame_resized

# Example main loop where object detection and activity recognition are performed
def main():
    # Example to capture video from the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera index

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        # Preprocess the frame for both YOLOv8x and activity recognition
        model_input = preprocess_frame(frame)

        # Run YOLOv8x object detection
        # Example: Detect objects using the YOLOv8x model
        # detections = yolo_model.detect(model_input)  # Use your YOLOv8x detection function

        # Run activity recognition
        try:
            # Assuming your activity model expects a certain signature
            predictions = activity_model.signatures['default'](model_input)
            print(predictions)
        except Exception as e:
            print(f"Error during activity recognition: {e}")

        # Show the frame (optional, for visualization)
        cv2.imshow("Activity Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
