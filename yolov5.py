import torch

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can replace 'yolov5s' with other models like 'yolov5m'
import cv2
import torch

# Initialize YOLOv5 model (pretrained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Using YOLOv5 small model

# Function to perform object detection
def detect_objects(frame):
    # Perform inference on the image
    results = model(frame)

    # Results
    labels = results.names  # List of object names
    detected_objects = results.xywh[0]  # Bounding boxes (x, y, width, height)

    # Filter only relevant objects (e.g., shapes, path, etc.)
    for obj in detected_objects:
        x_center, y_center, width, height, conf, label_id = obj[:6]
        label = labels[int(label_id)]  # Label for the object
        print(f"Detected {label} with confidence {conf:.2f}")
        
        # Draw the bounding boxes on the frame
        x1, y1, x2, y2 = int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(y_center + height / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Main function to process video and detect objects
def main():
    cap = cv2.VideoCapture(0)  # Webcam or replace with video file
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects using YOLOv5
        frame_with_detections = detect_objects(frame)
        
        # Show the resulting frame
        cv2.imshow('YOLOv5 Object Detection', frame_with_detections)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
