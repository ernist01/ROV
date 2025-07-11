import cv2
import numpy as np

# Shape detection function
def detect_shape(cnt):
    # Approximate the contour
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Return shape based on the number of vertices
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        # Check if it's a square or rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif len(approx) == 5:
        return "Pentagon"
    elif len(approx) == 6:
        return "Hexagon"
    elif len(approx) > 6:
        return "Circle"  # We assume objects with more than 6 vertices are circles/ellipses
    return "Unknown"

# Function to detect the objects in the image
def detect_objects(frame, objects_to_detect):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_objects = []
    
    for cnt in contours:
        shape = detect_shape(cnt)
        if shape in objects_to_detect:
            detected_objects.append(shape)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)  # Draw detected object
        
    return detected_objects, frame

# Path following function
def follow_path(frame):
    # Assuming the path is a distinct color or brightness range, use color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([35, 50, 50])  # Lower bound for color (adjust for your path)
    upper_bound = np.array([85, 255, 255])  # Upper bound for color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours along the path
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assuming it's the path)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Draw a center point to track
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            
            # Logic for ROV to follow the path (based on cx, cy) can be added here
            
    return frame

# Main function
def main():
    # Initialize webcam or video feed (simulate ROV camera)
    cap = cv2.VideoCapture(0)  # or replace with video file
    
    # Define the objects to detect
    objects_to_detect = ["Triangle", "Square", "Circle", "Hexagon"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        detected_objects, frame_with_objects = detect_objects(frame, objects_to_detect)
        print(f"Detected Objects: {detected_objects}")

        # Path Following Logic
        frame_with_path = follow_path(frame_with_objects)

        # Show the processed frame
        cv2.imshow('ROV View', frame_with_path)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
