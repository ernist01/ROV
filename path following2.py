import cv2
import numpy as np

# Define the pixel-to-centimeter ratio (this value needs to be determined by calibration)
pixel_to_cm_ratio = 0.1  # Adjust this based on your calibration

# Kalman Filter to smooth out the path tracking
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

    def correct(self, measurement):
        self.kf.correct(measurement)

    def predict(self):
        return self.kf.predict()

# Path following function
def follow_path(frame, kf):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds of the path color in HSV (e.g., green)
    lower_bound = np.array([35, 50, 50])  # Lower bound for the color (adjust for your path)
    upper_bound = np.array([85, 255, 255])  # Upper bound for the color (adjust for your path)
    
    # Create a mask for the path
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply adaptive thresholding to deal with varying lighting conditions
    mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Use Canny edge detection to detect edges
    edges = cv2.Canny(mask, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If we have detected contours
    if contours:
        # Find the largest contour (assuming it's the path)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the center of the largest contour (path)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Correct the measurement using the Kalman Filter
            kf.correct(np.array([cx, cy], dtype=np.float32))
            
            # Predict the next state (smooth the path)
            predicted = kf.predict()
            predicted_x, predicted_y = int(predicted[0]), int(predicted[1])
            
            # Draw the predicted center on the frame
            cv2.circle(frame, (predicted_x, predicted_y), 10, (0, 0, 255), -1)
            
            # Logic to steer the ROV towards the center of the path
            # Positive offset means steer left, negative offset means steer right
            height, width, _ = frame.shape
            offset = predicted_x - width // 2
            
            print(f"Offset: {offset}")  # Debugging info for steering
            
            # Here you can send commands to the ROV to steer it
            # Example: PID control can be added here to improve steering accuracy
            
    return frame

# Main function
def main():
    # Initialize the video capture (use a video file or webcam)
    cap = cv2.VideoCapture(0)  # Change this to a video file if needed

    # Initialize Kalman Filter for smooth tracking
    kf = KalmanFilter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Follow the path with advanced detection
        frame_with_path = follow_path(frame, kf)
        
        # Show the frame with detected path
        cv2.imshow('Advanced Path Following', frame_with_path)

        # Exit if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
