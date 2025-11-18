import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# PID control variables
Kp = 0.5  # Proportional gain for steering
Kd = 0.2  # Derivative gain for steering
previous_error = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define the region of interest (ROI) to focus on the track
    height, width = edges.shape
    roi = edges[int(height/2):height, 0:width]  # Focus on the bottom half of the image

    # Find contours (edges of the track)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are found, process the largest one (the path)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the centroid of the contour (this represents the center of the path)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Visualize the center of the line
            cv2.circle(frame, (cX, cY), 7, (0, 255, 0), -1)
            cv2.putText(frame, f"Center: ({cX}, {cY})", (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate steering angle (error in X direction)
            error = cX - width // 2  # Error between the center of the line and the center of the image

            # PID control for steering
            steering_adjustment = Kp * error + Kd * (error - previous_error)
            previous_error = error
            
            # Adjust motor speeds based on steering adjustment (send commands to Teensy or motor driver)
            if steering_adjustment < 0:
                print("Turn Left")
                # Send motor command to turn left
            elif steering_adjustment > 0:
                print("Turn Right")
                # Send motor command to turn right
            else:
                print("Move Forward")
                # Send motor command to move forward

    # Show the processed frame
    cv2.imshow('Path Detection', frame)

    # Wait for key press to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Assume obstacles are a specific color (e.g., red or green)
# Convert the image to HSV for better color detection
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define the range of color (e.g., green for obstacles)
lower_bound = np.array([35, 100, 100])  # Lower bound for green in HSV
upper_bound = np.array([85, 255, 255])  # Upper bound for green in HSV

# Create a mask for the specific color (obstacle)
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Find contours of the obstacles
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > 500:  # Minimum area for obstacle detection
        # Draw bounding box around the obstacle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Obstacle detected, send motor control command to stop or avoid
        print("Obstacle Detected!")
        # Send command to stop or steer around the obstacle

# Show the mask and the frame with detected obstacles
cv2.imshow("Obstacle Detection", mask)
cv2.imshow("Frame with Obstacles", frame)
# Find edges using the same Canny edge detection as before
edges = cv2.Canny(blurred, 50, 150)

# Hough Line Transform to find lines in the image
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate the slope angle based on the line's theta
        angle = np.degrees(theta)
        if angle < 45 or angle > 135:
            print("Climbing or descending slope detected!")
            # Adjust motor speed for climbing or descending
cap.release()
cv2.destroyAllWindows()
