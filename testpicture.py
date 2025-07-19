import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg')  # Replace 'image.jpg' with your image path

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detector
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours
for contour in contours:
    # Approximate the contour to a polygon with less vertices
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # If the polygon has 4 vertices, it is a square (or rectangle)
    if len(approx) == 4:
        # Draw the contour and the square on the image
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

        # Get the coordinates of the square's corners and draw them
        for point in approx:
            cv2.circle(image, tuple(point[0]), 5, (0, 0, 255), -1)

# Display the output image with squares detected
cv2.imshow("Squares Detected", image)

# Wait until a key is pressed, then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
