import cv2

# Load an image (you can replace 'image.jpg' with any image of your choice)
image = cv2.imread('image.jpg')

# Check if the image was successfully loaded
if image is None:
    print("Error: Could not read the image.")
    exit()

# Define the coordinates for the rectangle (x, y, width, height)
x, y, w, h = 50, 50, 200, 150  # Adjust these values as needed

# Draw a rectangle on the image: (x, y) is the top-left corner, (x+w, y+h) is the bottom-right corner
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the rectangle
cv2.imshow('Image with Rectangle', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
