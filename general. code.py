import cv2
import numpy as np
import math
import time

# Constants for path following
CABLE_COLOR_LOWER = (0, 0, 0)    # Black cable - adjust based on actual color
CABLE_COLOR_UPPER = (50, 50, 50) # Black cable - adjust based on actual color
MIN_CABLE_WIDTH = 10             # Minimum expected cable width in pixels
MAX_CABLE_WIDTH = 100            # Maximum expected cable width in pixels
CENTER_DEADZONE = 50             # Deadzone in pixels for centering
MAX_TURN_ANGLE = 30              # Maximum turn angle in degrees

# Object detection parameters
OBJECT_COLORS = {
    'red': ((0, 100, 100), (10, 255, 255)),
    'blue': ((100, 100, 100), (140, 255, 255)),
    'green': ((40, 100, 100), (80, 255, 255)),
    'yellow': ((20, 100, 100), (40, 255, 255))
}
MIN_OBJECT_AREA = 500
OBJECTS_TO_DETECT = ['triangle', 'square', 'circle', 'cylinder']  # The 4 objects we need to detect

class ROVController:
    def __init__(self):
        self.detected_objects = []
        self.cap = cv2.VideoCapture(0)  # Adjust for underwater camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize thruster controls (placeholder values)
        self.left_thruster = 0
        self.right_thruster = 0
        self.vertical_thruster = 0
        
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Preprocess the frame
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Process for cable following
        cable_mask = cv2.inRange(blurred, CABLE_COLOR_LOWER, CABLE_COLOR_UPPER)
        cable_mask = cv2.erode(cable_mask, None, iterations=2)
        cable_mask = cv2.dilate(cable_mask, None, iterations=2)
        
        # Find cable contours
        cable_contours, _ = cv2.findContours(cable_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cable_contours) > 0:
            # Find the largest contour (assuming it's the cable)
            largest_contour = max(cable_contours, key=cv2.contourArea)
            
            # Get the bounding rectangle of the cable
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Check if the cable width is within expected range
            if MIN_CABLE_WIDTH < w < MAX_CABLE_WIDTH:
                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate center of the cable and frame
                cable_center = x + w // 2
                frame_center = frame.shape[1] // 2
                
                # Calculate deviation from center
                deviation = cable_center - frame_center
                
                # Adjust thrusters based on deviation
                self.adjust_thrusters(deviation)
                
                # Draw center line and deviation
                cv2.line(frame, (frame_center, 0), (frame_center, frame.shape[0]), (255, 0, 0), 2)
                cv2.line(frame, (cable_center, y + h // 2), (frame_center, y + h // 2), (0, 0, 255), 2)
                
                # Add text for deviation
                cv2.putText(frame, f"Deviation: {deviation}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Process for object detection
        self.detect_objects(frame, hsv)
        
        # Display thruster values
        self.display_thruster_values(frame)
        
        return frame
    
    def adjust_thrusters(self, deviation):
        """Adjust thruster outputs based on cable deviation"""
        if abs(deviation) < CENTER_DEADZONE:
            # Within deadzone - go straight
            self.left_thruster = 0.7
            self.right_thruster = 0.7
        else:
            # Outside deadzone - turn
            turn_ratio = min(abs(deviation) / (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), 1.0)
            turn_direction = 1 if deviation > 0 else -1
            
            # Calculate thruster outputs
            base_power = 0.5
            turn_power = 0.3 * turn_ratio
            
            self.left_thruster = base_power - (turn_power * turn_direction)
            self.right_thruster = base_power + (turn_power * turn_direction)
            
            # Ensure thruster values stay within 0-1 range
            self.left_thruster = max(0, min(1, self.left_thruster))
            self.right_thruster = max(0, min(1, self.right_thruster))
    
    def detect_objects(self, frame, hsv_frame):
        """Detect the specified objects in the frame"""
        for color_name, (lower, upper) in OBJECT_COLORS.items():
            # Create mask for the color
            color_mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
            color_mask = cv2.erode(color_mask, None, iterations=2)
            color_mask = cv2.dilate(color_mask, None, iterations=2)
            
            # Find contours for the color
            contours, _ = cv2.findContours(color_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter small contours
                if cv2.contourArea(contour) < MIN_OBJECT_AREA:
                    continue
                
                # Approximate the contour to a polygon
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip if the object is too small or too large
                if w < 20 or h < 20 or w > 300 or h > 300:
                    continue
                
                # Calculate the aspect ratio
                aspect_ratio = float(w) / h
                
                # Detect shape based on number of vertices and aspect ratio
                shape = "unknown"
                vertices = len(approx)
                
                if vertices == 3:
                    shape = "triangle"
                elif vertices == 4:
                    # Could be square, rectangle, or cylinder (if 3D)
                    if 0.9 <= aspect_ratio <= 1.1:
                        shape = "square"
                    else:
                        shape = "rectangle"
                elif vertices == 5:
                    shape = "pentagon"
                elif vertices == 6:
                    shape = "hexagon"
                else:
                    # For circle, ellipse, cylinder (appears as ellipse from side), clover
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * math.pi * (area / (perimeter * perimeter))
                    
                    if circularity > 0.8:
                        shape = "circle"
                    elif circularity > 0.6:
                        shape = "ellipse"
                    else:
                        # Check for four-leaf clover (complex shape)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area) / hull_area
                        
                        if solidity < 0.7 and vertices > 6:
                            shape = "four leaf clover"
                
                # Check if the detected shape is one we're looking for
                if shape in OBJECTS_TO_DETECT and shape not in [obj['shape'] for obj in self.detected_objects]:
                    # Add to detected objects
                    self.detected_objects.append({
                        'shape': shape,
                        'color': color_name,
                        'position': (x + w // 2, y + h // 2),
                        'time': time.time()
                    })
                    
                    # Draw the contour and label
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    cv2.putText(frame, f"{color_name} {shape}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def display_thruster_values(self, frame):
        """Display current thruster values on the frame"""
        cv2.putText(frame, f"Left Thruster: {self.left_thruster:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Right Thruster: {self.right_thruster:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Detected Objects: {len(self.detected_objects)}/4", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # List detected objects
        for i, obj in enumerate(self.detected_objects):
            cv2.putText(frame, f"{obj['shape']} ({obj['color']})", (10, 150 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        while True:
            frame = self.process_frame()
            if frame is None:
                break
                
            cv2.imshow("ROV View", frame)
            
            # Exit if 'q' is pressed or all objects detected
            if cv2.waitKey(1) & 0xFF == ord('q') or len(self.detected_objects) >= 4:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rov = ROVController()
    rov.run()