import cv2
import numpy as np
from collections import deque

class Underwater_object_detection:
    def __init__(self):
        # Keep track of recent positions for smooth path display
        self.recent_positions = deque(maxlen=15)
        
    def identify_contour_shape(self, contour):
        # Simplify contour to determine shape based on corner points
        precision = 0.04 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, precision, True)

        # Categorize shape by number of corners
        corner_count = len(simplified)
        if corner_count == 3:
            return "Triangle"
        elif corner_count == 4:
            # Check if it's a square or rectangle
            x, y, width, height = cv2.boundingRect(simplified)
            ratio = float(width) / height
            if 0.95 <= ratio <= 1.05:
                return "Square"
            else:
                return "Rectangle"
        elif corner_count == 5:
            return "Pentagon"
        elif corner_count == 6:
            return "Hexagon"
        elif corner_count > 6:
            return "Circle"
        return "Unidentified"

    def find_underwater_items(self, video_frame, target_shapes):
        # Prepare image for analysis
        grayscale = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.GaussianBlur(grayscale, (5, 5), 0)
        outlines = cv2.Canny(smoothed, 50, 150)
        
        # Locate all contours in the processed image
        all_contours, _ = cv2.findContours(outlines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        found_items = []
        
        for contour in all_contours:
            # Ignore small contours that might be noise
            if cv2.contourArea(contour) < 500:
                continue
                
            shape_name = self.identify_contour_shape(contour)
            if shape_name in target_shapes:
                found_items.append(shape_name)
                
                # Mark the detected shape
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(video_frame, [contour], -1, (0, 255, 0), 3)
                cv2.rectangle(video_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(video_frame, shape_name, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return found_items, video_frame

    def track_navigation_path(self, video_frame):
        
        hsv_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)
        
        
        color_low = np.array([35, 50, 50])
        color_high = np.array([85, 255, 255])
        color_mask = cv2.inRange(hsv_frame, color_low, color_high)
        
        
        filter_kernel = np.ones((5,5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, filter_kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, filter_kernel)
        
        path_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if path_contours:
            
            main_path = max(path_contours, key=cv2.contourArea)
            
            
            if cv2.contourArea(main_path) > 1000:
                moments = cv2.moments(main_path)
                if moments["m00"] != 0:
                    center_x = int(moments["m10"] / moments["m00"])
                    center_y = int(moments["m01"] / moments["m00"])
                    
                    
                    self.recent_positions.append((center_x, center_y))
                    
                    
                    for point in range(1, len(self.recent_positions)):
                        cv2.line(video_frame, self.recent_positions[point-1], 
                                self.recent_positions[point], (0, 255, 255), 3)
                    
                    
                    cv2.circle(video_frame, (center_x, center_y), 10, (0, 0, 255), -1)
                    
                    
                    screen_center = video_frame.shape[1] // 2
                    if center_x < screen_center - 50:
                        cv2.putText(video_frame, "Adjust Right", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif center_x > screen_center + 50:
                        cv2.putText(video_frame, "Adjust Left", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(video_frame, "On Course", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return video_frame

    def create_status_display(self, video_frame, detected_items):
        """Add informational overlay with detection results"""
        
        background = video_frame.copy()
        cv2.rectangle(background, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(background, 0.7, video_frame, 0.3, 0, video_frame)
        
        
        cv2.putText(video_frame, f"Items found: {len(detected_items)}", (20, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        
        for index, item in enumerate(detected_items):
            cv2.putText(video_frame, item, (20, 60 + index*25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return video_frame

def start_navigation_system():
    navigation = Underwater_object_detection()        
    camera = cv2.VideoCapture(0)        
    target_configurations = ["Triangle", "Square", "Circle", "Hexagon"]

    while True:
        success, current_frame = camera.read()
        if not success:
            break

        found_objects, processed_frame = navigation.find_underwater_items(current_frame, target_configurations)
        navigated_frame = navigation.track_navigation_path(processed_frame)
        complete_frame = navigation.create_status_display(navigated_frame, found_objects)        
        cv2.imshow('Underwater Navigation Interface', complete_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_navigation_system()