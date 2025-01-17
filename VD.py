import cv2
import numpy as np

# Open video file
cap = cv2.VideoCapture("video.mp4")

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Constants for vehicle detection
min_width_rect = 80  # min width of rectangle
min_height_rect = 80  # min height of rectangle
count_line_position = 550
offset = 6  # Allowable error between pixel
counter = 0

# Initialize Subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Function to find the center of a rectangle
def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# List to store detected centers
detect = []

while True:
    ret, frame1 = cap.read()
    if not ret:
        break  # Exit loop if no frame is read
    
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    
    # Apply background subtraction
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw counting line
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for contour in contours:
        if cv2.contourArea(contour) < min_width_rect * min_height_rect:
            continue  # Ignore small contours that are not likely to be vehicles

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)
    
    # Count vehicles
    for center in detect:
        cx, cy = center
        if (count_line_position - offset) < cy < (count_line_position + offset):
            counter += 1
            cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
            detect.remove(center)
    
    
    # Display vehicle counter
    cv2.putText(frame1, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Show video frame
    cv2.imshow('video Original', frame1)

    if cv2.waitKey(1) == 13:  # Press 'Enter' to break the loop
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
print("Total Vehicle Counted:" + str(counter))
