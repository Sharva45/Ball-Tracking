import cv2
import numpy as np
from kalmanfilter import KalmanFilter

# Initialize Kalman Filter
kf = KalmanFilter()

# Open live webcam feed (0 for default camera)
cap = cv2.VideoCapture(0)

# Define HSV threshold ranges for a Red Cricket Ball
# Note: Red wraps around in HSV color space (0-10 & 170-180)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# List to store historical tracking points for drawing trajectory lines
trajectory_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filter out red ball color mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Smooth the mask to reduce noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours of the detected ball
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    center_x, center_y = None, None

    if len(contours) > 0:
        # Find the largest contour assuming it's the cricket ball
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 8:  # Minimum size threshold
            center_x, center_y = int(x), int(y)
            
            # Draw real detection (Red Circle) [00:27:41]
            cv2.circle(frame, (center_x, center_y), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Predict next location using Kalman Filter [00:28:23]
            predicted_x, predicted_y = kf.predict(center_x, center_y)

            # Draw predicted position (Blue Circle) [00:28:46]
            cv2.circle(frame, (predicted_x, predicted_y), int(radius), (255, 0, 0), 2)
            cv2.putText(frame, "Predicted", (predicted_x + 10, predicted_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Track trajectory
            trajectory_points.append((center_x, center_y))

    # Draw line trail of actual movement trajectory
    for i in range(1, len(trajectory_points)):
        if trajectory_points[i - 1] is None or trajectory_points[i] is None:
            continue
        cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 2)

    # Limit trail length to avoid cluttering screen
    if len(trajectory_points) > 30:
        trajectory_points.pop(0)

    # Display video stream
    cv2.imshow("Cricket Ball Tracker & Trajectory Predictor", frame)

    # Press 'ESC' key to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()