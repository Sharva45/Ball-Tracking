import cv2
import numpy as np
from ultralytics import YOLO
from kalmanfilter import KalmanFilter

# Initialize YOLO model (yolov8n.pt will auto-download on first run)
model = YOLO("yolov8n.pt")

# Initialize Kalman Filter
kf = KalmanFilter()

# Open camera stream (0 for default webcam)
cap = cv2.VideoCapture(0)

trajectory_points = []

# Class ID for 'sports ball' in COCO dataset
SPORTS_BALL_CLASS_ID = 32

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame (stream=True for faster live streaming)
    results = model(frame, stream=True, verbose=False)

    best_ball = None
    max_conf = 0.0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Filter only for 'sports ball' class with confidence > 0.3
            if cls_id == SPORTS_BALL_CLASS_ID and conf > 0.3:
                if conf > max_conf:
                    max_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_ball = (x1, y1, x2, y2)

    # If YOLO detected a ball in this frame
    if best_ball is not None:
        x1, y1, x2, y2 = best_ball
        
        # Calculate bounding box center point (center_x, center_y)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Draw YOLO Detection (Red Bounding Box and Center)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Ball {max_conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Predict next location using Kalman Filter
        predicted_x, predicted_y = kf.predict(center_x, center_y)

        # Draw Kalman Predicted Position (Blue Circle)
        cv2.circle(frame, (predicted_x, predicted_y), 15, (255, 0, 0), 2)
        cv2.putText(frame, "Predicted", (predicted_x + 10, predicted_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Save trajectory history
        trajectory_points.append((center_x, center_y))

    # Draw past trajectory path (Green line trail)
    for i in range(1, len(trajectory_points)):
        if trajectory_points[i - 1] is None or trajectory_points[i] is None:
            continue
        cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 2)

    # Maintain maximum length of trajectory line
    if len(trajectory_points) > 30:
        trajectory_points.pop(0)

    cv2.imshow("Cricket Ball Tracking (YOLO + Kalman Filter)", frame)

    if cv2.waitKey(1) == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()