import cv2
import numpy as np

# Load image
img = cv2.imread("ball.png")

if img is None:
    print("Error: Could not load ball.png")
    exit()

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Tennis ball HSV range
lower_ball = np.array([25, 30, 120])
upper_ball = np.array([65, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_ball, upper_ball)

# Remove noise
kernel = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

best_contour = None
best_area = 0

# Find the largest suitable object
for contour in contours:

    area = cv2.contourArea(contour)

    if area < 5:
        continue

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        continue

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Tennis ball should be roughly circular
    if circularity > 0.4 and area > best_area:
        best_area = area
        best_contour = contour


# If ball found
if best_contour is not None:

    # Minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(best_contour)

    center = (int(x), int(y))
    radius = int(radius)

    print("Tennis ball detected!")
    print("Center:", center)
    print("Radius:", radius)

    # Draw circle around ball
    cv2.circle(
        img,
        center,
        radius,
        (0, 255, 0),
        2
    )

    # Draw center
    cv2.circle(
        img,
        center,
        3,
        (0, 0, 255),
        -1
    )

    # Display coordinates
    cv2.putText(
        img,
        f"Ball: {center}",
        (center[0] + 10, center[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

else:
    print("Tennis ball not detected")


# Show results
cv2.imshow("Tennis Ball Detection", img)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()