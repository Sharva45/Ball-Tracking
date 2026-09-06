import cv2
import numpy as np

# Load image
img = cv2.imread("ball.png")

if img is None:
    print("Error: Could not load ball.png")
    exit()

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Tennis ball color range
lower_ball = np.array([30, 80, 150])
upper_ball = np.array([50, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_ball, upper_ball)

# Remove noise
kernel = np.ones((3, 3), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)

# Find contours
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

best_contour = None
best_area = 0

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 5:
        continue

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        continue

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    if circularity > 0.4 and area > best_area:
        best_area = area
        best_contour = contour


# ---------------------------------------
# Ball detected
# ---------------------------------------

if best_contour is not None:

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(best_contour)

    # Make it SQUARE
    size = max(w, h)

    # Center of detected object
    center_x = x + w // 2
    center_y = y + h // 2

    # Make square centered on ball
    x1 = center_x - size // 2
    y1 = center_y - size // 2

    x2 = x1 + size
    y2 = y1 + size

    # Draw square
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw center point
    cv2.circle(
        img,
        (center_x, center_y),
        3,
        (0, 0, 255),
        -1
    )

    # Display coordinates
    cv2.putText(
        img,
        f"Ball: ({center_x}, {center_y})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    print("Tennis ball detected")
    print(f"Center: ({center_x}, {center_y})")
    print(f"Bounding box: ({x1}, {y1}) -> ({x2}, {y2})")

else:

    print("Tennis ball not detected")


# Show result
cv2.imshow("Tennis Ball Detection", img)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()