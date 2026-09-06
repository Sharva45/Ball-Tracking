import cv2
import numpy as np

# ---------------------------------------
# Load images
# ---------------------------------------

image = cv2.imread("ball.png")
template = cv2.imread("ball_zoomed.png")

if image is None:
    print("Could not load ball.png")
    exit()

if template is None:
    print("Could not load ball_zoomed.png")
    exit()


# ---------------------------------------
# COURT ORIGIN
# ---------------------------------------
# Approximate bottom-left corner of court
origin_x = 0
origin_y = 466

# Draw origin point
cv2.circle(
    image,
    (origin_x, origin_y),
    7,
    (0, 0, 255),
    -1
)

# Draw origin label
cv2.putText(
    image,
    "ORIGIN (0,0)",
    (origin_x + 10, origin_y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)


# ---------------------------------------
# Template matching
# ---------------------------------------

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(
    gray_image,
    gray_template,
    cv2.TM_CCOEFF_NORMED
)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print("Match confidence:", max_val)


# ---------------------------------------
# Detect ball
# ---------------------------------------

threshold = 0.70

if max_val >= threshold:

    x = max_loc[0]
    y = max_loc[1]

    h, w = gray_template.shape

    # Ball center
    center_x = x + w // 2
    center_y = y + h // 2

    # -----------------------------------
    # Square around ball
    # -----------------------------------

    size = max(w, h) + 10

    x1 = center_x - size // 2
    y1 = center_y - size // 2

    x2 = x1 + size
    y2 = y1 + size

    # Draw square
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw ball center
    cv2.circle(
        image,
        (center_x, center_y),
        4,
        (255, 0, 0),
        -1
    )

    # -----------------------------------
    # Image coordinates
    # -----------------------------------

    cv2.putText(
        image,
        f"Ball pixel: ({center_x}, {center_y})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    print("BALL FOUND")
    print(f"Pixel position = ({center_x}, {center_y})")

else:

    print("Ball NOT found")


# ---------------------------------------
# Display
# ---------------------------------------

cv2.imshow("Tennis Ball Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()