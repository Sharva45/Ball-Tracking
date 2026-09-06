import cv2
import numpy as np

# =========================================================
# FILES
# =========================================================

VIDEO = "tennis.mp4"
TEMPLATE = "ball_zoomed.png"


# =========================================================
# PARAMETERS
# =========================================================

# Color: yellow/green tennis ball
LOWER_HSV = np.array([25, 60, 100])
UPPER_HSV = np.array([65, 255, 255])

# Candidate size in pixels
MIN_RADIUS = 2
MAX_RADIUS = 25

# Motion threshold
MOTION_THRESHOLD = 15

# Minimum percentage of moving pixels
MIN_MOTION = 3

# Minimum template similarity
MIN_TEMPLATE_SCORE = 0.45

# Overall score required
MIN_TOTAL_SCORE = 0.50


# =========================================================
# LOAD VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()


# =========================================================
# LOAD TEMPLATE
# =========================================================

template = cv2.imread(TEMPLATE)

if template is None:
    print("ERROR: Could not load ball_zoomed.png")
    exit()

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

template_h, template_w = template_gray.shape


# =========================================================
# PREVIOUS FRAME
# =========================================================

previous_gray = None


# =========================================================
# VIDEO LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ---------------------------------------------
    # Resize processing image if desired
    # ---------------------------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # =====================================================
    # 1. COLOR DETECTION
    # =====================================================

    color_mask = cv2.inRange(
        hsv,
        LOWER_HSV,
        UPPER_HSV
    )

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    color_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # =====================================================
    # 2. MOTION DETECTION
    # =====================================================

    if previous_gray is None:

        previous_gray = gray.copy()

        cv2.imshow("Tennis Ball Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue


    frame_difference = cv2.absdiff(
        previous_gray,
        gray
    )

    motion_mask = cv2.threshold(
        frame_difference,
        MOTION_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )[1]

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_OPEN,
        kernel
    )


    # =====================================================
    # 3. FIND COLOR CANDIDATES
    # =====================================================

    contours, _ = cv2.findContours(
        color_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    candidates = []


    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 3:
            continue


        # =================================================
        # 4. SIZE
        # =================================================

        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue


        # =================================================
        # 5. SHAPE / CIRCULARITY
        # =================================================

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
            / (perimeter * perimeter)
        )

        # Players and court regions generally fail here
        if circularity < 0.45:
            continue


        # =================================================
        # 6. MOTION AROUND CANDIDATE
        # =================================================

        x, y, w, h = cv2.boundingRect(contour)

        # Slightly enlarge region
        pad = 5

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)

        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        motion_region = motion_mask[
            y1:y2,
            x1:x2
        ]

        if motion_region.size == 0:
            continue

        motion_percentage = (
            np.count_nonzero(motion_region)
            / motion_region.size
        ) * 100

        if motion_percentage < MIN_MOTION:
            continue


        # =================================================
        # 7. TEMPLATE MATCHING
        # =================================================

        # Create a region around candidate
        search_size = max(
            template_w,
            template_h,
            int(radius * 4)
        )

        sx1 = max(
            0,
            int(cx - search_size)
        )

        sy1 = max(
            0,
            int(cy - search_size)
        )

        sx2 = min(
            frame.shape[1],
            int(cx + search_size)
        )

        sy2 = min(
            frame.shape[0],
            int(cy + search_size)
        )

        roi = gray[
            sy1:sy2,
            sx1:sx2
        ]

        if (
            roi.shape[0] < template_h
            or roi.shape[1] < template_w
        ):
            continue


        template_result = cv2.matchTemplate(
            roi,
            template_gray,
            cv2.TM_CCOEFF_NORMED
        )

        template_score = np.max(
            template_result
        )

        if template_score < MIN_TEMPLATE_SCORE:
            continue


        # =================================================
        # 8. CALCULATE FINAL SCORE
        # =================================================

        # Normalize circularity
        shape_score = min(
            circularity / 0.9,
            1.0
        )

        # Motion score
        motion_score = min(
            motion_percentage / 30.0,
            1.0
        )

        # Template score
        template_score_normalized = max(
            0,
            template_score
        )


        # Combined score
        total_score = (
            0.30 * shape_score +
            0.30 * motion_score +
            0.40 * template_score_normalized
        )


        candidates.append(
            {
                "cx": int(cx),
                "cy": int(cy),
                "radius": int(radius),
                "shape": shape_score,
                "motion": motion_score,
                "template": template_score,
                "score": total_score
            }
        )


    # =====================================================
    # 9. SELECT BEST BALL
    # =====================================================

    if candidates:

        candidates.sort(
            key=lambda c: c["score"],
            reverse=True
        )

        ball = candidates[0]


        if ball["score"] >= MIN_TOTAL_SCORE:

            cx = ball["cx"]
            cy = ball["cy"]
            radius = ball["radius"]


            # ---------------------------------------------
            # Square around ball
            # ---------------------------------------------

            size = max(
                radius * 2 + 10,
                12
            )

            x1 = int(cx - size / 2)
            y1 = int(cy - size / 2)

            x2 = int(cx + size / 2)
            y2 = int(cy + size / 2)


            # ---------------------------------------------
            # Draw detection
            # ---------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                3,
                (0, 0, 255),
                -1
            )


            # ---------------------------------------------
            # Display information
            # ---------------------------------------------

            text = (
                f"BALL "
                f"({cx},{cy}) "
                f"S:{ball['score']:.2f}"
            )

            cv2.putText(
                frame,
                text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Tennis Ball Tracking",
        frame
    )

    # Uncomment these if you want to debug
    # cv2.imshow("Color Mask", color_mask)
    # cv2.imshow("Motion Mask", motion_mask)


    # =====================================================
    # UPDATE PREVIOUS FRAME
    # =====================================================

    previous_gray = gray.copy()


    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()