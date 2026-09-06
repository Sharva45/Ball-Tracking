import cv2
import numpy as np
import math

# ============================================================
# FILES
# ============================================================

VIDEO = "tennis.mp4"

# Optional. Put ball_zoomed.png in the same folder.
TEMPLATE = "ball_zoomed.png"


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

FPS = cap.get(cv2.CAP_PROP_FPS)

if FPS <= 0:
    FPS = 30.0

TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", FPS)
print("Frames:", TOTAL_FRAMES)


# ============================================================
# COURT REGION
# ============================================================
#
# These points are based on YOUR video.
#
# We deliberately exclude:
# - scoreboard
# - chairs
# - spectators
# - walls
# - most of the green area outside the court
#
# You can tune these four/six points later.
#
# The playable court is approximately inside this polygon.
#

COURT_POLYGON = np.array([
    [0,   440],
    [0,   285],
    [360, 105],
    [970, 105],
    [1328, 285],
    [1328, 580],
    [0,   580]
], dtype=np.int32)


# ============================================================
# TENNIS BALL COLOR
# ============================================================

# Tennis ball in this video is yellow/green.
#
# We intentionally require:
#   reasonably high saturation
#   reasonably high brightness
#
# This is NOT the entire green HSV range.

LOWER_YELLOW = np.array([24, 80, 120])
UPPER_YELLOW = np.array([42, 255, 255])


# ============================================================
# BALL SIZE
# ============================================================

# In your video the ball can be extremely small.
#
# These limits are deliberately conservative.

MIN_AREA = 3
MAX_AREA = 180

MIN_WIDTH = 2
MAX_WIDTH = 22

MIN_HEIGHT = 2
MAX_HEIGHT = 22


# ============================================================
# MOTION
# ============================================================

MOTION_THRESHOLD = 18


# ============================================================
# TRACKING
# ============================================================

MAX_MISSED_FRAMES = 8

SEARCH_RADIUS = 180


# ============================================================
# KALMAN FILTER
# State:
#
# x
# y
# vx
# vy
# ============================================================

kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)

dt= 1.0

kalman.transitionMatrix = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

kalman.processNoiseCov = np.array([
    [1e-2, 0,    0,    0],
    [0,    1e-2, 0,    0],
    [0,    0,    5e-2, 0],
    [0,    0,    0,    5e-2]
], np.float32)

kalman.measurementNoiseCov = np.array([
    [2, 0],
    [0, 2]
], np.float32)

kalman.errorCovPost = np.eye(4, dtype=np.float32)


# ============================================================
# VARIABLES
# ============================================================

tracking = False

ball_x = None
ball_y = None

missed = 0

previous_gray = None

paused = False

frame_number = 0


# ============================================================
# FUNCTIONS
# ============================================================

def inside_court(x, y):
    """
    Check whether point lies inside court polygon.
    """

    return cv2.pointPolygonTest(
        COURT_POLYGON,
        (float(x), float(y)),
        False
    ) >= 0


def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def get_candidates(frame, previous_gray, predicted):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --------------------------------------------------------
    # COLOR MASK
    # --------------------------------------------------------

    mask = cv2.inRange(
        hsv,
        LOWER_YELLOW,
        UPPER_YELLOW
    )

    # Remove everything outside court

    court_mask = np.zeros_like(mask)

    cv2.fillPoly(
        court_mask,
        [COURT_POLYGON],
        255
    )

    mask = cv2.bitwise_and(
        mask,
        court_mask
    )

    # Very small morphological operation.
    #
    # Don't blur heavily because the ball is tiny.

    kernel = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # MOTION MASK
    # --------------------------------------------------------

    motion = None

    if previous_gray is not None:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        diff = cv2.absdiff(
            previous_gray,
            gray
        )

        _, motion = cv2.threshold(
            diff,
            MOTION_THRESHOLD,
            255,
            cv2.THRESH_BINARY
        )

        # Don't heavily dilate.
        # A tennis ball may only occupy a few pixels.

        motion = cv2.morphologyEx(
            motion,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8)
        )

    # --------------------------------------------------------
    # FIND COLOR OBJECTS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_WIDTH or w > MAX_WIDTH:
            continue

        if h < MIN_HEIGHT or h > MAX_HEIGHT:
            continue

        cx = x + w / 2
        cy = y + h / 2

        if not inside_court(cx, cy):
            continue

        # ----------------------------------------------------
        # MOTION SCORE
        # ----------------------------------------------------

        motion_score = 0

        if motion is not None:

            x1 = max(0, x - 3)
            y1 = max(0, y - 3)

            x2 = min(frame.shape[1], x + w + 3)
            y2 = min(frame.shape[0], y + h + 3)

            motion_roi = motion[y1:y2, x1:x2]

            if motion_roi.size > 0:

                motion_pixels = np.count_nonzero(
                    motion_roi
                )

                motion_score = min(
                    motion_pixels / 20.0,
                    1.0
                )

        # ----------------------------------------------------
        # CIRCULARITY
        # ----------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter > 0:

            circularity = (
                4 * math.pi * area
                / (perimeter * perimeter)
            )

        else:
            circularity = 0

        # Tiny blurred balls won't necessarily be circular,
        # so don't make this a hard rejection.

        shape_score = min(
            circularity / 0.75,
            1.0
        )

        # ----------------------------------------------------
        # ASPECT RATIO
        # ----------------------------------------------------

        aspect = min(w, h) / max(w, h)

        aspect_score = aspect

        # ----------------------------------------------------
        # PREDICTION DISTANCE
        # ----------------------------------------------------

        prediction_score = 0.0

        if predicted is not None:

            d = distance(
                (cx, cy),
                predicted
            )

            prediction_score = max(
                0.0,
                1.0 - d / SEARCH_RADIUS
            )

        # ----------------------------------------------------
        # COMBINED SCORE
        # ----------------------------------------------------

        score = (
            0.35 * motion_score +
            0.25 * prediction_score +
            0.20 * shape_score +
            0.20 * aspect_score
        )

        candidates.append({
            "x": cx,
            "y": cy,
            "w": w,
            "h": h,
            "score": score,
            "motion": motion_score,
            "shape": shape_score,
            "prediction": prediction_score
        })

    return candidates


# ============================================================
# RESET
# ============================================================

def reset_tracker():

    global tracking
    global ball_x
    global ball_y
    global missed
    global previous_gray

    tracking = False

    ball_x = None
    ball_y = None

    missed = 0

    previous_gray = None

    kalman.statePost = np.zeros(
        (4, 1),
        dtype=np.float32
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    if not paused:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number = int(
            cap.get(cv2.CAP_PROP_POS_FRAMES)
        )

        # ----------------------------------------------------
        # RAW GRAYSCALE
        # IMPORTANT:
        #
        # We save this BEFORE drawing anything.
        # Otherwise our green/red tracking boxes become
        # part of the motion detector.
        # ----------------------------------------------------

        raw_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # KALMAN PREDICTION
        # ----------------------------------------------------

        predicted = None

        if tracking:

            prediction = kalman.predict()

            predicted = (
                float(prediction[0, 0]),
                float(prediction[1, 0]
)
            )

        # ----------------------------------------------------
        # DETECT CANDIDATES
        # ----------------------------------------------------

        candidates = get_candidates(
            frame,
            previous_gray,
            predicted
        )

        best = None

        # ----------------------------------------------------
        # SELECT BEST CANDIDATE
        # ----------------------------------------------------

        if len(candidates) > 0:

            candidates.sort(
                key=lambda c: c["score"],
                reverse=True
            )

            # If already tracking, strongly prefer
            # candidates close to predicted position.

            if tracking and predicted is not None:

                nearby = [
                    c for c in candidates
                    if distance(
                        (c["x"], c["y"]),
                        predicted
                    ) < SEARCH_RADIUS
                ]

                if nearby:

                    nearby.sort(
                        key=lambda c: c["score"],
                        reverse=True
                    )

                    best = nearby[0]

            else:

                best = candidates[0]

        # ----------------------------------------------------
        # UPDATE TRACKER
        # ----------------------------------------------------

        if best is not None:

            bx = best["x"]
            by = best["y"]

            measurement = np.array([
                [np.float32(bx)],
                [np.float32(by)]
            ])

            if not tracking:

                # Initialize Kalman filter

                kalman.statePost = np.array([
                    [bx],
                    [by],
                    [0],
                    [0]
                ], dtype=np.float32)

                tracking = True

            kalman.correct(measurement)

            ball_x = bx
            ball_y = by

            missed = 0

        else:

            missed += 1

            if missed > MAX_MISSED_FRAMES:

                tracking = False
                ball_x = None
                ball_y = None

        # ----------------------------------------------------
        # DRAW ONLY THE BALL
        # ----------------------------------------------------

        if best is not None:

            x = int(best["x"])
            y = int(best["y"])

            w = int(best["w"])
            h = int(best["h"])

            # Make a larger visible box around tiny ball

            box_size = max(
                24,
                int(max(w, h) * 2.5)
            )

            half = box_size // 2

            x1 = max(0, x - half)
            y1 = max(0, y - half)

            x2 = min(
                frame.shape[1],
                x + half
            )

            y2 = min(
                frame.shape[0],
                y + half
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"BALL ({x}, {y})",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_number}/{TOTAL_FRAMES}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {FPS:.0f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if tracking:

            cv2.putText(
                frame,
                "TRACKING",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "SEARCHING",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # SAVE RAW FRAME FOR NEXT MOTION CALCULATION
        # ----------------------------------------------------

        previous_gray = raw_gray

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Tennis Ball Tracking",
        frame
    )

    # ========================================================
    # ORIGINAL VIDEO SPEED
    #
    # 30 FPS -> 33 ms per frame
    #
    # NO SPEED MULTIPLIER
    # ========================================================

    key = cv2.waitKey(
        max(1, int(1000 / FPS))
    ) & 0xFF

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    if key == ord('q') or key == 27:

        break

    elif key == ord(' '):

        paused = not paused

    elif key == ord('r'):

        reset_tracker()

    # A = previous frame
    elif key == ord('a'):

        paused = True

        current = int(
            cap.get(cv2.CAP_PROP_POS_FRAMES)
        )

        new_frame = max(
            0,
            current - 2
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            new_frame
        )

        reset_tracker()

    # D = next frame
    elif key == ord('d'):

        paused = True

        reset_tracker()

    # ========================================================
    # SPACE / A / D
    #
    # No playback-speed changes.
    # ========================================================


cap.release()
cv2.destroyAllWindows()