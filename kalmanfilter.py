import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # State transition matrix (x_k = x_{k-1} + v*dt)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # Measurement matrix (we only measure position x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # Noise covariance matrices
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

    def predict(self, coordX, coordY):
        """Corrects state with actual measurements and predicts the next position."""
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0][0]), int(predicted[1][0])
        return x, y