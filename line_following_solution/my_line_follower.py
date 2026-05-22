#!/usr/bin/env python3
"""
HSHL Line Following Student Lab — Your Implementation
=====================================================

Implement your line following algorithm by filling in the function below:

    detect_line(image)  — called for every camera frame (~30 fps)

─────────────────────────────────────────────────────────────────────────────
INPUTS  (what you receive from the camera)
─────────────────────────────────────────────────────────────────────────────
Camera frame  →  detect_line(image)
  image         np.ndarray, shape (720, 1280, 3), BGR colour order
                Same convention as OpenCV.

─────────────────────────────────────────────────────────────────────────────
OUTPUTS  (what your function must return)
─────────────────────────────────────────────────────────────────────────────
detect_line(image)  →  float | None
  Return a steering value in range [-1.0, 1.0]:
    -1.0  = steer full left
     0.0  = go straight (line is centered)
    +1.0  = steer full right
        None  = cannot detect line (framework uses neutral steering fallback)

─────────────────────────────────────────────────────────────────────────────
ALGORITHM TIPS
─────────────────────────────────────────────────────────────────────────────
1. The line is painted GREEN on the road (BGR: 0, 255, 0)
2. Use color range thresholding to detect green pixels
3. Find the line center using contour moments
4. Compare line center to image center to get steering offset
5. Use morphological operations to reduce noise
6. Return None if no line is detected

See docs/line_detection_example.py for a complete example implementation.

─────────────────────────────────────────────────────────────────────────────
HELPERS
─────────────────────────────────────────────────────────────────────────────
    self.show_notification(text)  white  — general info
    self.show_warning(text)       yellow — caution
    self.show_alert(text)         red    — critical
    self.current_image            latest camera frame (or None)
"""
from pathlib import Path
from collections import deque

import cv2          # type: ignore
import numpy as np  # type: ignore
import rclpy        # type: ignore
import joblib       # type: ignore

from .interface import LineFollowingInterface


# ─────────────────────────────────────────────────────────────────────────────
# SVM MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# The trained SVM model is stored in:
#     models/svm_line_follower.pkl
#
# This model was trained using:
#   1. resized camera frames
#   2. cropped lower road region
#   3. grayscale conversion
#   4. Gaussian blur
#   5. Canny edge detection
#   6. SVM classification into left / straight / right
#
# Important:
# The preprocessing here must match the preprocessing used during training.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = Path("models/svm_line_follower.pkl")

if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "svm_line_follower.pkl"

MODEL = joblib.load(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# These values must match the training script.
# ─────────────────────────────────────────────────────────────────────────────

RESIZE_WIDTH = 80
RESIZE_HEIGHT = 40
ROI_START_RATIO = 0.45


# ─────────────────────────────────────────────────────────────────────────────
# STEERING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# The SVM predicts three classes:
#   -1 = left
#    0 = straight
#    1 = right
#
# These classes are converted into actual steering commands.
# Smoothing is applied to avoid sudden jumps between left / straight / right.
# ─────────────────────────────────────────────────────────────────────────────

LEFT_STEER = -0.30
STRAIGHT_STEER = 0.0
RIGHT_STEER = 0.30

SMOOTHING_WINDOW = 5
STEERING_HISTORY = deque(maxlen=SMOOTHING_WINDOW)


def preprocess_for_svm(image: np.ndarray) -> np.ndarray:
    """
    Preprocess camera image for the trained SVM model.

    Steps:
        1. Resize image
        2. Crop lower road region
        3. Convert to grayscale
        4. Apply Gaussian blur
        5. Apply Canny edge detection
        6. Flatten into feature vector

    Args:
        image: BGR image from camera

    Returns:
        Feature vector shaped as (1, n_features)
    """

    img = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))

    height = img.shape[0]
    roi = img[int(height * ROI_START_RATIO):, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    features = edges.flatten().astype(np.float32) / 255.0

    return features.reshape(1, -1)


def label_to_steering(label: int) -> float:
    """
    Convert SVM class label to steering command.

    Args:
        label: SVM output class (-1, 0, or 1)

    Returns:
        Steering value in range [-1.0, 1.0]
    """

    if label == -1:
        return LEFT_STEER
    elif label == 1:
        return RIGHT_STEER
    else:
        return STRAIGHT_STEER


def smooth_steering(new_steering: float) -> float:
    """
    Smooth steering using a moving average over the last few predictions.

    Args:
        new_steering: latest steering value

    Returns:
        Smoothed steering value
    """

    STEERING_HISTORY.append(new_steering)
    return float(np.mean(STEERING_HISTORY))


class MyLineFollower(LineFollowingInterface):
    """
    Student implementation of line following.
    
    Detect a green line and steer to stay centered on it.
    """

    def __init__(self):
        super().__init__("my_line_follower")
        self._frame_count = 0
        
        # Register camera callback
        self.on_camera_image(self.detect_line)
        self.get_logger().info("MyLineFollower initialized — ready to detect green line")

    def detect_line(self, image: np.ndarray) -> float | None:
        """
        Detect the green line and return steering command.
        
        Args:
            image: BGR image from camera, shape (720, 1280, 3)
        
        Returns:
            Steering value in [-1.0, 1.0], or None if line not detected.
        """

        if image is None:
            self.show_warning("No camera image received")
            return None

        # Preprocess the camera frame in the same way as during SVM training
        features = preprocess_for_svm(image)

        # Predict steering class:
        #   -1 = left
        #    0 = straight
        #    1 = right
        predicted_label = int(MODEL.predict(features)[0])

        # Convert class label into steering command
        raw_steering = label_to_steering(predicted_label)

        # Smooth steering to reduce sudden left/right jumps
        steering = smooth_steering(raw_steering)

        self._frame_count += 1

        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f"SVM MODE label={predicted_label} "
                f"raw_steer={raw_steering:.2f} "
                f"smooth_steer={steering:.2f} "
                f"frame={self._frame_count}"
            )

        self.show_notification(f"SVM steer={steering:.2f}")

        return steering


def main(args=None):
    """Main entry point for the line follower node."""
    rclpy.init(args=args)
    follower = MyLineFollower()
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        pass
    finally:
        follower.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
