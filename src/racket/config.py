"""Tunable numbers and file paths for the whole pipeline."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
VIDEO_DIR = ROOT / "video"
OUT_DIR = ROOT / "out"

IMU_CSV = DATA_DIR / "raw_data.csv"
ANCHORS_JSON = DATA_DIR / "anchors.json"
VIDEOS = {"front": VIDEO_DIR / "swing_angle_1.mp4", "rear": VIDEO_DIR / "swing_angle_2.mp4"}

# --- IMU -------------------------------------------------------------------
FS_HZ = 416.0
IMPACT_SAMPLE = 200
ACCEL_RANGE_G = 16.0
ACCEL_CLIPPED = (182, 199)        # inclusive, ax saturated
GYRO_UNRELIABLE = (200, 215)      # inclusive, post-impact transient

# --- Video -----------------------------------------------------------------
IMPACT_FRAME = {"front": 131, "rear": 172}
FRAME_OFFSET_REAR = 41            # rear frame = front frame + 41
IMAGE_SIZE = (1280, 720)

# Label fixes for hand-clicked anchors: {(view, frame): {true_name: clicked_name}}.
# Front frame 118 has its labels rotated by one (verified visually in stage 1).
ANCHOR_RELABEL = {
    ("front", 118): {"butt": "edge", "throat": "butt", "tip": "throat", "edge": "tip"},
}

# --- Racket geometry, sensor frame (metres), origin at throat ---------------
# X along handle toward butt cap, Y across string face, Z normal to face.
POINT_NAMES = ("butt", "throat", "tip", "edge")
RACKET_POINTS = {
    "butt": np.array([0.36, 0.0, 0.0]),
    "throat": np.array([0.0, 0.0, 0.0]),
    "tip": np.array([-0.33, 0.0, 0.0]),
    "edge": np.array([-0.165, 0.13, 0.0]),
}
RACKET_LENGTH_M = 0.685
HEAD_LENGTH_M = 0.33
HEAD_WIDTH_M = 0.26
CENTRIPETAL_RADIUS_M = 0.65
