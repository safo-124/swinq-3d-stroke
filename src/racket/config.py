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
# The brief's guess (butt +0.36, tip -0.33, edge -0.165/+0.13) does not match the
# clicks: in both views the clicked throat sits at 0.32 of butt->tip and the edge
# at 0.70. These starting values follow the images; stage 3 refines them
# (throat fraction, edge x, edge y) with the butt-tip length held fixed for scale.
RACKET_POINTS = {                 # stage-3 refined values (start: 0.223/-0.462/-0.257/0.13)
    "butt": np.array([0.231, 0.0, 0.0]),
    "throat": np.array([0.0, 0.0, 0.0]),
    "tip": np.array([-0.454, 0.0, 0.0]),
    "edge": np.array([-0.272, 0.149, 0.0]),
}
RACKET_LENGTH_M = 0.685
HEAD_LENGTH_M = 0.33
HEAD_WIDTH_M = 0.26
CENTRIPETAL_RADIUS_M = 0.65

# --- Bundle adjustment (stage 3) --------------------------------------------
BA_F_GRID = (1000.0, 2400.0)   # focal guesses, px (multistart; all tested starts converge)
BA_F_SCALE_PX = 3.0               # soft_l1 knee: residuals beyond ~3 px down-weighted
BA_INIT_INLIER_DEG = 15.0         # agreement threshold for camera-2 rotation consensus
PNP_MAX_RMS_PX = 15.0            # single-view pose candidates (init only) must fit this well
BA_REFINE_GEOMETRY = True        # free throat fraction + edge (x, y); length fixed

# --- Joint IMU/video fit (stage 4) -------------------------------------------
IMPACT_DOWNWEIGHT = (195, 230)    # moments with imu_sample in this range ...
IMPACT_WEIGHT = 0.2               # ... get this residual weight
JOINT_F_SCALE_DEG = 5.0           # soft_l1 knee on each error component
JOINT_MAX_DT_S = 0.05             # time-offset bound (+/- 1.5 video frames)
JOINT_DT_STARTS = (-10.0, 0.0, 10.0)   # time-offset starts, samples
JOINT_PAD_SAMPLES = 30            # gyro held constant this far beyond each end

# --- Sensor-only model (stage 4, revised) -------------------------------------
GYRO_SIGN = -1                    # sign -1 matches video over moments 0-4 (3-5 deg)
ANCHOR_MOMENTS = 5                # first video moments used for the start orientation
RADIUS_FIT_WINDOW = (0, 182)      # unclipped pre-impact samples for the pivot radius
AX_FIT_WINDOW = (170, 182)        # last unclipped samples for the saturation fill
