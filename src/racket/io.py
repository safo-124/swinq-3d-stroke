"""Loaders for the IMU log, the hand-clicked anchors and the video frames."""
import json
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from . import config


@dataclass
class ImuData:
    """IMU samples. accel in g, gyro in deg/s, t in seconds from sample 0."""
    t: np.ndarray        # (N,)
    accel: np.ndarray    # (N, 3)
    gyro: np.ndarray     # (N, 3)

    @property
    def n(self) -> int:
        return len(self.t)


def load_imu(path=config.IMU_CSV, fs=config.FS_HZ) -> ImuData:
    """Read raw_data.csv; time is reconstructed from the sample index."""
    df = pd.read_csv(path)
    t = df["index"].to_numpy(float) / fs
    return ImuData(t=t,
                   accel=df[["ax", "ay", "az"]].to_numpy(float),
                   gyro=df[["gx", "gy", "gz"]].to_numpy(float))


@dataclass
class Anchors:
    """Clicked points for one view, aligned across views by moment index."""
    frames: np.ndarray        # (M,) decode frame indices
    t_from_impact: np.ndarray # (M,) seconds
    imu_sample: np.ndarray    # (M,) fractional IMU sample
    uv: np.ndarray            # (M, 4, 2) pixels, order config.POINT_NAMES


def _relabel(view, key, entry, fixes):
    """Apply a {true_name: clicked_name} fix to one frame entry, if any."""
    fix = fixes.get((view, int(key)))
    if not fix:
        return entry
    return {**entry, **{true: entry[clicked] for true, clicked in fix.items()}}


def load_anchors(path=config.ANCHORS_JSON, fixes=config.ANCHOR_RELABEL) -> dict[str, Anchors]:
    """Return {'front': Anchors, 'rear': Anchors}, frames sorted ascending.

    Known label mix-ups listed in `fixes` are corrected; pass fixes={} for raw.
    """
    raw = json.loads(path.read_text())
    out = {}
    for view, block in raw["views"].items():
        keys = sorted(block["frames"], key=int)
        fr = [_relabel(view, k, block["frames"][k], fixes) for k in keys]
        out[view] = Anchors(
            frames=np.array([int(k) for k in keys]),
            t_from_impact=np.array([f["t_from_impact_s"] for f in fr]),
            imu_sample=np.array([f["imu_sample"] for f in fr]),
            uv=np.array([[f[p] for p in config.POINT_NAMES] for f in fr], float),
        )
    return out


def read_frames(path, wanted=None):
    """Decode a video; return {frame_index: (bgr_image, pts_seconds)}.

    `wanted` limits which decode indices are kept (all if None). The pts time
    is used because the front video has a variable frame rate.
    """
    cap = cv2.VideoCapture(str(path))
    wanted = None if wanted is None else set(int(w) for w in wanted)
    frames, i = {}, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if wanted is None or i in wanted:
            frames[i] = (img, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        i += 1
        if wanted is not None and i > max(wanted):
            break
    cap.release()
    return frames


def frame_times(path):
    """pts time (s) of every decoded frame of a video."""
    cap = cv2.VideoCapture(str(path))
    ts = []
    while cap.grab():
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()
    return np.array(ts)
