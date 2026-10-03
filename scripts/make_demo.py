"""Build out/demo.mp4: front | rear | Blender render (or sensor overlay), 10x slow motion.

Each output frame is one instant on the IMU clock (30 fps x SLOWMO). The two
videos show their nearest frame; the third panel shows out/blender_stroke.mp4 if
it exists, otherwise the front video with the sensor-only racket drawn on it
(orientation interpolated to the exact sample). A top bar shows the sample
number, time from impact and the angle error of the displayed video moment.
Frames are piped to ffmpeg (imageio-ffmpeg binary) from Python.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

from racket import config, gyro, quat
from racket.io import load_anchors, read_frames

sys.path.insert(0, str(Path(__file__).parent))
from stage6_overlay import outline, to_pixels  # noqa: E402

SLOWMO, FPS = 10, 30
PANEL_W, PANEL_H, BAR_H = 640, 480, 70
RENDER = config.OUT_DIR / "blender_stroke.mp4"


def crop_box(uv, aspect=4 / 3, margin=90):
    """4:3 box around all clicked points of a view, kept inside the frame."""
    pts = uv.reshape(-1, 2)
    (x0, y0), (x1, y1) = pts.min(0) - margin, pts.max(0) + margin
    w = max(x1 - x0, (y1 - y0) * aspect)
    w = min(w, 1280, 720 * aspect)
    h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(np.clip(cx - w / 2, 0, 1280 - w))
    y0 = int(np.clip(cy - h / 2, 0, 720 - h))
    return x0, y0, int(w), int(h)


def panel(img, box, title):
    x0, y0, w, h = box
    p = cv2.resize(img[y0:y0 + h, x0:x0 + w], (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (PANEL_W, 30), (20, 20, 20), -1)
    cv2.putText(p, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def render_reader():
    """Frame getter for the Blender render, or None if it is not there."""
    if not RENDER.exists():
        return None
    cap = cv2.VideoCapture(str(RENDER))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames or None


def main():
    cam = json.loads((config.OUT_DIR / "stage3_cameras.json").read_text())
    f = np.array([cam["focal_px"]["front"], cam["focal_px"]["rear"]])
    camp = (f, np.array(cam["R_cam1_to_cam2"]), np.array(cam["t_cam1_to_cam2_m"]))
    model = np.array([cam["racket_points_sensor_frame_m"][n] for n in config.POINT_NAMES])
    shapes = outline(model)

    vid = pd.read_csv(config.OUT_DIR / "video_orientation.csv")
    q_imu = pd.read_csv(config.OUT_DIR / "orientation.csv")[["qw", "qx", "qy", "qz"]].to_numpy()
    ms = vid.imu_sample.to_numpy()
    t_m = vid[["tx", "ty", "tz"]].to_numpy()
    # honest number: leave-one-out error from stage 5 (the GP never saw that moment)
    err = np.array(json.loads((config.OUT_DIR / "stage5_gp.json").read_text())["per_moment"]["loo"])

    A = load_anchors()
    frames = {v: read_frames(config.VIDEOS[v], wanted=A[v].frames) for v in ("front", "rear")}
    boxes = {v: crop_box(A[v].uv) for v in ("front", "rear")}
    render = render_reader()
    third_title = "Blender render (gyro + GP)" if render else "Gyro + GP racket on front video"

    n_out = int((len(q_imu) - 1) / config.FS_HZ * FPS * SLOWMO) + 1
    W, H = 3 * PANEL_W, PANEL_H + BAR_H
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(config.OUT_DIR / "demo.mp4")]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for k in range(n_out):
        s = k * config.FS_HZ / (FPS * SLOWMO)                 # IMU sample at this instant
        t_imp = (s - config.IMPACT_SAMPLE) / config.FS_HZ
        m = int(np.clip(np.rint(t_imp * 30) + 14, 0, len(ms) - 1))   # displayed video moment
        imgs = {v: frames[v][A[v].frames[m]][0] for v in ("front", "rear")}
        p1 = panel(imgs["front"], boxes["front"], f"Front camera, frame {A['front'].frames[m]}")
        p2 = panel(imgs["rear"], boxes["rear"], f"Rear camera, frame {A['rear'].frames[m]}")
        if render:
            r = render[min(int(k * len(render) / n_out), len(render) - 1)]
            p3 = cv2.resize(r, (PANEL_W, PANEL_H))
            cv2.rectangle(p3, (0, 0), (PANEL_W, 30), (20, 20, 20), -1)
            cv2.putText(p3, third_title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            R = quat.to_matrix(gyro.sample_at(q_imu, s))
            t = np.array([np.interp(s, ms, t_m[:, i]) for i in range(3)])
            img = imgs["front"].copy()
            for pts, col, wd in zip(shapes, [(40, 200, 40), (40, 255, 40), (0, 0, 255)], [4, 2, 2]):
                px = to_pixels(pts @ R.T + t, "front", *camp).astype(np.int32)
                cv2.polylines(img, [px], False, col, wd, cv2.LINE_AA)
            p3 = panel(img, boxes["front"], third_title)
        bar = np.full((BAR_H, W, 3), (45, 32, 19), np.uint8)
        phase = "IMPACT" if abs(t_imp) < 0.004 else ("before impact" if t_imp < 0 else "after impact")
        cv2.putText(bar, f"sample {s:6.1f} / 399", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bar, f"t = {t_imp:+.3f} s  ({phase})", (480, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (106, 196, 233), 2, cv2.LINE_AA)
        colour = (80, 220, 80) if err[m] < 15 else ((60, 170, 255) if err[m] < 45 else (80, 80, 255))
        cv2.putText(bar, f"error {err[m]:5.1f} deg (leave-one-out)", (1000, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    colour, 2, cv2.LINE_AA)
        cv2.putText(bar, f"{SLOWMO}x slow", (1700, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 1, cv2.LINE_AA)
        ff.stdin.write(np.vstack([bar, np.hstack([p1, p2, p3])]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote out/demo.mp4: {n_out} frames, {n_out / FPS:.1f} s, third panel = {third_title}")


if __name__ == "__main__":
    main()
