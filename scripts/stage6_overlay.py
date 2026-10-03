"""Stage 6: draw the sensor-only racket onto both videos.

Orientation: out/orientation.csv (IMU only, start anchored on video moments 0-4).
Position: stage-3 racket translation per moment (the IMU gives orientation only).
Cameras: stage-3 focal lengths and relative pose.
Green = sensor racket, red dots = hand-clicked points, white text = angle to video pose.

Outputs out/overlay_front.mp4, out/overlay_rear.mp4, out/overlay_contact_sheet.png
"""
import json

import cv2
import numpy as np
import pandas as pd

from racket import cameras, config, gyro, quat
from racket.io import load_anchors, read_frames


def outline(model):
    """Handle segment + head ellipse + face-normal stub, sensor frame."""
    names = list(config.POINT_NAMES)
    butt, tip, edge = (model[names.index(n)] for n in ("butt", "tip", "edge"))
    cx, hl, hw = edge[0], edge[0] - tip[0], edge[1]
    th = np.linspace(0, 2 * np.pi, 48)
    head = np.c_[cx + hl * np.cos(th), hw * np.sin(th), 0 * th]
    handle = np.array([butt, [cx + hl, 0, 0]])
    normal = np.array([[cx, 0, 0], [cx, 0, 0.12]])
    return handle, head, normal


def to_pixels(P_cam1, view, f, R21, t21):
    P = P_cam1 if view == "front" else P_cam1 @ R21.T + t21
    return cameras.project(P, f[0 if view == "front" else 1])


def draw(img, R, t, view, cam, shapes, clicks, label):
    f, R21, t21 = cam
    for pts, colour, width in zip(shapes, [(40, 200, 40), (40, 255, 40), (0, 0, 255)], [4, 2, 2]):
        px = to_pixels(pts @ R.T + t, view, f, R21, t21).astype(np.int32)
        cv2.polylines(img, [px], False, colour, width, cv2.LINE_AA)
    for p in clicks:
        cv2.circle(img, tuple(int(v) for v in p), 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (1280, 34), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    cam_json = json.loads((config.OUT_DIR / "stage3_cameras.json").read_text())
    f = np.array([cam_json["focal_px"]["front"], cam_json["focal_px"]["rear"]])
    cam = (f, np.array(cam_json["R_cam1_to_cam2"]), np.array(cam_json["t_cam1_to_cam2_m"]))
    model = np.array([cam_json["racket_points_sensor_frame_m"][n] for n in config.POINT_NAMES])
    shapes = outline(model)

    vid = pd.read_csv(config.OUT_DIR / "video_orientation.csv")
    ori = pd.read_csv(config.OUT_DIR / "orientation.csv")
    q_imu = ori[["qw", "qx", "qy", "qz"]].to_numpy()
    s = vid.imu_sample.to_numpy()
    q_m = gyro.sample_at(q_imu, s)
    R_m = quat.to_matrix(q_m)
    t_m = vid[["tx", "ty", "tz"]].to_numpy()
    ang = np.rad2deg(quat.angle_between(q_m, vid[["qw", "qx", "qy", "qz"]].to_numpy()))
    anchors = load_anchors()

    sheet_rows = []
    pick = [0, 2, 4, 6, 8, 14, 20, 28]
    for view in ("front", "rear"):
        a = anchors[view]
        frames = read_frames(config.VIDEOS[view], wanted=a.frames)
        out = cv2.VideoWriter(str(config.OUT_DIR / f"overlay_{view}.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), 6, (1280, 720))
        tiles = []
        for m, fr in enumerate(a.frames):
            phase = "impact" if m == 14 else ("before" if m < 14 else "after")
            label = (f"{view} frame {fr}  t={a.t_from_impact[m]:+.3f}s ({phase})  "
                     f"sensor vs video pose: {ang[m]:.0f} deg"
                     + ("  [start anchor]" if m < config.ANCHOR_MOMENTS else ""))
            img = draw(frames[fr][0].copy(), R_m[m], t_m[m], view, cam, shapes, a.uv[m], label)
            out.write(img)
            if m in pick:
                cx, cy = a.uv[m].mean(0).astype(int)
                x0, y0 = np.clip([cx - 150, cy - 120], 0, [1280 - 300, 720 - 240])
                tile = img[y0:y0 + 240, x0:x0 + 300].copy()
                cv2.putText(tile, f"{view} t={a.t_from_impact[m]:+.2f}s {ang[m]:.0f}deg", (6, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
        out.release()
        sheet_rows.append(np.hstack(tiles))
    cv2.imwrite(str(config.OUT_DIR / "overlay_contact_sheet.png"), np.vstack(sheet_rows))
    print("angle sensor vs video per moment:", ang.round(0).astype(int).tolist())
    print("wrote out/overlay_front.mp4, out/overlay_rear.mp4, out/overlay_contact_sheet.png")


if __name__ == "__main__":
    main()
