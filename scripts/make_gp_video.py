"""Build out/gp_correction.mp4: the GP drift correction made visible, 10x slow motion.

Panels: front | rear | error chart. On both videos the sensor-only racket (orange)
and the gyro + GP racket (green) are drawn together; the chart shows the error per
video moment before and after the correction (leave-one-out) with a time cursor.
Frames are piped to ffmpeg (imageio-ffmpeg binary) from Python.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from racket import config, gyro, quat
from racket.io import load_anchors, read_frames

sys.path.insert(0, str(Path(__file__).parent))
from make_demo import BAR_H, FPS, PANEL_H, PANEL_W, SLOWMO, crop_box, panel  # noqa: E402
from stage6_overlay import outline, to_pixels  # noqa: E402

ORANGE, GREEN = (0, 140, 255), (60, 220, 60)        # BGR


def load_q(name):
    return pd.read_csv(config.OUT_DIR / name)[["qw", "qx", "qy", "qz"]].to_numpy()


def error_chart(t_m, before, loo):
    """Chart image (PANEL_H x PANEL_W) and a function time -> cursor x pixel."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(PANEL_W / dpi, PANEL_H / dpi), dpi=dpi)
    ax.plot(t_m, before, "o-", color="#FF8C00", ms=4, label="sensor only")
    ax.plot(t_m, loo, "s-", color="#2EB82E", ms=4, label="gyro + GP (leave-one-out)")
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(1, 300)
    ax.set_xlim(t_m[0] - 0.02, t_m[-1] + 0.02)
    ax.set_xlabel("time from impact (s)")
    ax.set_ylabel("angle to video racket (deg)")
    ax.set_title("Orientation error per video frame", fontsize=11)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()
    x0, y0 = ax.transData.transform((t_m[0], 1))
    x1, _ = ax.transData.transform((t_m[-1], 1))
    top = fig.bbox.height - ax.transData.transform((0, 300))[1]
    bottom = fig.bbox.height - y0
    plt.close(fig)

    def cursor_x(t):
        return int(round(x0 + (t - t_m[0]) / (t_m[-1] - t_m[0]) * (x1 - x0)))
    return cv2.resize(img, (PANEL_W, PANEL_H)), cursor_x, int(top), int(bottom)


def draw_racket(img, R, t, view, camp, shapes, colour, width):
    for pts in shapes[:2]:                                    # handle + head outline
        px = to_pixels(pts @ R.T + t, view, *camp).astype(np.int32)
        cv2.polylines(img, [px], False, colour, width, cv2.LINE_AA)


def main():
    cam = json.loads((config.OUT_DIR / "stage3_cameras.json").read_text())
    f = np.array([cam["focal_px"]["front"], cam["focal_px"]["rear"]])
    camp = (f, np.array(cam["R_cam1_to_cam2"]), np.array(cam["t_cam1_to_cam2_m"]))
    model = np.array([cam["racket_points_sensor_frame_m"][n] for n in config.POINT_NAMES])
    shapes = outline(model)

    q_raw, q_gp = load_q("orientation_sensor_only.csv"), load_q("orientation.csv")
    vid = pd.read_csv(config.OUT_DIR / "video_orientation.csv")
    ms, t_m = vid.imu_sample.to_numpy(), vid[["tx", "ty", "tz"]].to_numpy()
    gpj = json.loads((config.OUT_DIR / "stage5_gp.json").read_text())
    before, loo = np.array(gpj["per_moment"]["before"]), np.array(gpj["per_moment"]["loo"])
    t_imp_m = (ms - config.IMPACT_SAMPLE) / config.FS_HZ
    chart, cursor_x, ctop, cbot = error_chart(t_imp_m, before, loo)

    A = load_anchors()
    frames = {v: read_frames(config.VIDEOS[v], wanted=A[v].frames) for v in ("front", "rear")}
    boxes = {v: crop_box(A[v].uv) for v in ("front", "rear")}

    n_out = int((len(q_raw) - 1) / config.FS_HZ * FPS * SLOWMO) + 1
    W, H = 3 * PANEL_W, PANEL_H + BAR_H
    out_path = config.OUT_DIR / "gp_correction.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_path)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for k in range(n_out):
        s = k * config.FS_HZ / (FPS * SLOWMO)
        t_imp = (s - config.IMPACT_SAMPLE) / config.FS_HZ
        m = int(np.clip(np.rint(t_imp * 30) + 14, 0, len(ms) - 1))
        t = np.array([np.interp(s, ms, t_m[:, i]) for i in range(3)])
        R_raw = quat.to_matrix(gyro.sample_at(q_raw, s))
        R_gp = quat.to_matrix(gyro.sample_at(q_gp, s))
        tiles = []
        for v, name in (("front", "Front"), ("rear", "Rear")):
            img = frames[v][A[v].frames[m]][0].copy()
            draw_racket(img, R_raw, t, v, camp, shapes, ORANGE, 3)
            draw_racket(img, R_gp, t, v, camp, shapes, GREEN, 3)
            p = panel(img, boxes[v], f"{name} camera   orange = sensor only   green = gyro + GP")
            tiles.append(p)
        c = chart.copy()
        x = cursor_x(t_imp)
        cv2.line(c, (x, ctop), (x, cbot), (200, 60, 60), 2, cv2.LINE_AA)
        tiles.append(c)

        bar = np.full((BAR_H, W, 3), (45, 32, 19), np.uint8)
        phase = "IMPACT" if abs(t_imp) < 0.004 else ("before impact" if t_imp < 0 else "after impact")
        cv2.putText(bar, f"sample {s:5.1f}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bar, f"t = {t_imp:+.3f} s ({phase})", (290, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    (106, 196, 233), 2, cv2.LINE_AA)
        cv2.putText(bar, f"sensor only {before[m]:5.1f} deg", (850, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    ORANGE, 2, cv2.LINE_AA)
        cv2.putText(bar, f"with GP {loo[m]:5.1f} deg", (1280, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    GREEN, 2, cv2.LINE_AA)
        cv2.putText(bar, f"{SLOWMO}x slow", (1760, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        ff.stdin.write(np.vstack([bar, np.hstack(tiles)]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out_path.name}: {n_out} frames, {n_out / FPS:.1f} s")


if __name__ == "__main__":
    main()
