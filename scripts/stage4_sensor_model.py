"""Stage 4 (revised): sensor-only stroke model, motion parameters, 3D animation.

Outputs:
  out/orientation_sensor_only.csv  per sample: sensor-only orientation (camera-1 and Blender z-up)
  out/motion_params.json   key stroke numbers from the IMU
  out/stage4_summary.png   rates, speeds, reconstructed ax, video agreement
  out/stroke_3d.mp4        3D animation of the racket from the sensor data
"""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from racket import config, gyro, motion, quat
from racket.io import load_imu

def racket_outline(model):
    """Polyline points (sensor frame): handle + elliptical head through the edge point."""
    names = list(config.POINT_NAMES)
    butt, tip, edge = (model[names.index(n)] for n in ("butt", "tip", "edge"))
    centre_x, half_len, half_w = edge[0], edge[0] - tip[0], edge[1]
    th = np.linspace(0, 2 * np.pi, 60)
    head = np.c_[centre_x + half_len * np.cos(th), half_w * np.sin(th), 0 * th]
    handle = np.array([butt, [centre_x + half_len, 0, 0]])
    return handle, head


def save_orientation_csv(q_cam, imu, path):
    """Per-sample orientation, camera-1 frame and Blender frame."""
    q_bl = motion.to_blender(q_cam)
    df = pd.DataFrame({"sample": np.arange(imu.n), "t_s": imu.t,
                       "qw": q_cam[:, 0], "qx": q_cam[:, 1], "qy": q_cam[:, 2], "qz": q_cam[:, 3],
                       "bl_qw": q_bl[:, 0], "bl_qx": q_bl[:, 1], "bl_qy": q_bl[:, 2], "bl_qz": q_bl[:, 3]})
    df.to_csv(path, index=False, float_format="%.6f")


def plot_summary(imu, ax_filled, radius, q, video, params, path):
    s = np.arange(imu.n)
    w = np.linalg.norm(imu.gyro, axis=1)
    w_perp = np.deg2rad(np.hypot(imu.gyro[:, 1], imu.gyro[:, 2]))
    head_off = -config.RACKET_POINTS["edge"][0]
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(s, w, label="|ω| total")
    axes[0].plot(s, np.abs(imu.gyro[:, 0]), lw=.8, label="|roll rate| (about handle)")
    axes[0].set_ylabel("deg/s")
    axes[0].set_title(f"Angular velocity, peak {params['peak_angular_velocity_dps']:.0f} deg/s")
    axes[1].plot(s, w_perp * (radius + head_off), label="head centre")
    axes[1].plot(s, w_perp * radius, label="sensor")
    axes[1].set_ylabel("m/s")
    axes[1].set_title(f"Estimated speed (ω⊥ × radius), head centre at impact "
                      f"{params['speed_at_head_centre_ms']:.1f} m/s")
    axes[2].plot(s, imu.accel[:, 0], label="ax measured (clips at 16 g)")
    j = np.arange(config.ACCEL_CLIPPED[0], config.ACCEL_CLIPPED[1] + 1)
    axes[2].plot(j, ax_filled[j], "r-", lw=2, label="ax reconstructed = r·ω⊥² + c (fit on samples 170-181)")
    axes[2].set_ylabel("g")
    axes[2].set_title("Saturation repair")
    vs, qv = video
    err = np.rad2deg(quat.angle_between(gyro.sample_at(q, vs), qv))
    axes[3].plot(vs, err, "o-")
    axes[3].axvspan(vs[0], vs[config.ANCHOR_MOMENTS - 1], color="green", alpha=.15,
                    label="moments used for start orientation")
    axes[3].set_ylabel("deg")
    axes[3].set_title("Sensor-only orientation vs video (stage 3) per moment")
    axes[3].set_xlabel("IMU sample (416 Hz)")
    for ax in axes:
        ax.axvline(config.IMPACT_SAMPLE, color="k", ls=":")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return err


def render_animation(q_bl, model, path, step=2, fps=30):
    """Matplotlib 3D racket animation written with OpenCV (no ffmpeg needed)."""
    handle, head = racket_outline(model)
    R = quat.to_matrix(q_bl)
    tip_trail = np.einsum("nij,j->ni", R, head[30])          # far end of the head
    fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
    writer = None
    for k in range(0, len(R), step):
        fig.clf()
        ax = fig.add_subplot(projection="3d")
        h, d = handle @ R[k].T, head @ R[k].T
        ax.plot(*h.T, color="saddlebrown", lw=5)
        ax.plot(*d.T, color="tab:blue", lw=3)
        ax.plot(*tip_trail[:k + 1].T, color="grey", lw=1)
        normal = R[k] @ np.array([0, 0, 0.15])
        c = d.mean(0)
        ax.quiver(*c, *normal, color="red")
        lim = 0.6
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), xlabel="x", ylabel="y (depth)",
               zlabel="z (up)")
        ax.view_init(elev=20, azim=-60)
        phase = "impact" if abs(k - config.IMPACT_SAMPLE) < 3 else ("follow-through" if k > 200 else "forward swing")
        ax.set_title(f"sample {k}  t={k / config.FS_HZ - 200 / config.FS_HZ:+.3f}s  {phase}\n"
                     f"red = string-face normal, grey = head path")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1]
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (img.shape[1], img.shape[0]))
        writer.write(np.ascontiguousarray(img))
    writer.release()
    plt.close(fig)


def main():
    imu = load_imu()
    v = pd.read_csv(config.OUT_DIR / "video_orientation.csv")
    vs, qv = v.imu_sample.to_numpy(), v[["qw", "qx", "qy", "qz"]].to_numpy()
    cam = json.loads((config.OUT_DIR / "stage3_cameras.json").read_text())
    model = np.array([cam["racket_points_sensor_frame_m"][n] for n in config.POINT_NAMES])

    q0, anchor_err = motion.anchor_start(imu.gyro, vs, qv)
    q = motion.orientation(imu.gyro, q0)
    radius, radius_corr = motion.pivot_radius(imu.accel, imu.gyro)
    ax_filled, r_local, c_local, corr = motion.reconstruct_clipped_ax(imu.accel, imu.gyro)
    f_vib, aliases = motion.string_vibration_hz(imu.accel)
    params = motion.motion_parameters(imu, q, radius)
    params.pop("point_names")
    params.update({"gyro_sign": config.GYRO_SIGN, "start_orientation_q_cam1": q0.tolist(),
                   "anchor_error_deg_moments_0_4": anchor_err.tolist(),
                   "pivot_radius_corr": radius_corr, "ax_peak_reconstructed_g": float(ax_filled.max()),
                   "ax_fill_local_r_m": r_local, "ax_fill_offset_g": c_local, "ax_fill_corr": corr,
                   "string_vibration_sampled_hz": f_vib, "string_vibration_aliases_hz": aliases})

    save_orientation_csv(q, imu, config.OUT_DIR / "orientation_sensor_only.csv")
    err = plot_summary(imu, ax_filled, radius, q, (vs, qv), params, config.OUT_DIR / "stage4_summary.png")
    params["video_agreement_deg_per_moment"] = err.round(1).tolist()
    (config.OUT_DIR / "motion_params.json").write_text(json.dumps(params, indent=2))
    render_animation(motion.to_blender(q), model, config.OUT_DIR / "stroke_3d.mp4")

    print(f"start orientation from moments 0-4: error {np.round(anchor_err, 1)} deg")
    for k in ("peak_angular_velocity_dps", "swing_rate_at_impact_dps", "roll_rate_at_impact_dps",
              "pivot_radius_m", "speed_at_sensor_ms", "speed_at_head_centre_ms", "speed_at_tip_ms",
              "rotation_last_100ms_before_impact_deg", "total_rotation_path_deg", "impact_gyro_drop_dps",
              "pivot_radius_corr", "ax_peak_reconstructed_g", "ax_fill_local_r_m", "ax_fill_offset_g",
              "ax_fill_corr", "string_vibration_sampled_hz"):
        print(f"  {k:42s} {params[k]:.2f}")
    print(f"  string vibration aliases (Hz): {np.round(aliases, 0)}")
    print(f"video agreement per moment (deg): {err.round(0).astype(int).tolist()}")
    print("wrote out/orientation_sensor_only.csv, motion_params.json, stage4_summary.png, stroke_3d.mp4")


if __name__ == "__main__":
    main()
