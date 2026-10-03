"""Stage 5: GP drift correction of the sensor orientation, with leave-one-out check.

Reads  out/orientation_sensor_only.csv (stage 4) and out/video_orientation.csv (stage 3).
Writes out/orientation.csv   sample, t_s, qw, qx, qy, qz, gp_std_deg, bl_q* (Blender z-up)
       out/stage5_gp.png     residual components with GP mean +/- 2 sigma; error before/after
       out/stage5_gp.json    kernels and the error numbers
       out/stroke_3d_gp.mp4  3D animation of the corrected orientation
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from racket import config, gp, gyro, motion, quat

sys.path.insert(0, str(Path(__file__).parent))
from stage4_sensor_model import render_animation  # noqa: E402


def plot(t_m, r, model, err_before, err_fit, err_loo, path):
    t = np.arange(400) / config.FS_HZ
    mean, std = model.predict(t)
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for k, lab in enumerate(["X (handle roll)", "Y", "Z (face normal)"]):
        ax = axes[k]
        ax.fill_between(t, np.rad2deg(mean[:, k] - 2 * std[:, k]), np.rad2deg(mean[:, k] + 2 * std[:, k]),
                        alpha=.25, label="GP ±2σ")
        ax.plot(t, np.rad2deg(mean[:, k]), label="GP mean")
        ax.plot(t_m, np.rad2deg(r[:, k]), "ko", ms=4, label="error at video moment")
        ax.set_ylabel("deg")
        ax.set_title(f"Error rotation vector, component {lab}")
    axes[3].plot(t_m, err_before, "o-", label=f"sensor only (mean {err_before.mean():.0f}°)")
    axes[3].plot(t_m, err_loo, "s-", label=f"GP, leave-one-out (mean {err_loo.mean():.1f}°)")
    axes[3].plot(t_m, err_fit, ".-", lw=.8, label=f"GP, all moments (mean {err_fit.mean():.1f}°)")
    axes[3].set_yscale("log")
    axes[3].set_ylabel("angle to video (deg)")
    axes[3].set_title("Orientation error per video moment")
    axes[3].set_xlabel("time from sample 0 (s)")
    for ax in axes:
        ax.axvline(config.IMPACT_SAMPLE / config.FS_HZ, color="k", ls=":")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    q_raw = pd.read_csv(config.OUT_DIR / "orientation_sensor_only.csv")[["qw", "qx", "qy", "qz"]].to_numpy()
    v = pd.read_csv(config.OUT_DIR / "video_orientation.csv")
    s, qv = v.imu_sample.to_numpy(), v[["qw", "qx", "qy", "qz"]].to_numpy()
    t_m = s / config.FS_HZ

    r = gp.residuals(q_raw, s, qv)
    model = gp.fit(t_m, r, gp.sample_noise(s))
    q_corr, std_deg = gp.correct(q_raw, model)
    err_before = np.rad2deg(quat.angle_between(gyro.sample_at(q_raw, s), qv))
    err_fit = np.rad2deg(quat.angle_between(gyro.sample_at(q_corr, s), qv))
    err_loo = gp.leave_one_out(q_raw, s, qv)

    q_bl = motion.to_blender(q_corr)
    pd.DataFrame({"sample": np.arange(len(q_corr)), "t_s": np.arange(len(q_corr)) / config.FS_HZ,
                  "qw": q_corr[:, 0], "qx": q_corr[:, 1], "qy": q_corr[:, 2], "qz": q_corr[:, 3],
                  "gp_std_deg": std_deg,
                  "bl_qw": q_bl[:, 0], "bl_qx": q_bl[:, 1], "bl_qy": q_bl[:, 2], "bl_qz": q_bl[:, 3]}
                 ).to_csv(config.OUT_DIR / "orientation.csv", index=False, float_format="%.6f")

    summary = {"kernels": [str(g.kernel_) for g in model.gps],
               "error_before_deg": {"mean": err_before.mean(), "median": float(np.median(err_before)),
                                    "max": err_before.max()},
               "error_gp_all_moments_deg": {"mean": err_fit.mean(), "median": float(np.median(err_fit)),
                                            "max": err_fit.max()},
               "error_gp_leave_one_out_deg": {"mean": err_loo.mean(), "median": float(np.median(err_loo)),
                                              "max": err_loo.max()},
               "per_moment": {"before": err_before.round(1).tolist(), "loo": err_loo.round(1).tolist()},
               "gp_std_deg": {"median": float(np.median(std_deg)), "max": float(std_deg.max())}}
    (config.OUT_DIR / "stage5_gp.json").write_text(json.dumps(summary, indent=2, default=float))
    plot(t_m, r, model, err_before, err_fit, err_loo, config.OUT_DIR / "stage5_gp.png")
    cam = json.loads((config.OUT_DIR / "stage3_cameras.json").read_text())
    model_pts = np.array([cam["racket_points_sensor_frame_m"][n] for n in config.POINT_NAMES])
    render_animation(q_bl, model_pts, config.OUT_DIR / "stroke_3d_gp.mp4")

    print("kernels:", *summary["kernels"], sep="\n  ")
    for name, e in [("sensor only", err_before), ("GP, all moments", err_fit), ("GP, leave-one-out", err_loo)]:
        print(f"{name:20s} mean {e.mean():6.1f}  median {np.median(e):6.1f}  max {e.max():6.1f} deg")
    print("per moment (before -> leave-one-out):")
    for i in range(len(s)):
        print(f"  m{i:2d} sample {s[i]:6.1f}: {err_before[i]:6.1f} -> {err_loo[i]:5.1f}")
    print(f"GP 1-sigma per sample: median {np.median(std_deg):.1f} deg, max {std_deg.max():.1f} deg")
    print("wrote out/orientation.csv, stage5_gp.png, stage5_gp.json, stroke_3d_gp.mp4")


if __name__ == "__main__":
    main()
