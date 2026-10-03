"""Stage 3: two-camera bundle adjustment with the racket as calibration object.

Usage: python scripts/stage3_bundle.py [--fixed-geometry]
"""
import argparse
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from racket import cameras, config, quat
from racket.io import load_anchors, read_frames


def save_orientation_csv(res, anchors, path):
    """One row per moment: racket -> camera-1 rotation and translation + diagnostics."""
    f, r = anchors["front"], anchors["rear"]
    q = quat.make_continuous(quat.from_matrix(res.R))
    df = pd.DataFrame({
        "moment": np.arange(len(q)), "front_frame": f.frames, "rear_frame": r.frames,
        "t_from_impact_s": f.t_from_impact, "imu_sample": f.imu_sample,
        "qw": q[:, 0], "qx": q[:, 1], "qy": q[:, 2], "qz": q[:, 3],
        "tx": res.t[:, 0], "ty": res.t[:, 1], "tz": res.t[:, 2],
        "rms_front_px": res.rms_px[:, 0], "rms_rear_px": res.rms_px[:, 1],
        "sigma_roll_x_deg": res.rot_sigma_deg[:, 0], "sigma_y_deg": res.rot_sigma_deg[:, 1],
        "sigma_z_deg": res.rot_sigma_deg[:, 2],
    })
    df.to_csv(path, index=False, float_format="%.6f")
    return df


def save_cameras_json(res, starts, path):
    """Camera intrinsics/extrinsics, refined geometry and the multistart table."""
    C2 = -res.R21.T @ res.t21
    out = {
        "world_frame": "camera 1 (front)",
        "focal_px": {"front": res.f[0], "rear": res.f[1]},
        "principal_point_px": cameras.image_centre().tolist(),
        "R_cam1_to_cam2": res.R21.tolist(), "t_cam1_to_cam2_m": res.t21.tolist(),
        "cam2_centre_in_cam1_m": C2.tolist(),
        "cam2_relative_angle_deg": float(np.rad2deg(quat.angle_between(quat.IDENTITY, quat.from_matrix(res.R21)))),
        "racket_points_sensor_frame_m": dict(zip(config.POINT_NAMES, res.model.tolist())),
        "cost": res.cost,
        "multistart": [{"f_init": f0, "cost": r.cost, "f": r.f.tolist()} for f0, r in starts],
    }
    path.write_text(json.dumps(out, indent=2, default=float))


def plot_errors(res, anchors, path):
    """Per-moment reprojection RMS (both views) and orientation sigma."""
    s = anchors["front"].imu_sample
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    w = 4.0
    axes[0].bar(s - w / 2, res.rms_px[:, 0], w, label="front")
    axes[0].bar(s + w / 2, res.rms_px[:, 1], w, label="rear")
    axes[0].set_ylabel("RMS reprojection (px)")
    axes[0].legend()
    for k, lab in enumerate(["about X (handle roll)", "about Y", "about Z (face normal)"]):
        axes[1].plot(s, res.rot_sigma_deg[:, k], "o-", ms=3, label=lab)
    axes[1].set_ylabel("orientation 1σ (deg)\n(conditional on cameras)")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("IMU sample")
    for ax in axes:
        ax.axvline(config.IMPACT_SAMPLE, color="k", ls=":")
        ax.grid(alpha=.3)
    axes[0].set_title("Bundle adjustment residuals per moment")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_scene(res, path):
    """Top-down (x-z of camera 1) view: both cameras and the handle at each moment."""
    fig, ax = plt.subplots(figsize=(7, 7))
    pts = np.einsum("mij,pj->mpi", res.R, res.model) + res.t[:, None, :]
    for i, p in enumerate(pts):
        c = plt.cm.viridis(i / (len(pts) - 1))
        ax.plot(p[[0, 2], 0], p[[0, 2], 2], color=c, lw=1.5)        # butt -> tip
        ax.plot(*p[0, [0, 2]], "o", color=c, ms=3)
    C2 = -res.R21.T @ res.t21
    for C, R, name in [(np.zeros(3), np.eye(3), "front cam"), (C2, res.R21.T, "rear cam")]:
        d = R @ np.array([0, 0, 1.0])
        ax.plot(C[0], C[2], "ks")
        ax.arrow(C[0], C[2], d[0], d[2], head_width=0.2, color="k")
        ax.text(C[0], C[2], f"  {name}")
    ax.set_xlabel("x (m, camera 1)")
    ax.set_ylabel("z (m, camera 1 depth)")
    ax.set_aspect("equal")
    ax.grid(alpha=.3)
    ax.set_title("Top-down scene (handle butt→tip, colour = time)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def reprojection_sheet(res, anchors, moments, path):
    """Crops showing clicked points (circles) and reprojected model (lines + crosses)."""
    prob = cameras.Problem(np.stack([anchors["front"].uv, anchors["rear"].uv]),
                           res.model, refine_geometry=False)
    x = prob.pack(res.f, cv2.Rodrigues(res.R21)[0].ravel(), res.t21, res.model,
                  np.array([cv2.Rodrigues(R)[0].ravel() for R in res.R]), res.t)
    pred = prob.predict(x)
    rows = []
    for v, view in enumerate(("front", "rear")):
        a = anchors[view]
        frames = read_frames(config.VIDEOS[view], wanted=a.frames[moments])
        tiles = []
        for m in moments:
            img = frames[a.frames[m]][0].copy()
            P = pred[v, m]
            for i, j in [(0, 1), (1, 2), (1, 3)]:
                cv2.line(img, tuple(P[i].astype(int)), tuple(P[j].astype(int)), (0, 255, 255), 1)
            for k in range(4):
                cv2.circle(img, tuple(a.uv[m, k].astype(int)), 5, (0, 0, 255), 1)
                cv2.drawMarker(img, tuple(P[k].astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
            cx, cy = a.uv[m].mean(0).astype(int)
            x0, y0 = np.clip([cx - 160, cy - 130], 0, [1280 - 320, 720 - 260])
            tile = img[y0:y0 + 260, x0:x0 + 320].copy()
            cv2.putText(tile, f"{view} m{m} {res.rms_px[m, v]:.1f}px", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-geometry", action="store_true")
    args = ap.parse_args()
    config.OUT_DIR.mkdir(exist_ok=True)
    anchors = load_anchors()
    obs = np.stack([anchors["front"].uv, anchors["rear"].uv])

    res, starts = cameras.solve_multistart(obs, refine_geometry=not args.fixed_geometry)
    print("multistart:")
    for f0, r in starts:
        print(f"  f_init {f0:5.0f}: cost {r.cost:8.1f}  f = ({r.f[0]:.0f}, {r.f[1]:.0f})")
    ang = np.rad2deg(quat.angle_between(quat.IDENTITY, quat.from_matrix(res.R21)))
    C2 = -res.R21.T @ res.t21
    print(f"moments repaired from the wrong mirror pose: {res.flipped}")
    print(f"focal lengths: front {res.f[0]:.0f} px, rear {res.f[1]:.0f} px")
    print(f"camera 2: rotated {ang:.0f} deg from camera 1, centre at {C2.round(2)} m "
          f"(baseline {np.linalg.norm(C2):.2f} m)")
    print(f"racket depth in camera 1: {res.t[:, 2].min():.2f}..{res.t[:, 2].max():.2f} m")
    g = dict(zip(config.POINT_NAMES, res.model.round(3).tolist()))
    print(f"racket points (sensor frame, m): {g}")
    e = res.err_px
    print(f"reprojection error, all points: median {np.median(e):.2f} px, "
          f"RMS front {np.sqrt(np.mean(e[0]**2)):.2f}, rear {np.sqrt(np.mean(e[1]**2)):.2f} px")
    print("median error per point (front / rear):",
          {n: (round(float(np.median(e[0, :, k])), 1), round(float(np.median(e[1, :, k])), 1))
           for k, n in enumerate(config.POINT_NAMES)})
    med = np.median(res.rot_sigma_deg, axis=0)
    print(f"orientation 1-sigma (conditional on cameras), median about X/Y/Z: "
          f"{med[0]:.2f} / {med[1]:.2f} / {med[2]:.2f} deg")
    print("\nper moment:  m  front  rear   sample   rms_F  rms_R   sigma X / Y / Z (deg)")
    a = anchors["front"]
    for m in range(len(a.frames)):
        print(f"           {m:2d}  {a.frames[m]:4d}  {anchors['rear'].frames[m]:4d}  {a.imu_sample[m]:6.1f}"
              f"  {res.rms_px[m, 0]:6.2f} {res.rms_px[m, 1]:6.2f}   "
              + " / ".join(f"{v:5.1f}" for v in res.rot_sigma_deg[m]))

    tag = "_fixedgeom" if args.fixed_geometry else ""
    save_orientation_csv(res, anchors, config.OUT_DIR / f"video_orientation{tag}.csv")
    save_cameras_json(res, starts, config.OUT_DIR / f"stage3_cameras{tag}.json")
    plot_errors(res, anchors, config.OUT_DIR / f"stage3_errors{tag}.png")
    plot_scene(res, config.OUT_DIR / f"stage3_scene{tag}.png")
    worst = np.argsort(res.rms_px.max(1))[::-1][:3]
    typical = np.argsort(res.rms_px.max(1))[len(res.rms_px) // 2 - 1:len(res.rms_px) // 2 + 2]
    reprojection_sheet(res, anchors, sorted(set(worst) | set(typical)),
                       config.OUT_DIR / f"stage3_reprojection{tag}.png")
    print(f"\nwrote out/video_orientation{tag}.csv, stage3_cameras{tag}.json, "
          f"stage3_errors{tag}.png, stage3_scene{tag}.png, stage3_reprojection{tag}.png")


if __name__ == "__main__":
    main()
