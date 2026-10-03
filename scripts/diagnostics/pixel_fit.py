"""Diagnostic: fit the IMU orientation model directly to the clicked pixels.

Cameras and racket geometry are fixed from stage 3; each moment gets a free
translation. Orientation comes only from q0, bias, mounting, time offset and
sign, so no per-frame (mirror-ambiguous) video orientation is involved.

Usage: python scripts/diagnostics/pixel_fit.py [imu_rate_hz]
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from racket import cameras, config, joint, quat
from racket.io import load_anchors, load_imu


def load_stage3():
    cam = json.loads((config.OUT_DIR / "stage3_cameras.json").read_text())
    f = np.array([cam["focal_px"]["front"], cam["focal_px"]["rear"]])
    model = np.array([cam["racket_points_sensor_frame_m"][n] for n in config.POINT_NAMES])
    return f, np.array(cam["R_cam1_to_cam2"]), np.array(cam["t_cam1_to_cam2_m"]), model


def main(fs=416.0):
    f, R21, t21, model = load_stage3()
    A = load_anchors()
    obs = np.stack([A["front"].uv, A["rear"].uv])
    v = pd.read_csv(config.OUT_DIR / "video_orientation.csv")
    t_ba = v[["tx", "ty", "tz"]].to_numpy()
    g = load_imu().gyro
    k = 416.0 / fs                                   # gyro*k at 416 Hz == gyro at fs
    s = 200 + v.t_from_impact_s.to_numpy() * fs
    use = (s >= -20) & (s <= 419)
    w = joint.moment_weights(s)[use]
    n = int(use.sum())
    lb = np.r_[np.full(9, -np.inf), -21, np.full(3 * n, -np.inf)]
    ub = np.r_[np.full(9, np.inf), 21, np.full(3 * n, np.inf)]

    def residual(x, sign):
        q = joint.predict(g * k, s[use], quat.from_rotvec(x[:3]), x[3:6], sign,
                          quat.from_rotvec(x[6:9]), x[9])
        Pc1 = np.einsum("mij,pj->mpi", quat.to_matrix(q), model) + x[10:].reshape(-1, 1, 3)
        pred = np.stack([cameras.project(Pc1, f[0]), cameras.project(Pc1 @ R21.T + t21, f[1])])
        return ((pred - obs[:, use]) * w[None, :, None, None]).ravel()

    # residual rows: (view, moment, point, xy); translation of moment i only
    # touches its own 16 rows, so its 3 columns can share finite differences
    S = lil_matrix((2 * n * 8, 10 + 3 * n), dtype=int)
    S[:, :10] = 1
    for v in range(2):
        for i in range(n):
            S[(v * n + i) * 8:(v * n + i) * 8 + 8, 10 + 3 * i:13 + 3 * i] = 1
    S = S.tocsr()

    def jac(x, sign):
        return approx_derivative(residual, x, args=(sign,), sparsity=S, bounds=(lb, ub)).toarray()

    def run(x0, sign, nfev):
        return least_squares(residual, x0, jac=jac, args=(sign,), loss="soft_l1", f_scale=3.0,
                             bounds=(lb, ub), tr_solver="exact", x_scale="jac", max_nfev=nfev)

    for sign in (1, -1):
        trials = [run(np.r_[r0, np.zeros(6), 0.0, t_ba[use].ravel()], sign, 30)
                  for r0 in Rotation.create_group("O").as_rotvec()]
        best = min(trials, key=lambda t: t.cost)
        sol = min((run(np.r_[best.x[:9], dt0, best.x[10:]], sign, 500) for dt0 in (-10.0, 0.0, 10.0)),
                  key=lambda t: t.cost)
        e = np.linalg.norm(sol.fun.reshape(2, n, 4, 2), axis=-1)
        print(f"fs {fs:.0f} sign {sign:+d}: cost {sol.cost:.0f}  pixel err median {np.median(e):.1f} "
              f"RMS {np.sqrt(np.mean(e ** 2)):.1f}  bias {(sol.x[3:6] / k).round(1)}  "
              f"mount {np.rad2deg(np.linalg.norm(sol.x[6:9])):.1f} deg  dt {sol.x[9]:.1f} samples")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 416.0)
