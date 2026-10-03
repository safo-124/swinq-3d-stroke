"""Joint fit of the integrated gyro to the video orientations.

Model, for video moment i at nominal IMU sample s_i:
    q_video_i  ~=  q0 * Q(s_i + dt; bias, sign) * M
  q0    starting orientation (sensor -> camera 1) at IMU sample 0
  Q     gyro-integrated rotation from sample 0 (body frame increments)
  dt    time offset in samples (absorbs the +/- 1 frame video sync)
  M     small mounting correction, racket model frame -> sensor frame
Residual = rotation vector of (q_pred^-1 * q_video) in the racket frame, so its
norm is the angle between IMU and video orientation.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import config, gyro, quat


@dataclass
class JointResult:
    sign: int
    q0: np.ndarray            # (4,) sensor -> world at IMU sample 0
    bias_dps: np.ndarray      # (3,)
    mount: np.ndarray         # (4,) racket -> sensor
    dt_samples: float
    err_deg: np.ndarray       # (M,) angle IMU vs video per moment
    err_vec_deg: np.ndarray   # (M, 3) error rotation vector, racket frame
    weights: np.ndarray       # (M,)
    cost: float
    x: np.ndarray = field(repr=False)

    @property
    def dt_s(self):
        return self.dt_samples / config.FS_HZ

    @property
    def mount_deg(self):
        return float(np.rad2deg(np.linalg.norm(quat.to_rotvec(self.mount))))


# --- forward model -------------------------------------------------------------

def padded_gyro(g, pad=config.JOINT_PAD_SAMPLES):
    """Hold the first/last gyro sample for `pad` samples each side."""
    return np.concatenate([np.repeat(g[:1], pad, 0), g, np.repeat(g[-1:], pad, 0)])


def relative_rotation(g, bias_dps, sign, pad=config.JOINT_PAD_SAMPLES):
    """Q on the padded grid, normalised so Q = identity at real sample 0.

    Returns (N + 2 pad, 4); real sample k is at index k + pad.
    """
    q = gyro.integrate(padded_gyro(g, pad), bias_dps=bias_dps, sign=sign)
    return quat.mul(quat.conj(q[pad]), q)


def predict(g, samples, q0, bias_dps, sign, mount, dt, pad=config.JOINT_PAD_SAMPLES):
    """Predicted racket -> world orientation at (fractional) IMU samples."""
    Q = relative_rotation(g, bias_dps, sign, pad)
    Qs = gyro.sample_at(Q, np.asarray(samples) + dt + pad)
    return quat.mul(quat.mul(q0, Qs), mount)


def orientation_per_sample(g, res: JointResult):
    """Racket -> world orientation at every real IMU sample, (N, 4), continuous."""
    q = predict(g, np.arange(len(g)), res.q0, res.bias_dps, res.sign, res.mount, 0.0)
    return quat.make_continuous(q)


# --- fitting -------------------------------------------------------------------

def moment_weights(samples, lo=config.IMPACT_DOWNWEIGHT[0], hi=config.IMPACT_DOWNWEIGHT[1],
                   w=config.IMPACT_WEIGHT):
    """Weight w for moments inside the impact window [lo, hi], 1 elsewhere."""
    s = np.asarray(samples)
    return np.where((s >= lo) & (s <= hi), w, 1.0)


def _unpack(x, q0_ref):
    """x = [dq0 (3), bias (3, deg/s), mount rotvec (3), dt (samples)]."""
    return quat.mul(q0_ref, quat.from_rotvec(x[:3])), x[3:6], quat.from_rotvec(x[6:9]), x[9]


def error_vectors(x, g, samples, q_video, sign, q0_ref):
    """(M, 3) error rotation vectors (rad), racket frame: q_pred^-1 * q_video."""
    q0, bias, mount, dt = _unpack(x, q0_ref)
    q_pred = predict(g, samples, q0, bias, sign, mount, dt)
    return quat.to_rotvec(quat.mul(quat.conj(q_pred), q_video))


def init_q0(g, samples, q_video, sign, dt=0.0):
    """Closed-form q0: chordal mean of q_video_i * Q_i^-1 (bias 0, M = I)."""
    Q = gyro.sample_at(relative_rotation(g, np.zeros(3), sign), np.asarray(samples) + dt
                       + config.JOINT_PAD_SAMPLES)
    c = quat.mul(q_video, quat.conj(Q))
    c = c * np.sign(np.sum(c * c[0], axis=1, keepdims=True))         # same hemisphere
    w, v = np.linalg.eigh(c.T @ c)
    return quat.normalize(v[:, -1])


def fit_sign(g, samples, q_video, sign, dt_starts=config.JOINT_DT_STARTS,
             f_scale_deg=config.JOINT_F_SCALE_DEG, max_dt=config.JOINT_MAX_DT_S * config.FS_HZ):
    """Best joint fit for one rotation sign over several time-offset starts."""
    wts = moment_weights(samples)
    best = None
    for dt0 in dt_starts:
        q0_ref = init_q0(g, samples, q_video, sign, dt0)

        def res(x):
            return (error_vectors(x, g, samples, q_video, sign, q0_ref) * wts[:, None]).ravel()

        x0 = np.r_[np.zeros(9), dt0]
        lb = np.r_[np.full(9, -np.inf), -max_dt]
        ub = np.r_[np.full(9, np.inf), max_dt]
        sol = least_squares(res, x0, bounds=(lb, ub), loss="soft_l1",
                            f_scale=np.deg2rad(f_scale_deg), x_scale="jac")
        if best is None or sol.cost < best[0].cost:
            best = (sol, q0_ref)
    sol, q0_ref = best
    q0, bias, mount, dt = _unpack(sol.x, q0_ref)
    ev = np.rad2deg(error_vectors(sol.x, g, samples, q_video, sign, q0_ref))
    return JointResult(sign=sign, q0=q0, bias_dps=bias, mount=mount, dt_samples=float(dt),
                       err_deg=np.linalg.norm(ev, axis=1), err_vec_deg=ev, weights=wts,
                       cost=float(sol.cost), x=sol.x)


def fit(g, samples, q_video):
    """Fit both rotation signs; return (best, {sign: result})."""
    results = {s: fit_sign(g, samples, q_video, s) for s in (1, -1)}
    return min(results.values(), key=lambda r: r.cost), results
