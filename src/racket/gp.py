"""Gaussian-process drift correction of the gyro orientation (stage 5).

At each video moment the leftover error E_i = q_imu(s_i)^-1 * q_video_i is taken
as a rotation vector in the racket frame. One GP per component (RBF + white
noise, scikit-learn) models it over time; the predicted error is then removed
from every sample: q_corr(s) = q_imu(s) * exp(gp(s)).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from . import config, gyro, quat


def unwrap_rotvecs(r):
    """Make a rotation-vector series continuous.

    r and r - 2*pi*r/|r| are the same rotation; near 180 deg the log map flips
    between them. Pick, for each vector, the form closest to the previous one.
    """
    out = r.copy()
    for i in range(1, len(r)):
        th = np.linalg.norm(r[i])
        if th < 1e-9:
            continue
        alt = r[i] - 2 * np.pi * r[i] / th
        if np.linalg.norm(alt - out[i - 1]) < np.linalg.norm(r[i] - out[i - 1]):
            out[i] = alt
    return out


def residuals(q_imu, samples, q_video):
    """(M, 3) continuous error rotation vectors (rad), racket frame."""
    q_s = gyro.sample_at(q_imu, samples)
    return unwrap_rotvecs(quat.to_rotvec(quat.mul(quat.conj(q_s), q_video)))


def sample_noise(samples, base=config.GP_NOISE_RAD ** 2, impact=config.GP_IMPACT_NOISE_RAD ** 2):
    """Per-moment extra noise variance: larger inside the impact window."""
    s = np.asarray(samples)
    lo, hi = config.IMPACT_DOWNWEIGHT
    return np.where((s >= lo) & (s <= hi), impact, base)


def make_gp(alpha):
    kernel = (ConstantKernel(0.5, (1e-3, 1e2)) * RBF(config.GP_LENGTH_S, config.GP_LENGTH_BOUNDS_S)
              + WhiteKernel(1e-3, (1e-6, 1.0)))
    return GaussianProcessRegressor(kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=3,
                                    random_state=0)


@dataclass
class GPModel:
    gps: list           # one regressor per rotation-vector component

    def predict(self, t_s):
        """Mean (N, 3) rad and std (N, 3) rad at times t_s (seconds)."""
        out = [g.predict(np.asarray(t_s)[:, None], return_std=True) for g in self.gps]
        return np.stack([o[0] for o in out], 1), np.stack([o[1] for o in out], 1)


def fit(t_s, r, alpha):
    """Fit one GP per component of r (M, 3) over time t_s (M,)."""
    return GPModel([make_gp(alpha).fit(np.asarray(t_s)[:, None], r[:, k]) for k in range(3)])


def correct(q_imu, model, fs=config.FS_HZ):
    """Corrected orientation per sample and GP 1-sigma (deg) per sample."""
    t = np.arange(len(q_imu)) / fs
    mean, std = model.predict(t)
    q = quat.make_continuous(quat.mul(q_imu, quat.from_rotvec(mean)))
    return q, np.rad2deg(np.linalg.norm(std, axis=1))


def leave_one_out(q_imu, samples, q_video, fs=config.FS_HZ):
    """Angle error (deg) at each moment when that moment is left out of the fit."""
    t = np.asarray(samples) / fs
    r = residuals(q_imu, samples, q_video)
    alpha = sample_noise(samples)
    err = np.zeros(len(samples))
    for i in range(len(samples)):
        keep = np.arange(len(samples)) != i
        m = fit(t[keep], r[keep], alpha[keep])
        mean, _ = m.predict(t[i:i + 1])
        q_i = quat.mul(gyro.sample_at(q_imu, samples[i:i + 1]), quat.from_rotvec(mean))
        err[i] = np.rad2deg(quat.angle_between(q_i, q_video[i:i + 1]))[0]
    return err
