"""Gyro strapdown integration into a quaternion orientation history."""
import numpy as np

from . import config, quat


def body_rates(gyro_dps, bias_dps=(0.0, 0.0, 0.0), sign=1):
    """Corrected angular rate in rad/s: sign * (gyro - bias).

    `sign` (+1 / -1) covers the unconfirmed handedness of positive rotation.
    """
    return sign * np.deg2rad(np.asarray(gyro_dps, float) - np.asarray(bias_dps, float))


def increments(omega, fs=config.FS_HZ):
    """Per-interval rotation quaternions from the mean rate of each interval.

    Interval k (sample k -> k+1) rotates by exp(0.5 * (w_k + w_{k+1}) * dt),
    expressed in the body frame at sample k.
    """
    mean_rate = 0.5 * (omega[:-1] + omega[1:])
    return quat.from_rotvec(mean_rate / fs)


def integrate(gyro_dps, q0=quat.IDENTITY, bias_dps=(0.0, 0.0, 0.0), sign=1, fs=config.FS_HZ):
    """Orientation at every sample, (N, 4), starting from q0 at sample 0.

    q_{k+1} = q_k * dq_k (body-frame increment, right multiplication).
    """
    dq = increments(body_rates(gyro_dps, bias_dps, sign), fs)
    q = np.empty((len(dq) + 1, 4))
    q[0] = quat.normalize(q0)
    for k, d in enumerate(dq):
        q[k + 1] = quat.mul(q[k], d)
    return quat.normalize(q)


def sample_at(q, s):
    """Orientation at fractional sample index/indices s, by slerp.

    Indices outside [0, N-1] are clamped to the ends.
    """
    s = np.clip(np.asarray(s, float), 0.0, len(q) - 1.0)
    i0 = np.minimum(np.floor(s).astype(int), len(q) - 2)
    return quat.slerp(q[i0], q[i0 + 1], s - i0)
