"""Sensor-only stroke model: orientation from the gyro plus key motion parameters.

The video is used for one thing only: the starting orientation (a gyro cannot
observe it, and there is no still period for gravity). It is taken from the
first few video moments, where gyro and video agree, and everything after
that comes from the IMU alone.
"""
import numpy as np

from . import config, gyro, joint, quat

# camera-1 frame (x right, y down, z forward) -> Blender world (z up)
CAM_TO_BLENDER = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def anchor_start(g, samples, q_video, sign=config.GYRO_SIGN, n=config.ANCHOR_MOMENTS):
    """Starting orientation (sensor -> camera 1) from the first n video moments.

    Zero bias, no mounting correction: q0 = chordal mean of q_video_i * Q_i^-1.
    Returns (q0, per-moment angle error over those n moments in degrees).
    """
    q0 = joint.init_q0(g, samples[:n], q_video[:n], sign)
    Q = gyro.sample_at(gyro.integrate(g, sign=sign), samples[:n])
    err = np.rad2deg(quat.angle_between(quat.mul(q0, Q), q_video[:n]))
    return q0, err


def orientation(g, q0, sign=config.GYRO_SIGN):
    """Racket -> camera-1 orientation at every sample, sign-continuous."""
    return quat.make_continuous(gyro.integrate(g, q0=q0, sign=sign))


def to_blender(q):
    """Re-express camera-1 orientations in a z-up Blender world."""
    return quat.make_continuous(quat.mul(quat.from_matrix(CAM_TO_BLENDER), q))


def pivot_radius(accel, gyro_dps, window=config.RADIUS_FIT_WINDOW):
    """Effective pivot radius: ax = r * (wy^2 + wz^2), through the origin.

    Returns (radius m, correlation)."""
    w2 = np.deg2rad(gyro_dps[:, 1]) ** 2 + np.deg2rad(gyro_dps[:, 2]) ** 2
    a = accel[:, 0] * 9.81
    i = np.arange(*window)
    return float(a[i] @ w2[i] / (w2[i] @ w2[i])), float(np.corrcoef(w2[i], a[i])[0, 1])


def reconstruct_clipped_ax(accel, gyro_dps, clipped=config.ACCEL_CLIPPED,
                           fit_window=config.AX_FIT_WINDOW):
    """Fill saturated ax with a local centripetal model ax = r * (wy^2 + wz^2) + c.

    r and c are fitted on the last unclipped samples so the fill joins the
    measured curve; c absorbs gravity and tangential terms.
    Returns (ax_filled in g, local r m, c g, fit correlation).
    """
    w2 = np.deg2rad(gyro_dps[:, 1]) ** 2 + np.deg2rad(gyro_dps[:, 2]) ** 2
    a = accel[:, 0] * 9.81
    i = np.arange(*fit_window)
    r, c = np.polyfit(w2[i], a[i], 1)
    corr = float(np.corrcoef(w2[i], a[i])[0, 1])
    filled = accel[:, 0].copy()
    j = np.arange(clipped[0], clipped[1] + 1)
    filled[j] = (r * w2[j] + c) / 9.81
    return filled, float(r), float(c / 9.81), corr


def string_vibration_hz(accel, start=config.IMPACT_SAMPLE + 1, n=128, fs=config.FS_HZ):
    """Dominant post-impact az frequency (as sampled) and its likely aliases."""
    x = accel[start:start + n, 2]
    x = (x - np.polyval(np.polyfit(np.arange(n), x, 2), np.arange(n))) * np.hanning(n)
    spec = np.abs(np.fft.rfft(x, 8 * n))
    freqs = np.fft.rfftfreq(8 * n, 1 / fs)
    f = float(freqs[np.argmax(spec[1:]) + 1])
    return f, [fs - f, fs + f]


def motion_parameters(imu, q, radius_m):
    """Key stroke numbers from the IMU alone."""
    g = imu.gyro
    w = np.linalg.norm(g, axis=1)
    i_imp = config.IMPACT_SAMPLE
    pre = slice(0, i_imp)
    k_peak = int(np.argmax(w[pre]))
    w_perp = np.deg2rad(np.hypot(g[:, 1], g[:, 2]))           # swing rate, not roll
    names = list(config.POINT_NAMES)
    tip_offset = -config.RACKET_POINTS["tip"][0]
    sweet_offset = -config.RACKET_POINTS["edge"][0]               # head centre
    impact_rate = w_perp[i_imp - 1]
    rot_last_100ms = np.rad2deg(quat.angle_between(q[i_imp - int(0.1 * config.FS_HZ)], q[i_imp - 1]))
    return {
        "peak_angular_velocity_dps": float(w[pre].max()),
        "peak_angular_velocity_sample": k_peak,
        "peak_angular_velocity_t_before_impact_ms": float((i_imp - k_peak) / config.FS_HZ * 1e3),
        "angular_velocity_at_impact_dps": float(w[i_imp - 1]),
        "roll_rate_at_impact_dps": float(g[i_imp - 1, 0]),
        "swing_rate_at_impact_dps": float(np.rad2deg(impact_rate)),
        "pivot_radius_m": float(radius_m),
        "speed_at_sensor_ms": float(impact_rate * radius_m),
        "speed_at_head_centre_ms": float(impact_rate * (radius_m + sweet_offset)),
        "speed_at_tip_ms": float(impact_rate * (radius_m + tip_offset)),
        "rotation_last_100ms_before_impact_deg": float(rot_last_100ms),
        "total_rotation_path_deg": float(np.rad2deg(quat.angle_between(q[1:], q[:-1]).sum())),
        "impact_sample": i_imp,
        "impact_gyro_drop_dps": float(w[i_imp - 1] - w[i_imp + 1]),
        "point_names": names,
    }
