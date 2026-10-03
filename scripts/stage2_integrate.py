"""Stage 2: integrate the real gyro (identity start, zero bias, both signs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from racket import config, gyro, quat
from racket.io import load_anchors, load_imu


def discretisation_error(g, up=4):
    """Max angle (deg) between 416 Hz integration and a spline-upsampled one."""
    s = np.arange(len(g))
    fine = np.linspace(0, len(g) - 1, (len(g) - 1) * up + 1)
    q_ref = gyro.integrate(CubicSpline(s, g, axis=0)(fine), fs=config.FS_HZ * up)[::up]
    return np.rad2deg(quat.angle_between(gyro.integrate(g), q_ref))


def bias_sensitivity(g, sign, step=1.0):
    """End-orientation change (deg) for a +1 deg/s bias on each axis."""
    q_ref = gyro.integrate(g, sign=sign)[-1]
    out = []
    for k in range(3):
        b = np.zeros(3); b[k] = step
        out.append(np.rad2deg(quat.angle_between(gyro.integrate(g, bias_dps=b, sign=sign)[-1], q_ref)))
    return np.array(out)


def plot(q_by_sign, samples, path):
    """Quaternion (sign +1), angle from start, and handle direction in world."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    s = np.arange(len(q_by_sign[1]))
    for k, c in enumerate("wxyz"):
        axes[0].plot(s, q_by_sign[1][:, k], label=f"q{c}")
    axes[0].set_title("Quaternion, sign=+1, q0=identity, bias=0")
    for sign, q in q_by_sign.items():
        ang = np.rad2deg(quat.angle_between(q, q[0]))
        axes[1].plot(s, ang, label=f"sign {sign:+d}")
        x_world = quat.rotate(q, np.array([1.0, 0, 0]))
        for k, c in enumerate("xyz"):
            axes[2].plot(s, x_world[:, k], ls="-" if sign == 1 else "--",
                         label=f"handle·{c} (sign {sign:+d})")
    axes[1].set_title("Angle from starting orientation (deg)")
    axes[2].set_title("Handle axis (+X body) in the starting frame")
    for ax in axes:
        ax.axvline(config.IMPACT_SAMPLE, color="k", ls=":", lw=1)
        ax.axvspan(*config.GYRO_UNRELIABLE, color="orange", alpha=.15)
        for m in samples:
            ax.axvline(m, color="grey", lw=.3, alpha=.5)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[2].set_xlabel("sample (grey lines = 29 video moments)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    imu = load_imu()
    samples = load_anchors()["front"].imu_sample
    q_by_sign = {s: gyro.integrate(imu.gyro, sign=s) for s in (1, -1)}

    for sign, q in q_by_sign.items():
        dq = quat.angle_between(q[1:], q[:-1])
        print(f"sign {sign:+d}: path length {np.rad2deg(dq.sum()):.0f} deg, "
              f"start->impact {np.rad2deg(quat.angle_between(q[0], q[200])):.0f} deg, "
              f"start->end {np.rad2deg(quat.angle_between(q[0], q[-1])):.0f} deg, "
              f"impact->end {np.rad2deg(quat.angle_between(q[200], q[-1])):.0f} deg")
        print(f"         end-orientation change per +1 deg/s bias (x,y,z): {bias_sensitivity(imu.gyro, sign).round(2)} deg")

    a = np.rad2deg(quat.angle_between(q_by_sign[1], q_by_sign[-1]))
    print(f"sign +1 vs -1 differ by up to {a.max():.0f} deg (so video must pick the sign)")
    err = discretisation_error(imu.gyro)
    print(f"416 Hz vs 4x spline-upsampled integration: max {err.max():.3f} deg, end {err[-1]:.3f} deg")

    print("angle from start at video moments (sign +1 / -1):")
    for m in samples[::4]:
        a1, a2 = (np.rad2deg(quat.angle_between(gyro.sample_at(q, m), q[0])) for q in q_by_sign.values())
        print(f"  sample {m:6.1f}: {a1:6.1f} / {a2:6.1f}")

    plot(q_by_sign, samples, config.OUT_DIR / "stage2_integration.png")
    print("wrote out/stage2_integration.png")


if __name__ == "__main__":
    main()
