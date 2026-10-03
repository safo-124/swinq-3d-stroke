"""Stage 1: plot IMU channels and overlay anchors on video frames."""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from racket import config
from racket.io import frame_times, load_anchors, load_imu, read_frames

COLORS = {"butt": (0, 0, 255), "throat": (0, 200, 0), "tip": (255, 0, 0), "edge": (0, 220, 255)}


def plot_imu(imu, path):
    """Six channels, impact and the clipped/unreliable regions shaded."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    s = np.arange(imu.n)
    for ax, data, lab, unit in [(axes[0], imu.accel, "a", "g"), (axes[1], imu.gyro, "g", "deg/s")]:
        for k, c in enumerate("xyz"):
            ax.plot(s, data[:, k], lw=1, label=f"{lab}{c}")
        ax.axvline(config.IMPACT_SAMPLE, color="k", ls="--", lw=1, label="impact")
        ax.set_ylabel(unit)
        ax.legend(loc="upper left", ncol=4, fontsize=8)
        ax.grid(alpha=.3)
    axes[0].axvspan(*config.ACCEL_CLIPPED, color="red", alpha=.12, label="ax clipped")
    axes[0].axhline(-config.ACCEL_RANGE_G, color="red", lw=.6, ls=":")
    axes[0].axhline(config.ACCEL_RANGE_G, color="red", lw=.6, ls=":")
    axes[1].axvspan(*config.GYRO_UNRELIABLE, color="orange", alpha=.2)
    axes[1].set_xlabel(f"sample (@ {config.FS_HZ:.0f} Hz)")
    axes[0].set_title("Raw IMU (red: ax clipped, orange: post-impact transient)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_anchors(img, uv, label):
    """Draw racket skeleton butt-throat-tip plus edge on a copy of img."""
    img = img.copy()
    p = {n: tuple(int(round(v)) for v in uv[i]) for i, n in enumerate(config.POINT_NAMES)}
    cv2.line(img, p["butt"], p["throat"], (255, 255, 255), 2)
    cv2.line(img, p["throat"], p["tip"], (255, 255, 255), 2)
    cv2.line(img, p["throat"], p["edge"], (180, 180, 180), 1)
    for n, xy in p.items():
        cv2.circle(img, xy, 5, COLORS[n], -1)
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def anchor_sheet(anchors, path, picks=(0, 7, 14, 21, 28)):
    """Grid: rows = views, cols = chosen moments, crops around the racket."""
    tiles = []
    for view in ("front", "rear"):
        a = anchors[view]
        frames = read_frames(config.VIDEOS[view], wanted=a.frames[list(picks)])
        row = []
        for m in picks:
            img, pts = frames[a.frames[m]]
            lab = f"{view} f{a.frames[m]} s{a.imu_sample[m]:.0f}"
            img = draw_anchors(img, a.uv[m], lab)
            cx, cy = a.uv[m].mean(0).astype(int)
            x0, y0 = np.clip([cx - 240, cy - 200], 0, [1280 - 480, 720 - 400])
            row.append(img[y0:y0 + 400, x0:x0 + 480])
        tiles.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(tiles))


def main():
    config.OUT_DIR.mkdir(exist_ok=True)
    imu = load_imu()
    anchors = load_anchors()
    print(f"IMU: {imu.n} samples, {imu.t[-1] + 1 / config.FS_HZ:.3f} s")
    print(f"  max |a| per axis (g):     {np.abs(imu.accel).max(0).round(2)}")
    print(f"  max |w| per axis (deg/s): {np.abs(imu.gyro).max(0).round(0)}")
    print(f"  max |w| total:            {np.linalg.norm(imu.gyro, axis=1).max():.0f} deg/s")
    axmax = np.abs(imu.accel[:, 0]).max()
    clipped = np.where(np.abs(imu.accel[:, 0]) >= axmax - 0.02)[0]
    print(f"  ax within 0.02 g of its max ({axmax:.3f} g) on samples: {clipped.tolist()}")
    i = config.IMPACT_SAMPLE
    print(f"  gyro at 199/200/201: {imu.gyro[i-1].round(0)} / {imu.gyro[i].round(0)} / {imu.gyro[i+1].round(0)}")
    w2 = np.deg2rad(imu.gyro[:, 1]) ** 2 + np.deg2rad(imu.gyro[:, 2]) ** 2
    ax_ms2 = imu.accel[:, 0] * 9.81
    for name, m in [("pre-impact 0-181", np.r_[0:182]), ("post-impact 231-399", np.r_[231:400])]:
        r = np.corrcoef(ax_ms2[m], w2[m])[0, 1]
        radius = (ax_ms2[m] @ w2[m]) / (w2[m] @ w2[m])
        print(f"  {name}: corr(ax, gy^2+gz^2) = {r:.2f}, radius (through origin) = {radius:.2f} m")

    for view, a in anchors.items():
        bt = np.linalg.norm(a.uv[:, 0] - a.uv[:, 1], axis=1)
        tt = np.linalg.norm(a.uv[:, 2] - a.uv[:, 1], axis=1)
        print(f"{view}: butt-throat / throat-tip pixel ratio {(bt/tt).min():.2f}..{(bt/tt).max():.2f} "
              f"(after relabel fixes: {[k for k in config.ANCHOR_RELABEL if k[0] == view]})")
        ts = frame_times(config.VIDEOS[view])
        dt = np.diff(ts)
        win = ts[a.frames]
        rel = win - ts[config.IMPACT_FRAME[view]]
        print(f"{view}: {len(ts)} frames, fps~{1/np.median(dt):.2f}, dt range {dt.min()*1e3:.1f}..{dt.max()*1e3:.1f} ms")
        print(f"  window frames {a.frames[0]}..{a.frames[-1]} ({len(a.frames)}), "
              f"pts rel. impact {rel[0]:+.4f}..{rel[-1]:+.4f} s; "
              f"max |pts - json t| = {np.abs(rel - a.t_from_impact).max()*1e3:.1f} ms")
    f, r_ = anchors["front"], anchors["rear"]
    print(f"front/rear frame offset all 41: {np.all(r_.frames - f.frames == 41)}; "
          f"imu_sample equal: {np.allclose(f.imu_sample, r_.imu_sample)}")

    plot_imu(imu, config.OUT_DIR / "stage1_imu.png")
    anchor_sheet(anchors, config.OUT_DIR / "stage1_anchors.png")
    print("wrote out/stage1_imu.png, out/stage1_anchors.png")


if __name__ == "__main__":
    main()
