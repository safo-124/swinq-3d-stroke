# Racket orientation from IMU (Hack For Humanity – Sports Telemetry)

Reconstruct the 3D orientation of a tennis racket through one forehand from a
6-axis IMU (416 Hz, 400 samples), validated against two videos.

## Layout

```
data/raw_data.csv     IMU: index, ax ay az (g), gx gy gz (deg/s)
data/anchors.json     hand-clicked butt/throat/tip/edge pixels, 29 moments x 2 views
video/                swing_angle_1.mp4 (front), swing_angle_2.mp4 (rear oblique)
src/racket/           package: config.py (all tunables), io.py, ...
scripts/stageN_*.py   one script per stage
tests/                pytest
out/                  generated plots / CSV / videos
```

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

## Stage 1 – inspection

```
.venv\Scripts\python scripts\stage1_inspect.py
```

Outputs `out/stage1_imu.png`, `out/stage1_anchors.png`.

- 400 samples = 0.962 s. Peak |ω| 1720 deg/s. ax pinned at 15.96 g on samples 182–199.
- ax vs centripetal load (gy²+gz²): corr 0.92 pre-impact (radius 0.65 m),
  0.99 post-impact (radius 0.40 m). X axis along the handle is confirmed.
- Video pts times match `t_from_impact_s` exactly in both views. Front video has
  one dropped frame (66.7 ms gap after frame 99), outside the IMU window.
- **Anchor fix:** front frame 118 had its four labels rotated by one
  (edge→butt, butt→throat, throat→tip, tip→edge). It is corrected at load time
  via `config.ANCHOR_RELABEL`; `anchors.json` itself is unchanged.
