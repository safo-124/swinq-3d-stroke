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

## Stage 2 – gyro integration

```
.venv\Scripts\python scripts\stage2_integrate.py
.venv\Scripts\python -m pytest
```

`racket.gyro.integrate(gyro_dps, q0, bias_dps, sign)` returns (N, 4) quaternions
[w, x, y, z], body→world, with q_{k+1} = q_k ⊗ exp(½·(ω_k+ω_{k+1})·dt) and
ω = sign·(gyro − bias). `gyro.sample_at(q, s)` slerps to fractional samples.
Quaternion series are kept sign-continuous (no q→−q jumps).

- Tests: exact for constant-axis rotation; matches a reference product of
  scipy exponentials; on an analytic moving-axis rotation (~500 deg/s) the max
  error is 0.027° at 416 Hz, falling 4× per doubling of the sample rate (second order).
- Real data: 571° of total rotation. Start→impact 163° (sign +1) vs 134° (sign −1).
  The two signs differ by up to 180°, so the video has to decide the sign.
- 416 Hz vs a 4× spline-upsampled integration: max 0.18°, so step size is negligible.
- A 1 deg/s bias on any axis moves the final orientation by 0.3–0.6°.
