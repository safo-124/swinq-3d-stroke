# SWINQ: 3D stroke from sensor data (Hack For Humanity – Sports Telemetry)

Reconstruct the 3D orientation of a tennis racket through one forehand from the
SWINQ dampener's 6-axis IMU (416 Hz, 400 samples, 0.96 s), and render it in Blender.
The racket model comes from the sensor alone; the two videos give the starting
orientation and an independent check.

## Headline results

| | |
|---|---|
| Orientation | gyro strapdown with exact exponential steps (0.03° error on a synthetic 500°/s moving-axis test) |
| Start orientation | from video moments 0–4; gyro (sign −1) then matches video to **2.0–4.4°** |
| Peak angular velocity | **1720°/s**, 2.4 ms before impact |
| Swing rate at impact | 1528°/s (roll about the handle 791°/s) |
| Estimated speed at impact | sensor 17.4 m/s, head centre **24.6 m/s (89 km/h)**, tip 29.5 m/s (pivot radius 0.65 m from ax ≈ r·ω⊥², corr 0.92) |
| Rotation in the last 100 ms | 85° |
| Saturation repair | ax clipped at 16 g on samples 182–199; filled with r·ω⊥² + c (fit corr 1.00) → peak about 29.5 g |
| String vibration | 164.5 Hz as sampled; at 416 Hz this is an alias of 251 or **581 Hz** (typical string-bed range) |
| Video check | agreement within 5° up to about 0.18 s before impact; diverges after (see "Known limitation") |

## Run everything

```
.venv\Scripts\python scripts\stage1_inspect.py
.venv\Scripts\python scripts\stage2_integrate.py
.venv\Scripts\python scripts\stage3_bundle.py          # ~1.5 min, video orientations
.venv\Scripts\python scripts\stage4_sensor_model.py    # orientation.csv, motion params, 3D animation
blender --python blender\animate_racket.py             # Blender scene, keyframed from out/orientation.csv
blender --background --python blender\animate_racket.py -- --render out\blender_stroke.mp4
```

Deliverables in `out/`: `orientation.csv` (per sample: racket quaternion in the
camera-1 frame, plus `bl_q*` in a z-up Blender world), `motion_params.json`,
`stroke_3d.mp4` (matplotlib 3D animation; no Blender needed), `stage4_summary.png`.

## Known limitation (stated plainly)

Gyro and video orientations agree to 2–5° for the first ~0.2 s, then diverge, by
100–170° around and after impact. We tested and ruled out:
- the rotation sign and all 48 signed axis permutations of the gyro (best 13° median between consecutive moments);
- the IMU sample rate (416–1666 Hz scanned);
- a joint fit with bias, mounting correction and time offset, both on per-frame video
  orientations and directly on the clicked pixels (`scripts/diagnostics/pixel_fit.py`).
  Every fit needs biases of 200–450 deg/s and a 70–150° mounting correction, which is
  not physical, and still leaves about 10 px.

Through mid-swing the gyro measures 2–3× more rotation than the video shows, yet the
accelerometer confirms the gyro scale (centripetal radius 0.65 m is physical). The
remaining suspects are the video/IMU time alignment and the per-frame video poses at
30 fps with heavy motion blur. The planned GP correction (stage 5) was dropped: it would
only paper over a discrepancy of this size.

## Layout

```
data/raw_data.csv     IMU: index, ax ay az (g), gx gy gz (deg/s)
data/anchors.json     hand-clicked butt/throat/tip/edge pixels, 29 moments x 2 views
video/                swing_angle_1.mp4 (front), swing_angle_2.mp4 (rear oblique)
src/racket/           package: config.py (all tunables), io.py, ...
scripts/stageN_*.py   one script per stage
blender/              Blender scene script
scripts/diagnostics/  IMU-vs-video investigation
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

## Stage 3 – cameras and racket from the videos

```
.venv\Scripts\python scripts\stage3_bundle.py            # ~1.5 min
.venv\Scripts\python scripts\stage3_bundle.py --fixed-geometry
```

Bundle adjustment (`racket.cameras`): focal length per camera (principal point
at the image centre, no distortion), camera-2 pose relative to camera 1, and the
racket pose at each of the 29 moments. Minimises 4-point reprojection error in both
views with a soft-L1 loss (3 px knee). World = camera 1; scale comes from the fixed
0.685 m butt–tip length. Outputs `out/video_orientation.csv` (racket→camera-1
quaternion, translation, per-moment RMS, 1σ per body axis), `out/stage3_cameras.json`,
and the plots `stage3_errors.png`, `stage3_scene.png`, `stage3_reprojection.png`.

How it works:
- Homography/IPPE PnP fails because butt, throat and tip are collinear. Single-view
  poses are found instead by a batched multi-start LM (24 octahedral starts). Each
  view gives 2 mirror-ambiguous poses; the camera-2 rotation most moments agree on
  seeds the joint fit.
- Translations are parametrised as (ray a, b, log f/z). The racket is about 7 m away
  (close to weak perspective), so focal length and depth trade off against each other.
- After each joint fit, every moment is re-seeded from all of its mirror candidates
  with the cameras held fixed. This repaired moments 0 and 10, which had settled in
  the wrong mirror pose. Both focal starts now converge to the same answer (0.02° spread).
- Synthetic test: a noise-free scene is recovered to < 0.05° and f to 0.1%.

Results:
- **Focal lengths: front 2224 px, rear 1613 px.** Camera 2 is rotated 87° from camera 1,
  10.2 m away; the racket is 6.9–7.1 m from camera 1.
- **Reprojection: median 4.3 px, RMS front 6.6 px, rear 6.0 px.** Worst moment: front
  frame 117 at 14.7 px, the most blurred frame. The throat click is the noisiest point
  (8.0 px median, front).
- **Orientation 1σ per moment** (with the cameras held fixed, using a robust noise
  estimate): about 4.9° roll about the handle and 1.2–1.3° about the other two axes.
- **Refined geometry** (sensor frame, m): butt +0.231, tip −0.454, edge (−0.272, +0.149).
  The brief's guess (butt +0.36 / tip −0.33) does not fit the clicks: the clicked throat
  sits at 0.33 of butt→tip. Fixed geometry raises the cost from 2430 to 2908.
- Early check against the gyro (zero bias): with sign −1, the rotations between moments
  0–4 agree to 3–5°; sign +1 is worse. Across all moments the rotation between
  consecutive moments disagrees by about 18° median. No signed axis permutation
  explains this (the best is 13°), which points to video noise and sync. Stage 4 handles it.

## Stage 4 – joint fit (attempted) and the sensor-only model

`racket.joint` fits q0, gyro bias, mounting correction, time offset and sign to the
29 video orientations (soft-L1, impact moments down-weighted). It does not converge
to a physical solution (see "Known limitation"). The delivered model
(`racket.motion`, `scripts/stage4_sensor_model.py`) therefore uses the video only for
the starting orientation (moments 0–4, sign −1, zero bias) and the IMU for everything else.
