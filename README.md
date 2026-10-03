# SWINQ: 3D stroke from sensor data

**Hack For Humanity – Sports Telemetry challenge.** We rebuild a full tennis forehand in 3D from the
6-axis IMU inside a SWINQ string dampener: 400 readings at 416 Hz (0.96 s), starting mid-swing
with no still moment, and with the accelerometer saturated at 16 g just before impact. The racket's
orientation is integrated from the gyroscope, the clipped acceleration is rebuilt from physics, and
the key stroke parameters come from the sensor alone. Two uncalibrated videos are turned into an
independent 3D reference by using the racket itself as the calibration object, and a
Gaussian-process drift correction, tested on frames it never saw, brings the sensor's racket onto
the video's.

## Final results

### Stroke parameters (sensor only)

| Quantity | Result |
|---|---|
| Peak angular velocity | **1720°/s**, 2.4 ms before impact |
| Swing rate / roll rate before impact | 1528°/s / 791°/s |
| Racket-head speed at impact | **24.6 m/s ≈ 89 km/h** (sensor 17.4 m/s, tip 29.5 m/s); pivot radius 0.65 m from aₓ ≈ r·ω⊥² (corr 0.92) |
| Racket rotation in the last 100 ms | **85°** |
| Total rotation over the record | 571° |
| Peak acceleration (saturated at 16 g) | **≈ 30 g** rebuilt from aₓ ≈ r·ω⊥² + c (fit corr 1.00); plausible range 30–47 g |
| String vibration | 164.5 Hz as sampled → most likely **581 Hz** (alias at 416 Hz sampling) |

### Accuracy

| Check | Result |
|---|---|
| Integrator on a synthetic 500°/s moving-axis rotation | max **0.027°** at 416 Hz (error falls 4× per doubling of the rate) |
| Two-camera reconstruction | focal lengths 2224 / 1613 px, cameras 87° apart; median reprojection **4.3 px**; orientation 1σ ≈ 1.3° (tilt), 4.9° (roll about the handle) |
| Sensor-only vs video | within **2–11°** for the first 0.27 s, then drifts to 100–170° |
| **Gyro + GP drift correction vs video, leave-one-out** | mean **12.4°**, median **8.3°** (was 88.4° mean); worst 44° at the impact frame |

`out/orientation.csv` is the GP-corrected orientation, which fuses sensor and video.
`out/orientation_sensor_only.csv` is the pure sensor result. All stroke parameters above are
sensor-only.

## Deliverables

| What | Where |
|---|---|
| Racket orientation for every sample (GP-corrected, with `gp_std_deg`, plus Blender z-up `bl_q*` columns) | `out/orientation.csv` |
| Same, sensor only | `out/orientation_sensor_only.csv` |
| Stroke parameters | `out/motion_params.json`, `out/stage4_summary.png` |
| Before/after GP video (orange = sensor only, green = gyro + GP, 10× slow) | `out/gp_correction.mp4` |
| Side-by-side demo (front, rear, corrected racket; 10× slow) | `out/demo.mp4` |
| 3D animations (no Blender needed) | `out/stroke_3d.mp4` (sensor only), `out/stroke_3d_gp.mp4` (corrected) |
| Overlays on both videos | `out/overlay_front.mp4`, `out/overlay_rear.mp4`, `out/overlay_contact_sheet.png` |
| Blender scene script, plus a ready-to-run bundle with the CSV | `blender/animate_racket.py`, `blender.zip` |
| Pitch script (3 min, plain-language notes, judge Q&A) | `out/SWINQ_pitch.pdf` |
| Beamer slides (Tampere purple, TikZ, transitions), Overleaf-ready | `latex/main.tex`, `out/swinq_beamer_overleaf.zip` |
| Scientific report (IMRaD, full maths, references) | `latex/report.tex` |

## Quick start

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest                          # 19 tests
```

Run the pipeline in order:

```
.venv\Scripts\python scripts\stage1_inspect.py          # IMU and anchor plots
.venv\Scripts\python scripts\stage2_integrate.py        # integration checks
.venv\Scripts\python scripts\stage3_bundle.py           # ~1.5 min: cameras + video orientations
.venv\Scripts\python scripts\stage4_sensor_model.py     # sensor-only orientation, stroke parameters
.venv\Scripts\python scripts\stage5_gp.py               # GP drift correction -> out/orientation.csv
.venv\Scripts\python scripts\stage6_overlay.py          # overlays on both videos
.venv\Scripts\python scripts\make_demo.py               # out/demo.mp4
.venv\Scripts\python scripts\make_gp_video.py           # out/gp_correction.mp4
.venv\Scripts\python scripts\make_pitch_pdf.py          # out/SWINQ_pitch.pdf from the slide notes
```

**Blender:** unzip `blender.zip` (it keeps the `blender/` and `out/` folders), then

```
blender --python blender\animate_racket.py
blender --background --python blender\animate_racket.py -- --render out\blender_stroke.mp4
```

The script builds a racket, keyframes it from `out/orientation.csv` at 8× slow motion and marks
impact on the timeline. It has not yet been run in Blender, so please report any errors.

**LaTeX:** upload `out/swinq_beamer_overleaf.zip` to Overleaf for the slides. For the report, use
`latex/report.tex` as the main file, with the images from `latex/img/` in an `img/` folder.

## Known limitation (stated plainly)

In the fast phase of the swing the gyroscope measures 2–3× more rotation than the 30 fps video
shows, so the sensor-only orientation drifts away from the video after about 0.2 s before impact.
We tested and ruled out:
- the rotation sign and all 48 signed axis permutations of the gyro (best 13° median between consecutive moments);
- the IMU sample rate (416–1666 Hz scanned);
- a constant gyro bias, a mounting rotation and a time offset, fitted both to the per-frame video
  orientations and directly to the clicked pixels (`scripts/diagnostics/pixel_fit.py`). Every fit
  needs 200–450°/s of bias and a 70–150° mounting rotation, which is not physical;
- mirror-pose errors in the video reconstruction (2 frames detected and repaired).

The accelerometer supports the gyroscope's scale: the 0.65 m pivot radius is physical, whereas a
gyro reading 2.5–3× too high would imply 4–6 m. The remaining suspects are the time alignment
between sensor and cameras (impact is known only to ±1 frame, 33 ms, in which the racket turns up to
57°) and motion blur at 30 fps. The GP correction compensates for the drift; proper synchronisation
is the next step towards a sensor-only result that holds through impact and towards 6-DoF tracking.

## Layout

```
data/raw_data.csv        IMU: index, ax ay az (g), gx gy gz (deg/s)
data/anchors.json        hand-clicked butt/throat/tip/edge pixels, 29 moments x 2 views
video/                   swing_angle_1.mp4 (front), swing_angle_2.mp4 (rear oblique)
src/racket/              package: config (all tunables), io, quat, gyro, cameras, joint, motion, gp
scripts/stageN_*.py      one script per stage; make_*.py for videos and the pitch PDF
scripts/diagnostics/     IMU-vs-video investigation (pixel-space fit)
blender/                 Blender scene script
latex/                   Beamer slides (main.tex), scientific report (report.tex), img/
pitch_deck/              slide-deck sources (HTML slides + speaker notes)
tests/                   pytest (quaternions, integration, loaders, unwrapping, synthetic BA)
out/                     generated CSVs, plots, videos, PDFs
```

---

# Stage details

## Stage 1 – inspection

```
.venv\Scripts\python scripts\stage1_inspect.py
```

Outputs `out/stage1_imu.png`, `out/stage1_anchors.png`.

- 400 samples = 0.962 s. Peak |ω| 1720 deg/s. aₓ pinned at 15.96 g on samples 182–199.
- aₓ vs centripetal load (gy²+gz²): corr 0.92 before impact (radius 0.65 m), 0.99 after impact
  (radius 0.40 m). The x axis along the handle is confirmed.
- Video timestamps match `t_from_impact_s` exactly in both views. The front video has one dropped
  frame (66.7 ms gap after frame 99), outside the IMU window.
- **Anchor fix:** front frame 118 had its four labels rotated by one (edge→butt, butt→throat,
  throat→tip, tip→edge). It is corrected at load time via `config.ANCHOR_RELABEL`;
  `anchors.json` itself is unchanged.

## Stage 2 – gyro integration

```
.venv\Scripts\python scripts\stage2_integrate.py
```

`racket.gyro.integrate(gyro_dps, q0, bias_dps, sign)` returns (N, 4) quaternions [w, x, y, z],
body→world, with q_{k+1} = q_k ⊗ exp(½·(ω_k+ω_{k+1})·dt) and ω = sign·(gyro − bias).
`gyro.sample_at(q, s)` slerps to fractional samples. Quaternion series are kept sign-continuous.

- Tests: exact for constant-axis rotation; matches a reference product of scipy exponentials; on an
  analytic moving-axis rotation (~500°/s) the max error is 0.027° at 416 Hz, falling 4× per
  doubling of the sample rate (second order).
- Real data: 571° of total rotation. Start→impact 163° (sign +1) vs 134° (sign −1).
- 416 Hz vs a 4× spline-upsampled integration: max 0.18°, so step size is negligible.
- A 1°/s bias on any axis moves the final orientation by 0.3–0.6°.

## Stage 3 – cameras and racket from the videos

```
.venv\Scripts\python scripts\stage3_bundle.py            # ~1.5 min
.venv\Scripts\python scripts\stage3_bundle.py --fixed-geometry
```

Bundle adjustment (`racket.cameras`): focal length per camera (principal point at the image centre,
no distortion), camera-2 pose relative to camera 1, and the racket pose at each of the 29 moments,
minimising 4-point reprojection error in both views with a soft-L1 loss (3 px knee). World =
camera 1; scale from the fixed 0.685 m butt–tip length. Outputs `out/video_orientation.csv`,
`out/stage3_cameras.json`, `stage3_errors.png`, `stage3_scene.png`, `stage3_reprojection.png`.

- Homography/IPPE PnP fails because butt, throat and tip are collinear; single-view poses come from
  a batched multi-start Levenberg–Marquardt (24 octahedral starts), and the camera-2 rotation most
  moments agree on seeds the joint fit.
- Translations are parametrised as (ray a, b, log f/z) because the racket is ~7 m away (close to
  weak perspective), where focal length and depth trade off.
- After each joint fit, every moment is re-seeded from all of its mirror candidates with the cameras
  fixed; this repaired moments 0 and 10. Both focal starts converge to the same answer (0.02°).
- Results: focal lengths 2224 / 1613 px; camera 2 rotated 87°, 10.2 m away; racket 6.9–7.1 m from
  camera 1; reprojection median 4.3 px (RMS 6.6 / 6.0 px); worst moment front frame 117 (14.7 px,
  most blurred); orientation 1σ ≈ 4.9° roll, 1.2–1.3° on the other axes.
- Refined geometry (sensor frame, m): butt +0.231, tip −0.454, edge (−0.272, +0.149). The brief's
  guess (butt +0.36 / tip −0.33) does not fit the clicks; fixed geometry raises the cost from 2430 to 2908.
- Synthetic test: a noise-free scene is recovered to < 0.05° and f to 0.1%.

## Stage 4 – joint fit (attempted) and the sensor-only model

`racket.joint` fits q0, gyro bias, mounting correction, time offset and sign to the 29 video
orientations (soft-L1, impact moments down-weighted). It does not converge to a physical solution
(see "Known limitation"). The sensor-only model (`racket.motion`, `scripts/stage4_sensor_model.py`)
uses the video only for the starting orientation (chordal mean over moments 0–4, sign −1, zero
bias; error 2.0–4.4° on those moments) and the IMU for everything else. It writes
`out/orientation_sensor_only.csv`, `out/motion_params.json`, `out/stage4_summary.png` and
`out/stroke_3d.mp4`.

## Stage 5 – GP drift correction

```
.venv\Scripts\python scripts\stage5_gp.py
```

The error at each video moment, E_i = q_sensor(s_i)⁻¹ · q_video_i, is taken as a rotation vector in
the racket frame and unwrapped so it stays continuous through 180°. One GP per component
(scikit-learn, constant × RBF + white noise; impact moments get extra noise) is fitted over time,
and the predicted error is removed at every sample: q(s) = q_sensor(s) · exp(gp(s)). Fitted length
scales are 0.11–0.14 s, about four video frames.

| | mean | median | max |
|---|---|---|---|
| sensor only | 88.4° | 104.2° | 168.6° |
| GP, fitted on all moments | 7.5° | 4.8° | 33.4° |
| **GP, leave-one-out** | **12.4°** | **8.3°** | 44.1° (impact frame) |

GP 1σ per sample: median 16.9°, max 22°, saved as `gp_std_deg`. Outputs: `out/orientation.csv`,
`stage5_gp.png`, `stage5_gp.json`, `stroke_3d_gp.mp4`.

## Stage 6 – overlays and videos

```
.venv\Scripts\python scripts\stage6_overlay.py
.venv\Scripts\python scripts\make_demo.py
.venv\Scripts\python scripts\make_gp_video.py
```

The reconstructed racket is projected into every IMU-window frame of both videos using the stage-3
cameras and racket positions (the IMU gives orientation only). `overlay_*.mp4` and
`overlay_contact_sheet.png` show the GP-corrected racket with the leave-one-out error in the
captions; `demo.mp4` puts front, rear and corrected racket side by side; `gp_correction.mp4` draws
the sensor-only (orange) and corrected (green) rackets together with a live error chart.
