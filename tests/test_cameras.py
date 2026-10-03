"""Synthetic recovery test for the two-camera bundle adjustment."""
import numpy as np
from scipy.spatial.transform import Rotation

from racket import cameras, quat


def make_scene(m=10, seed=0, noise_px=0.0):
    """Racket swinging ~7 m in front of camera 1; camera 2 at 87 deg to the side."""
    rng = np.random.default_rng(seed)
    model = cameras.model_points()
    f = np.array([2200.0, 1600.0])
    R21 = Rotation.from_euler("y", -87, degrees=True)
    C2 = np.array([-7.3, -2.1, 6.7])
    t21 = -R21.apply(C2)
    R = Rotation.from_euler("zyx", np.c_[np.linspace(-60, 90, m), np.linspace(10, 70, m),
                                         rng.uniform(-40, 40, m)], degrees=True)
    t = np.c_[rng.uniform(-0.5, 0.5, m), rng.uniform(-0.3, 0.3, m), rng.uniform(6.8, 7.1, m)]
    Pc1 = np.einsum("mij,pj->mpi", R.as_matrix(), model) + t[:, None]
    Pc2 = Pc1 @ R21.as_matrix().T + t21
    obs = np.stack([cameras.project(Pc1, f[0]), cameras.project(Pc2, f[1])])
    obs = obs + rng.normal(scale=noise_px, size=obs.shape)
    return obs, f, R.as_matrix(), R21.as_matrix()


def test_recovers_noise_free_scene():
    obs, f, R, R21 = make_scene()
    res = cameras.solve(obs, 1400.0, refine_geometry=False)
    assert np.allclose(res.f, f, rtol=1e-3)
    err = np.rad2deg(quat.angle_between(quat.from_matrix(res.R), quat.from_matrix(R)))
    assert err.max() < 0.05
    assert np.rad2deg(quat.angle_between(quat.from_matrix(res.R21), quat.from_matrix(R21))) < 0.05
    assert res.rms_px.max() < 1e-3


def test_orientation_error_scales_with_click_noise():
    obs, f, R, _ = make_scene(noise_px=3.0, seed=1)
    res = cameras.solve(obs, 1400.0, refine_geometry=False)
    err = np.rad2deg(quat.angle_between(quat.from_matrix(res.R), quat.from_matrix(R)))
    assert np.median(err) < 6.0           # roll about the handle dominates (~5 deg at 3 px)
