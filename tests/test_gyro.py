"""Synthetic-rotation tests for the quaternion helpers and gyro integration."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racket import gyro, quat

FS = 416.0


def to_scipy(q):
    return Rotation.from_quat(np.roll(q, -1, axis=-1))  # wxyz -> xyzw


def deg(x):
    return np.rad2deg(x)


# --- quaternion helpers ------------------------------------------------------

def test_rotvec_roundtrip_and_matrix_match_scipy():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    q = quat.from_rotvec(v)
    assert np.allclose(to_scipy(q).as_matrix(), quat.to_matrix(q), atol=1e-12)
    assert np.allclose(quat.from_rotvec(quat.to_rotvec(q)), quat.canonical(q), atol=1e-12)


def test_mul_order_matches_scipy_composition():
    rng = np.random.default_rng(1)
    a, b = quat.from_rotvec(rng.normal(size=3)), quat.from_rotvec(rng.normal(size=3))
    assert np.allclose(quat.to_matrix(quat.mul(a, b)), (to_scipy(a) * to_scipy(b)).as_matrix())


def test_slerp_endpoints_and_midpoint():
    a = quat.IDENTITY
    b = quat.from_rotvec([0, 0, np.pi / 2])
    assert np.allclose(quat.slerp(a, b, 0.0), a)
    assert np.allclose(quat.slerp(a, b, 1.0), b)
    assert np.isclose(deg(quat.angle_between(a, quat.slerp(a, b, 0.5))), 45.0)


# --- integration -------------------------------------------------------------

def test_zero_rate_keeps_start():
    q0 = quat.from_rotvec([0.3, -0.2, 1.0])
    q = gyro.integrate(np.zeros((100, 3)), q0=q0, fs=FS)
    assert np.allclose(q, quat.normalize(q0))


def test_constant_rate_single_axis_is_exact():
    n, rate = 417, 360.0                       # 1 s at 360 deg/s about z
    g = np.tile([0.0, 0.0, rate], (n, 1))
    q = gyro.integrate(g, fs=FS)
    expected = quat.from_rotvec([0, 0, np.deg2rad(rate * (n - 1) / FS)])
    assert deg(quat.angle_between(q[-1], expected)) < 1e-9


def test_bias_cancels_constant_rate():
    g = np.tile([10.0, -20.0, 5.0], (200, 1))
    q = gyro.integrate(g, bias_dps=[10.0, -20.0, 5.0], fs=FS)
    assert np.allclose(q, quat.IDENTITY)


@pytest.mark.parametrize("sign", [1, -1])
def test_sign_flips_direction(sign):
    g = np.tile([0.0, 90.0, 0.0], (int(FS) + 1, 1))   # +90 deg about y over 1 s
    q = gyro.integrate(g, sign=sign, fs=FS)
    expected = quat.from_rotvec([0, sign * np.deg2rad(90.0), 0])
    assert deg(quat.angle_between(q[-1], expected)) < 1e-9


def test_matches_reference_product_of_exponentials():
    """Order of composition: q_k = q0 * exp(w_mean_0 dt) * ... (body frame)."""
    rng = np.random.default_rng(2)
    g = rng.normal(scale=500.0, size=(60, 3))
    q0 = quat.from_rotvec([0.1, 0.2, 0.3])
    w = np.deg2rad(g)
    ref = to_scipy(q0)
    for k in range(len(g) - 1):
        ref = ref * Rotation.from_rotvec(0.5 * (w[k] + w[k + 1]) / FS)
    q = gyro.integrate(g, q0=q0, fs=FS)
    assert deg(quat.angle_between(q[-1], np.roll(ref.as_quat(), 1))) < 1e-9


def analytic_swing(t, alpha=6.0, beta=9.0):
    """R(t) = Rz(alpha t) Ry(beta t): rotation axis moves in the body frame.

    Body rate: w_b = Ry(beta t)^T [0, 0, alpha] + [0, beta, 0].
    Rates of ~500 deg/s, similar to the swing.
    """
    R = Rotation.from_rotvec(np.c_[0 * t, 0 * t, alpha * t]) * Rotation.from_rotvec(np.c_[0 * t, beta * t, 0 * t])
    w_b = Rotation.from_rotvec(np.c_[0 * t, beta * t, 0 * t]).inv().apply([0, 0, alpha]) + [0, beta, 0]
    return R, w_b


@pytest.mark.parametrize("fs, tol_deg", [(416.0, 0.05), (832.0, 0.0125)])
def test_time_varying_axis_converges(fs, tol_deg):
    t = np.arange(int(fs) + 1) / fs                    # 1 s
    R, w_b = analytic_swing(t)
    q = gyro.integrate(np.rad2deg(w_b), fs=fs)
    err = deg(quat.angle_between(q, np.roll(R.as_quat(), 1, axis=-1)))
    assert err.max() < tol_deg


def test_integrated_series_is_continuous():
    """No q -> -q jumps even when the rotation passes 180 deg."""
    g = np.tile([0.0, 0.0, 720.0], (417, 1))          # 720 deg about z in 1 s
    q = gyro.integrate(g, fs=FS)
    assert np.all(np.sum(q[1:] * q[:-1], axis=1) > 0.99)


def test_sample_at_integer_and_fraction():
    g = np.tile([0.0, 0.0, 416.0], (11, 1))           # 1 deg per sample about z
    q = gyro.integrate(g, fs=FS)
    assert np.allclose(gyro.sample_at(q, 4.0), q[4])
    ang = deg(quat.to_rotvec(gyro.sample_at(q, 4.25))[2])
    assert np.isclose(ang, 4.25)
