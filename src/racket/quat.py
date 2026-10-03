"""Minimal vectorised unit-quaternion helpers.

Convention: q = [w, x, y, z], Hamilton product, and q rotates a vector from
the body (sensor) frame to the world frame: v_world = q * v_body * q^-1.
All functions accept (..., 4) / (..., 3) arrays.
"""
import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def normalize(q):
    """Unit-normalise, keeping the sign (so integrated series stay continuous)."""
    q = np.asarray(q, float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def canonical(q):
    """Unit-normalise and flip to w >= 0 (q and -q are the same rotation)."""
    q = normalize(q)
    return np.where(q[..., :1] < 0, -q, q)


def make_continuous(q):
    """Flip signs along a (N, 4) series so consecutive quaternions agree."""
    q = normalize(q).copy()
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    return q


def mul(a, b):
    """Hamilton product a * b (apply b first, then a, for frame rotations)."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, float), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def conj(q):
    """Inverse of a unit quaternion."""
    return np.asarray(q, float) * np.array([1.0, -1.0, -1.0, -1.0])


def from_rotvec(v):
    """Exact exponential map: rotation vector (rad) -> unit quaternion."""
    v = np.asarray(v, float)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * th
    # sin(th/2)/th, with the Taylor limit 1/2 - th^2/48 near zero
    k = np.where(th > 1e-8, np.sin(half) / np.where(th > 1e-8, th, 1.0), 0.5 - th ** 2 / 48.0)
    return np.concatenate([np.cos(half), k * v], axis=-1)


def to_rotvec(q):
    """Logarithm map: unit quaternion -> rotation vector (rad), angle in [0, pi]."""
    q = canonical(q)
    w = np.clip(q[..., :1], -1.0, 1.0)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    th = 2.0 * np.arctan2(s, w)
    k = np.where(s > 1e-8, th / np.where(s > 1e-8, s, 1.0), 2.0)
    return k * q[..., 1:]


def to_matrix(q):
    """Unit quaternion -> 3x3 rotation matrix (body -> world)."""
    w, x, y, z = np.moveaxis(normalize(q), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def from_matrix(R):
    """3x3 rotation matrix -> unit quaternion (via scipy for robustness)."""
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(R).as_quat()
    return canonical(np.concatenate([xyzw[..., 3:], xyzw[..., :3]], axis=-1))


def rotate(q, v):
    """Rotate body-frame vector(s) v into the world frame."""
    return np.einsum("...ij,...j->...i", to_matrix(q), v)


def angle_between(a, b):
    """Geodesic angle (rad) between two orientations."""
    d = np.abs(np.sum(normalize(a) * normalize(b), axis=-1))
    return 2.0 * np.arccos(np.clip(d, 0.0, 1.0))


def slerp(a, b, u):
    """Spherical interpolation from a (u=0) to b (u=1), shortest path."""
    a, b = normalize(a), normalize(b)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    rel = mul(conj(a), b)
    return normalize(mul(a, from_rotvec(np.asarray(u, float)[..., None] * to_rotvec(rel))))
