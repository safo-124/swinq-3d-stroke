"""Two-camera bundle adjustment using the racket as the calibration object.

World frame = camera 1 (front). Each camera: pinhole, square pixels,
principal point at the image centre, no distortion, unknown focal length f.
A racket pose (R_i, t_i) maps sensor-frame points into camera 1:
    X_c1 = R_i P + t_i,     X_c2 = R_21 X_c1 + t_21.

Parameter vector:
    [log f1, log f2, rotvec_21 (3), t_21 (3), geometry (0 or 3), poses (6 per moment)]
Each pose is [rotvec (3), a, b, s] with t = z [a, b, 1] and z = f1 exp(-s): (a, b) is
the throat's ray in camera 1 and s its log image scale. Far away the image fixes
f1/z but barely f1 itself; this layout makes that valley a single coordinate.
Geometry (optional) = [throat fraction s, edge x, edge y]; butt = +s L, tip = -(1-s) L
on the sensor X axis, with the butt-tip length L fixed so the scale is metric.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from . import config, quat

N_CAM = 8      # log f1, log f2, rotvec_21 (3), t_21 (3)


@dataclass
class BAResult:
    f: np.ndarray              # (2,) focal lengths, px
    R21: np.ndarray            # (3, 3) camera 1 -> camera 2 rotation
    t21: np.ndarray            # (3,) camera 1 -> camera 2 translation, m
    R: np.ndarray              # (M, 3, 3) racket -> camera 1 rotations
    t: np.ndarray              # (M, 3) racket origin (throat) in camera 1, m
    model: np.ndarray          # (4, 3) racket points used, sensor frame
    rms_px: np.ndarray         # (M, 2) per-moment RMS reprojection error per view
    err_px: np.ndarray         # (2, M, 4) per-point reprojection error
    rot_sigma_deg: np.ndarray  # (M, 3) 1-sigma about body X (roll), Y, Z, deg
    cost: float
    x: np.ndarray


# --- geometry ----------------------------------------------------------------

def model_points(geom=None):
    """(4, 3) sensor-frame points in config.POINT_NAMES order."""
    geom = geom or config.RACKET_POINTS
    return np.array([geom[n] for n in config.POINT_NAMES], float)


def geom_to_params(model):
    """(4, 3) model -> [throat fraction, edge x, edge y]."""
    names = list(config.POINT_NAMES)
    butt, edge = model[names.index("butt")], model[names.index("edge")]
    return np.array([butt[0] / config.RACKET_LENGTH_M, edge[0], edge[1]])


def params_to_geom(g):
    """[throat fraction, edge x, edge y] -> (4, 3) model, butt-tip length fixed."""
    s, ex, ey = g
    L = config.RACKET_LENGTH_M
    pts = {"butt": [s * L, 0, 0], "throat": [0, 0, 0], "tip": [-(1 - s) * L, 0, 0], "edge": [ex, ey, 0]}
    return np.array([pts[n] for n in config.POINT_NAMES], float)


# --- camera model ------------------------------------------------------------

def image_centre():
    w, h = config.IMAGE_SIZE
    return np.array([w / 2.0, h / 2.0])


def project(P_cam, f):
    """Pinhole projection of camera-frame points (..., 3) -> pixels (..., 2)."""
    return f * P_cam[..., :2] / P_cam[..., 2:3] + image_centre()


def pose_translation(abs_, f1):
    """[a, b, s] (..., 3) -> metric translation in camera 1."""
    z = f1 * np.exp(-abs_[..., 2:3])
    return z * np.concatenate([abs_[..., :2], np.ones_like(z)], axis=-1)


def translation_params(t, f1):
    """Metric translation (..., 3) -> [a, b, s]."""
    return np.concatenate([t[..., :2] / t[..., 2:3], np.log(f1 / t[..., 2:3])], axis=-1)


class Problem:
    """Observations plus the parameter layout for one bundle adjustment."""

    def __init__(self, obs, model, refine_geometry, weight=None):
        self.obs = obs                                   # (2, M, 4, 2)
        self.m = obs.shape[1]
        self.model0 = model
        self.refine = refine_geometry
        self.n_glob = N_CAM + (3 if refine_geometry else 0)
        self.weight = np.ones(obs.shape[:3]) if weight is None else weight

    def unpack(self, x):
        """Split x into f, R21, t21, model, R (M,3,3), t (M,3)."""
        f = np.exp(x[:2])
        R21 = Rotation.from_rotvec(x[2:5]).as_matrix()
        model = params_to_geom(x[N_CAM:N_CAM + 3]) if self.refine else self.model0
        poses = x[self.n_glob:].reshape(self.m, 6)
        return f, R21, x[5:8], model, Rotation.from_rotvec(poses[:, :3]).as_matrix(),             pose_translation(poses[:, 3:], f[0])

    def pack(self, f, rv21, t21, model, rotvecs, t):
        """Inverse of unpack; rotvecs (M, 3), t (M, 3) metric translations."""
        geom = geom_to_params(model) if self.refine else []
        poses = np.concatenate([rotvecs, translation_params(t, f[0])], axis=1)
        return np.concatenate([np.log(f), rv21, t21, geom, np.ravel(poses)])

    def predict(self, x):
        """Predicted pixels, (2 views, M, 4 points, 2)."""
        f, R21, t21, model, R, t = self.unpack(x)
        Pc1 = np.einsum("mij,pj->mpi", R, model) + t[:, None, :]
        Pc2 = Pc1 @ R21.T + t21
        return np.stack([project(Pc1, f[0]), project(Pc2, f[1])])

    def residuals(self, x):
        return ((self.predict(x) - self.obs) * self.weight[..., None]).ravel()

    def jacobian(self, x):
        """Dense forward-difference Jacobian, using the sparsity to group columns.

        The problem is small (~460 x 190), so a dense exact trust-region step is
        much faster than scipy's sparse/LSMR path.
        """
        if not hasattr(self, "_sparsity"):
            self._sparsity = self.jac_sparsity().tocsr()
        return approx_derivative(self.residuals, x, method="2-point",
                                 sparsity=self._sparsity).toarray()

    def jac_sparsity(self):
        """Each residual depends on the globals and its own moment's 6 pose params."""
        S = lil_matrix((2 * self.m * 8, self.n_glob + 6 * self.m), dtype=int)
        for v in range(2):
            for i in range(self.m):
                r0 = (v * self.m + i) * 8
                S[r0:r0 + 8, :self.n_glob] = 1
                k = self.n_glob + 6 * i
                S[r0:r0 + 8, k:k + 6] = 1
        return S


# --- initialisation ------------------------------------------------------------

def _initial_translation(model, uv, f):
    """Depth from the butt-tip length in pixels; x, y from the image centroid ray."""
    names = list(config.POINT_NAMES)
    L_m = np.linalg.norm(model[names.index("butt")] - model[names.index("tip")])
    L_px = np.linalg.norm(uv[names.index("butt")] - uv[names.index("tip")])
    z = f * L_m / max(L_px, 1.0)
    ray = (uv.mean(0) - image_centre()) / f
    return np.array([ray[0] * z, ray[1] * z, z])


def _pose_residuals(p, model, uv, f):
    """Batched single-view residuals: p (B, 6) -> (B, 8) pixels."""
    R = quat.to_matrix(quat.from_rotvec(p[:, :3]))
    P = np.einsum("bij,pj->bpi", R, model) + p[:, None, 3:]
    return (f[:, None, None] * P[..., :2] / P[..., 2:3] + image_centre() - uv).reshape(len(p), -1)


def batch_lm(fun, x0, n_iter=150, eps=1e-6):
    """Levenberg-Marquardt on many small independent problems at once.

    fun: (B, n) -> (B, r). Forward-difference Jacobian. Returns (x, residuals).
    """
    x = x0.copy()
    r = fun(x)
    cost = np.sum(r ** 2, axis=1)
    lam = np.full(len(x), 1e-3)
    eye = np.eye(x.shape[1])
    for _ in range(n_iter):
        J = np.stack([(fun(x + eps * eye[k]) - r) / eps for k in range(x.shape[1])], axis=-1)
        JtJ = np.einsum("bri,brj->bij", J, J)
        A = JtJ + lam[:, None, None] * (JtJ * eye + 1e-9 * eye)
        dx = -np.linalg.solve(A, np.einsum("bri,br->bi", J, r)[..., None])[..., 0]
        rn = fun(x + dx)
        cn = np.sum(rn ** 2, axis=1)
        ok = np.isfinite(cn) & (cn < cost)
        x[ok], r[ok], cost[ok] = x[ok] + dx[ok], rn[ok], cn[ok]
        lam = np.where(ok, lam / 3.0, lam * 4.0)
    return x, r


def single_view_poses_batch(model, uv, f, max_rms=config.PNP_MAX_RMS_PX, dedupe_deg=5.0):
    """Distinct good poses for many views at once.

    uv: (V, 4, 2), f: (V,). Returns, per view, a list of (R, t) sorted by fit.
    Homography/IPPE PnP is degenerate here (butt, throat, tip are collinear),
    so each view is multi-started from the 24 octahedral rotations.
    """
    starts = Rotation.create_group("O").as_rotvec()                   # (S, 3)
    V, S = len(uv), len(starts)
    t0 = np.array([_initial_translation(model, uv[v], f[v]) for v in range(V)])
    x0 = np.concatenate([np.tile(starts, (V, 1)), np.repeat(t0, S, axis=0)], axis=1)
    uv_b, f_b = np.repeat(uv, S, axis=0), np.repeat(f, S)
    x, r = batch_lm(lambda p: _pose_residuals(p, model, uv_b, f_b), x0)
    rms = np.sqrt(np.mean(r ** 2, axis=1) * 2).reshape(V, S)
    x = x.reshape(V, S, 6)
    out = []
    for v in range(V):
        sols = []
        for s in np.argsort(rms[v]):
            R = quat.to_matrix(quat.from_rotvec(x[v, s, :3]))
            t = x[v, s, 3:]
            if rms[v, s] > max_rms or np.any((model @ R.T)[:, 2] + t[2] <= 0):
                continue
            if all(np.rad2deg(quat.angle_between(quat.from_matrix(R), quat.from_matrix(Rk))) > dedupe_deg
                   for Rk, _ in sols):
                sols.append((R, t))
        out.append(sols)
    return out


def single_view_poses(model, uv, f, **kw):
    """Distinct good poses for one view: list of (R, t), racket -> camera."""
    return single_view_poses_batch(model, uv[None], np.array([float(f)]), **kw)[0]


def relative_pose_candidates(model, obs, f):
    """Per moment: list of (R21, t21, R1, t1) for every pairing of view poses."""
    m = obs.shape[1]
    sv = single_view_poses_batch(model, obs.reshape(2 * m, 4, 2), np.repeat(f, m))
    out = []
    for i in range(m):
        c = []
        for R1, t1 in sv[i]:
            for R2, t2 in sv[m + i]:
                R21 = R2 @ R1.T
                c.append((R21, t2 - R21 @ t1, R1, t1))
        out.append(c)
    return out


def consensus_init(prob, f, inlier_deg=config.BA_INIT_INLIER_DEG):
    """Initial x from the camera-2 relative rotation that most moments agree on.

    Returns (x0, n_inlier_moments).
    """
    cands = relative_pose_candidates(prob.model0, prob.obs, f)
    flat = [c for cs in cands for c in cs]
    if not flat:
        raise RuntimeError("no single-view pose fits")
    rots = Rotation.from_matrix(np.array([c[0] for c in flat]))
    owner = np.repeat(np.arange(len(cands)), [len(cs) for cs in cands])
    best, best_n = 0, -1
    for k in range(len(flat)):
        close = (rots[k].inv() * rots).magnitude() < np.deg2rad(inlier_deg)
        n = len(np.unique(owner[close]))            # count moments, not candidates
        if n > best_n:
            best, best_n = k, n
    ref = rots[best]
    picks = []
    for cs in cands:
        d = [(ref.inv() * Rotation.from_matrix(c[0])).magnitude() for c in cs]
        picks.append(cs[int(np.argmin(d))] if cs and min(d) < np.deg2rad(inlier_deg) else None)
    good = [p for p in picks if p is not None]
    R21 = Rotation.from_matrix(np.array([p[0] for p in good])).mean()
    t21 = np.median(np.array([p[1] for p in good]), axis=0)
    poses = _fill_missing(prob, picks, f)
    return prob.pack(f, R21.as_rotvec(), t21, prob.model0, poses[:, :3], poses[:, 3:]), best_n


def _fill_missing(prob, picks, f):
    """Pose per moment as (rotvec, t); moments with no agreeing pair use their
    best camera-1 single-view pose, or the nearest moment's pose."""
    out = []
    for i, p in enumerate(picks):
        if p is None:
            sv = single_view_poses(prob.model0, prob.obs[0, i], f[0])
            p = (None, None, *sv[0]) if sv else None
        out.append(p)
    idx = [i for i, p in enumerate(out) if p is not None]
    return np.array([np.r_[Rotation.from_matrix(out[j][2]).as_rotvec(), out[j][3]]
                     for j in (i if out[i] is not None else min(idx, key=lambda k: abs(k - i))
                               for i in range(len(out)))])


# --- solve -------------------------------------------------------------------

def _robust_cost(r, f_scale):
    """scipy's soft_l1 cost for a residual vector."""
    z = (r / f_scale) ** 2
    return f_scale ** 2 * np.sum(np.sqrt(1 + z) - 1)


def fix_flips(prob, x, f_scale=config.BA_F_SCALE_PX):
    """Re-seat each moment in its best mirror-ambiguity basin, cameras held fixed.

    A racket seen nearly edge-on has two planar pose solutions per view; the
    joint fit can settle in the wrong one for an isolated moment. For each
    moment, every single-view pose (either camera, mapped to camera 1) seeds a
    6-DoF fit of that moment alone; the lowest robust cost wins.
    Returns (x, list of moments that changed).
    """
    f, R21, t21, model, R, t = prob.unpack(x)
    poses = x[prob.n_glob:].reshape(prob.m, 6).copy()
    changed = []
    for i in range(prob.m):
        def res_i(p):
            Pc1 = Rotation.from_rotvec(p[:3]).apply(model) + pose_translation(p[3:], f[0])
            pred = np.stack([project(Pc1, f[0]), project(Pc1 @ R21.T + t21, f[1])])
            return ((pred - prob.obs[:, i]) * prob.weight[:, i, :, None]).ravel()
        seeds = [poses[i]]
        for R1, t1 in single_view_poses(model, prob.obs[0, i], f[0]):
            seeds.append(np.r_[Rotation.from_matrix(R1).as_rotvec(), translation_params(t1, f[0])])
        for R2, t2 in single_view_poses(model, prob.obs[1, i], f[1]):
            R1, t1 = R21.T @ R2, R21.T @ (t2 - t21)
            seeds.append(np.r_[Rotation.from_matrix(R1).as_rotvec(), translation_params(t1, f[0])])
        fits = [least_squares(res_i, s0, loss="soft_l1", f_scale=f_scale) for s0 in seeds]
        best = min(fits, key=lambda s: _robust_cost(s.fun, f_scale))
        if _robust_cost(best.fun, f_scale) < _robust_cost(res_i(poses[i]), f_scale) - 1e-6:
            if np.rad2deg((Rotation.from_rotvec(best.x[:3]).inv()
                           * Rotation.from_rotvec(poses[i, :3])).magnitude()) > 20:
                changed.append(i)
            poses[i] = best.x
    return np.r_[x[:prob.n_glob], poses.ravel()], changed


def solve(obs, f_init, model=None, refine_geometry=config.BA_REFINE_GEOMETRY,
          weight=None, f_scale=config.BA_F_SCALE_PX, max_flip_rounds=3):
    """Bundle adjustment from one focal-length guess. obs: (2, M, 4, 2) px.

    Alternates the joint fit with per-moment mirror-flip repair until stable.
    """
    prob = Problem(obs, model_points() if model is None else model, refine_geometry, weight)
    x, n_in = consensus_init(prob, np.broadcast_to(f_init, 2).astype(float))
    flipped = []
    for _ in range(max_flip_rounds + 1):
        sol = least_squares(prob.residuals, x, jac=prob.jacobian, tr_solver="exact",
                            loss="soft_l1", f_scale=f_scale, x_scale="jac", max_nfev=3000)
        x, changed = fix_flips(prob, sol.x, f_scale)
        if not changed:
            break
        flipped += changed
    res = _result(prob, sol)
    res.n_init_inliers = n_in
    res.flipped = flipped
    return res


def _result(prob, sol):
    """Package a least_squares solution with per-moment diagnostics."""
    f, R21, t21, model, R, t = prob.unpack(sol.x)
    err = np.linalg.norm(prob.predict(sol.x) - prob.obs, axis=-1)          # (2, M, 4)
    rms = np.sqrt(np.mean(err ** 2, axis=2)).T                             # (M, 2)
    return BAResult(f=f, R21=R21, t21=t21, R=R, t=t, model=model, rms_px=rms, err_px=err,
                    rot_sigma_deg=_rotation_sigma(prob, sol), cost=float(sol.cost), x=sol.x)


def _rotation_sigma(prob, sol, eps=1e-6):
    """Per-moment 1-sigma orientation uncertainty about the body X, Y, Z axes (deg).

    Conditional on the camera parameters: each moment's 8+8 pixel residuals are
    differentiated w.r.t. a local body-frame rotation R exp([d]x) and its own
    3 translation params. Noise level = robust (MAD) residual sigma, so a few
    bad clicks do not inflate every moment. Local perturbations avoid the
    rotation-vector singularity near 180 deg.
    """
    f, R21, t21, model, R, t = prob.unpack(sol.x)
    r = prob.residuals(sol.x)
    noise = 1.4826 * np.median(np.abs(r))
    out = np.zeros((prob.m, 3))
    for i in range(prob.m):
        def res_i(p):
            Ri = R[i] @ Rotation.from_rotvec(p[:3]).as_matrix()
            Pc1 = model @ Ri.T + t[i] + p[3:]
            pred = np.stack([project(Pc1, f[0]), project(Pc1 @ R21.T + t21, f[1])])
            return ((pred - prob.obs[:, i]) * prob.weight[:, i, :, None]).ravel()
        r0 = res_i(np.zeros(6))
        J = np.stack([(res_i(eps * e) - r0) / eps for e in np.eye(6)], axis=1)
        cov = np.linalg.pinv(J.T @ J) * noise ** 2
        out[i] = np.sqrt(np.diag(cov)[:3])
    return np.rad2deg(out)


def solve_multistart(obs, f_grid=config.BA_F_GRID, verbose=True, **kw):
    """Run solve() from each focal guess; return (best result, [(f_init, result)])."""
    results = []
    for f in f_grid:
        try:
            results.append((f, solve(obs, f, **kw)))
        except RuntimeError as e:
            if verbose:
                print(f"  f_init={f:.0f}: failed ({e})")
    if not results:
        raise RuntimeError("bundle adjustment failed from every focal guess")
    return min(results, key=lambda r: r[1].cost)[1], results
