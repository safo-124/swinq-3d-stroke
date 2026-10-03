import numpy as np

from racket import config
from racket.io import load_anchors, load_imu


def test_imu_shape_and_rate():
    imu = load_imu()
    assert imu.n == 400
    assert np.isclose(imu.t[1] - imu.t[0], 1 / config.FS_HZ)


def test_anchors_aligned_across_views():
    a = load_anchors()
    assert len(a["front"].frames) == len(a["rear"].frames) == 29
    assert np.all(a["rear"].frames - a["front"].frames == config.FRAME_OFFSET_REAR)
    assert np.allclose(a["front"].imu_sample, a["rear"].imu_sample)


def test_front_118_relabel():
    raw = load_anchors(fixes={})["front"]
    fixed = load_anchors()["front"]
    i = list(fixed.frames).index(118)
    names = list(config.POINT_NAMES)
    assert np.allclose(fixed.uv[i, names.index("butt")], raw.uv[i, names.index("edge")])
    # after the fix the handle is shorter than the head, as on every other frame
    bt = np.linalg.norm(fixed.uv[i, 0] - fixed.uv[i, 1])
    tt = np.linalg.norm(fixed.uv[i, 2] - fixed.uv[i, 1])
    assert bt < tt
