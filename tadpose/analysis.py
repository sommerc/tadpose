import numpy as np
import pandas as pd

from skimage import measure
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

from .utils import smooth_diff, smooth


def ego_speeds(tadpole, parts=None):
    if parts is None:
        parts = tadpole.bodyparts

    locs = tadpole.ego_locs(parts=tuple(parts))

    speeds = {}
    for pi, p in enumerate(parts):
        speeds[p] = smooth_diff(locs[:, pi, :])

    return pd.DataFrame(speeds)


def speeds(tadpole, parts=None):
    if parts is None:
        parts = tadpole.bodyparts

    locs = tadpole.locs(parts=tuple(parts))

    speeds = {}
    for pi, p in enumerate(parts):
        speeds[p] = smooth_diff(locs[:, pi, :])

    return pd.DataFrame(speeds, columns=parts)


def angles(tad, part_tuple1, part_tuple2, win=5, track_idx=0):
    elocs = tad.ego_locs(track_idx=track_idx).copy()

    elocs[..., 0] *= -1

    parts1_idx = [tad.bodyparts.index(p) for p in part_tuple1]
    parts2_idx = [tad.bodyparts.index(p) for p in part_tuple2]

    parts1 = elocs[:, parts1_idx, :]
    parts2 = elocs[:, parts2_idx, :]

    if win is not None:
        for p in range(parts1.shape[1]):
            parts1[:, p, :] = smooth(parts1[:, p, :], win=win)
            parts2[:, p, :] = smooth(parts2[:, p, :], win=win)

    vec1 = np.diff(parts1, axis=1).squeeze()
    vec2 = np.diff(parts2, axis=1).squeeze()

    vec1 = (vec1.T / np.linalg.norm(vec1, axis=1)).T
    vec2 = (vec2.T / np.linalg.norm(vec2, axis=1)).T

    ortho_vec1 = np.c_[-vec1[:, 1], vec1[:, 0]]
    sign = np.sign(np.sum(ortho_vec1 * vec2, axis=1))

    c = np.sum(vec1 * vec2, axis=1)
    angles = sign * np.rad2deg(np.arccos(np.clip(c, -1, 1)))
    return angles


def episodes_iter(criteria, min_len=0, max_len=np.Inf):
    label_t = measure.label(criteria)[:, None]
    for rp in measure.regionprops(label_t):
        start = rp.slice[0].start
        stop = rp.slice[0].stop
        if max_len > (stop - start) > min_len:
            yield start, stop


class ReparametrizedSplineFit:
    def __init__(self, points, n_interpolants=64):
        self.points = points
        self.n_interpolants = n_interpolants

        s_inter = np.linspace(0, len(points), n_interpolants)

        s_init = np.linspace(0, len(points), len(points))
        spl_points = interpolate.CubicSpline(s_init, points, bc_type="natural")

        points_inter = spl_points(s_inter)

        points_tangents = spl_points.derivative(1)(s_inter)

        arc_len = np.cumsum(np.linalg.norm(points_tangents, axis=1))
        self.points_arc_length = arc_len

        arc_len = arc_len / arc_len.max() * n_interpolants
        arc_len = np.r_[[0], arc_len[:-1]]

        self.s = np.linspace(0, n_interpolants, n_interpolants)
        self.spline = interpolate.CubicSpline(arc_len, points_inter, bc_type="natural")

    def singed_curvature(self, sigma=0):
        xp, yp = self.spline.derivative(1)(self.s).T
        xpp, ypp = self.spline.derivative(2)(self.s).T

        mixed_term = xp * ypp - yp * xpp
        norm_term = (xp ** 2 + yp ** 2) ** (3 / 2)
        K = mixed_term / norm_term
        if sigma > 0:
            K = gaussian_filter1d(K, sigma)

        return K


# def get_rotation_angle(Rs):
#     return np.rad2deg(np.arctan2(Rs[:, 1, 0], -Rs[:, 0, 0],))


# def get_singed_angular_speed(Rs):
#     return np.array(
#         list(
#             map(
#                 lambda a: a if abs(a) < 180 else 360 - abs(a),
#                 np.diff(get_rotation_angle(Rs)),
#             )
#         )
#     )

