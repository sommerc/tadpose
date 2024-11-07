import numpy as np
import scipy as sp
import pandas as pd


from skimage import measure
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

from .utils import angles as angles_utils, angles_old, smooth_diff, smooth


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


### Got rid of *= -1 by negating angle later on
def angles_depr(tad, part_tuple1, part_tuple2, win=None, track_idx=0, frames=None):
    frames = tad._check_frames(frames)
    elocs = tad.locs(track_idx=track_idx)[frames].copy()

    elocs[..., 0] *= -1

    parts1_idx = [tad.bodyparts.index(p) for p in part_tuple1]
    parts2_idx = [tad.bodyparts.index(p) for p in part_tuple2]

    parts1 = elocs[:, parts1_idx, :]
    parts2 = elocs[:, parts2_idx, :]

    if win is not None:
        print("WARNING: angles(): 'win' parameter depricated")
        for p in range(parts1.shape[1]):
            parts1[:, p, :] = smooth(parts1[:, p, :], win=win)
            parts2[:, p, :] = smooth(parts2[:, p, :], win=win)

    vec1 = np.diff(parts1, axis=1).squeeze()
    vec2 = np.diff(parts2, axis=1).squeeze()

    return angles_old(vec1, vec2)


def angles(tad, part_tuple1, part_tuple2=None, track_idx=0, frames=None):
    frames = tad._check_frames(frames)
    locs = tad.locs(track_idx=track_idx)[frames]

    parts1_idx = [tad.bodyparts.index(p) for p in part_tuple1]
    parts1 = locs[:, parts1_idx, :]
    vec1 = np.diff(parts1, axis=1).squeeze()

    if part_tuple2 is not None:
        parts2_idx = [tad.bodyparts.index(p) for p in part_tuple2]
        parts2 = locs[:, parts2_idx, :]
        vec2 = np.diff(parts2, axis=1).squeeze()
    else:
        vec2 = np.zeros_like(vec1)
        vec2[:, 1] = 1

    return angles_utils(vec1, vec2)


def angles_diff(tad, part_tuple1, track_idx=0, frames=None):
    frames = tad._check_frames(frames)
    locs = tad.locs(track_idx=track_idx)[frames]

    parts1_idx = [tad.bodyparts.index(p) for p in part_tuple1]

    parts1 = locs[:, parts1_idx, :]

    vec1 = np.diff(parts1, axis=1).squeeze()

    return angles_utils(vec1[:-1], vec1[1:])


# Same as anlges diff
def angular_velocity(tad, part1, part2, track_idx=0, frames=None, in_degree=True):
    frames = tad._check_frames(frames)
    locs = tad.locs(track_idx=track_idx)[frames]

    part1_idx = tad.bodyparts.index(part1)
    part2_idx = tad.bodyparts.index(part2)

    part1 = locs[:, part1_idx, :]
    part2 = locs[:, part2_idx, :]

    vec1 = part1 - part2

    return angles_utils(vec1[:-1], vec1[1:], in_degree=in_degree)


def episodes_iter(criteria, min_len=0, max_len=np.inf):
    label_t = measure.label(criteria)[:, None]
    for rp in measure.regionprops(label_t):
        start = rp.slice[0].start
        stop = rp.slice[0].stop
        if max_len > (stop - start) > min_len:
            yield start, stop


class ReparametrizedSplineFit:
    def __init__(self, points, n_interpolants=64, spline_smooth=0):
        self.points = points
        self.n_interpolants = n_interpolants

        dist_points = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arclen_cum = np.concatenate(([0], dist_points.cumsum()))

        spline_init, u = sp.interpolate.splprep(
            points.T,
            u=arclen_cum,
            s=spline_smooth,
        )

        eval_points = np.linspace(0, arclen_cum[-1], n_interpolants)
        points_eval = sp.interpolate.splev(eval_points, spline_init)
        points_eval = np.stack(points_eval, -1)

        dist_points_eval = np.linalg.norm(np.diff(points_eval, axis=0), axis=1)
        arclen = np.concatenate(([0], dist_points_eval.cumsum()))
        arclen /= arclen[-1]

        self.arclen = np.linspace(0, 1, n_interpolants)
        self.spline, u = sp.interpolate.splprep(points_eval.T, u=arclen, s=0)

    def singed_curvature(self):
        xp, yp = sp.interpolate.splev(self.arclen, self.spline, der=1)
        xpp, ypp = sp.interpolate.splev(self.arclen, self.spline, der=2)

        mixed_term = xp * ypp - yp * xpp
        norm_term = (xp**2 + yp**2) ** (3 / 2)
        K = mixed_term / norm_term

        return K

    def interpolate(self):
        return np.stack(sp.interpolate.splev(self.arclen, self.spline, der=0), axis=1)


from csaps import CubicSmoothingSpline


class ReparametrizedCSAPSSplineFit:
    def __init__(self, points, n_interpolants=64, spline_smooth=1):
        self.points = points
        self.n_interpolants = n_interpolants

        arclen_cum = np.concatenate(
            ([0], np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum())
        )

        spline_init = CubicSmoothingSpline(
            arclen_cum, points.T, smooth=spline_smooth, normalizedsmooth=True
        )

        eval_points = np.linspace(0, arclen_cum[-1], n_interpolants)
        points_eval = spline_init(eval_points).T

        dist_points_eval = np.linalg.norm(np.diff(points_eval, axis=0), axis=1)
        arclen = np.concatenate(([0], dist_points_eval.cumsum()))
        arclen /= arclen[-1]

        self.spline = CubicSmoothingSpline(arclen, points_eval.T, smooth=1)

        self.arclen = np.linspace(0, 1, n_interpolants)

    def singed_curvature(self):
        xp, yp = self.spline(self.arclen, nu=1)
        xpp, ypp = self.spline(self.arclen, nu=2)

        mixed_term = xp * ypp - yp * xpp
        norm_term = (xp**2 + yp**2) ** (3 / 2)
        K = mixed_term / norm_term

        return K

    def interpolate(self):
        return self.spline(self.arclen).T
