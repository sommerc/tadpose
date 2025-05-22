import numpy as np
import scipy as sp
import pandas as pd


from skimage import measure
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from .utils import angles as angles_utils, angles_old, smooth_diff, smooth


def ego_speeds(tadpole, parts=None):
    if parts is None:
        parts = tadpole.bodyparts

    locs = tadpole.ego_locs(parts=tuple(parts))

    speeds = {}
    for pi, p in enumerate(parts):
        speeds[p] = smooth_diff(locs[:, pi, :])

    return pd.DataFrame(speeds)


def speeds(tadpole, parts=None, track_idx=0, fill_missing=False):
    if parts is None:
        parts = tadpole.bodyparts

    locs = tadpole.locs(
        parts=tuple(parts), track_idx=track_idx, fill_missing=fill_missing
    )

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


class SocialReceptiveField:
    def __init__(
        self,
        tad,
        focal_track_idx,
        srf_bins=None,
        srf_size=None,
        other_part=None,
    ):
        """
        Social Receptive Field (SRF) of a focal animal.
        The SRF is the receptive field of a focal animal, which is defined as the
        spatial distribution of the other animal's body part locations in the
        focal animal's egocentric coordinate system. The center of the resulting
        heatmap is the focal animal's centrlly aligned body part location.

        Parameters
        ----------
        tad : tadpose.Tadpole
            The Tadpole object containing the pose estimation data.
        focal_track_idx : int
            The index of the focal animal's track.
        srf_bins : tuple of int, optional
            The number of bins in the x and y dimensions for the SRF heatmap.
            If None, defaults to (srf_size[0] // 16, srf_size[1] // 16).
        srf_size : tuple of int, optional
            The size of the SRF heatmap x, and y (in pixels). If None, defaults to the
            shape of the image in the Tadpole object.
        other_part : str, optional
            The name of the body part of the other animal to compute the SRF for.
            If None, defaults to the alinged part of the focal mouce.

        Methods
        -------
        compute(frames=None)
            Computes the SRF heatmap for the specified frames as [start, end].
            If frames is None, computes the SRF for all frames in the Tadpole object.

        plot(frame=0, heatmap_sigma=0)
            Plots the SRF heatmap and an example image of the focal animal at the
            specified frame. The heatmap is smoothed using a Gaussian filter with
            the specified sigma value. The focal animal's
        """

        self.tad = tad
        self.focal_track_idx = focal_track_idx

        self.other_part = other_part

        self.central_part, self.up_part = self.tad.aligner.bodyparts_to_align

        if other_part is None:
            self.other_part = self.up_part
        else:
            self.other_part = other_part

        self.other_track_ids = set(range(len(self.tad)))
        self.other_track_ids.remove(self.focal_track_idx)

        if srf_size is None:
            self.srf_size = self.tad.image(0).shape[::1]
        else:
            self.srf_size = srf_size

        if srf_bins is None:
            self.srf_bins = (self.srf_size[0] // 16, self.srf_size[1] // 16)
        else:
            self.srf_bins = srf_bins

        self.rotation_matrices = np.stack(
            self.tad.aligner.transformations[self.focal_track_idx]
        )

    def compute(self, frames=None):
        frames = self.tad.check_frames(frames)
        print(frames)

        result_heatmaps = []
        for other_tid in self.other_track_ids:
            other_locs = self.tad.locs(
                parts=(self.other_part,), track_idx=other_tid
            ).squeeze()

            vecs = np.c_[other_locs, np.ones(other_locs.shape[0])]
            vecs_ego_centric = np.einsum("fij,fj->fi", self.rotation_matrices, vecs)[
                :, :2
            ][frames]

            print(vecs_ego_centric)

            srf_heatmap_T, self.x_edges, self.y_edges = np.histogram2d(
                *vecs_ego_centric.T,
                bins=self.srf_bins,
                range=[
                    [-self.srf_size[0] // 2, self.srf_size[0] // 2],
                    [-self.srf_size[1] // 2, self.srf_size[1] // 2],
                ],
            )

            result_heatmaps.append(srf_heatmap_T.T)

        self.srf_heatmaps = np.stack(result_heatmaps)
        return self.srf_heatmaps, self.y_edges, self.x_edges

    def plot(self, frame=0, heatmap_sigma=0):
        from matplotlib import pyplot as plt

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

        up_part_pos = self.tad.ego_locs(track_idx=0, parts=(self.up_part,))[frame, 0]

        ax1 = self.tad.ego_plot(
            frame,
            parts=None,
            track_idx=self.focal_track_idx,
            ax=ax1,
            dest_height=self.srf_size[1],
            dest_width=self.srf_size[0],
        )
        ax1.arrow(0, 0, *up_part_pos, color="red", width=2, head_width=5)
        ax1.set_title(f"Focal animal with body axis")

        # sef_masked = np.ma.masked_where(self.srf_heatmap == 0, self.srf_heatmap)

        heatmap = self.srf_heatmaps.mean(0)
        if heatmap_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=heatmap_sigma)

        ax2.imshow(
            heatmap,
            origin="lower",
            extent=[
                -self.srf_size[0] // 2,
                self.srf_size[0] // 2,
                -self.srf_size[1] // 2,
                self.srf_size[1] // 2,
            ],
            aspect="equal",
            cmap="viridis",
        )
        up_part_pos = self.tad.ego_locs(track_idx=0, parts=(self.up_part,))[frame, 0]
        ax2.arrow(0, 0, *up_part_pos, color="red", width=2, head_width=5)
        ax2.set_title(f"SRF heatmap of other's mouse {self.other_part}")
