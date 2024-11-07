import numpy as np
import pandas as pd
from .analysis import angular_velocity

from collections import defaultdict

from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from matplotlib.patches import Rectangle

from skimage import measure as skm


class Stride:
    def __init__(
        self,
        mouse,
        paw_part,
        min_peak_prominence_px,
        sigma=0,
    ):
        self.mouse = mouse
        self.paw_part = paw_part
        self.min_peak_prominence = min_peak_prominence_px
        self.sigma = sigma

        self._stride_frames = None
        self.paw_signal = None
        self._valid_strides_bind = None

    def find_strides(self):

        self.paw_signal = self.mouse.ego_locs(
            parts=(self.paw_part,), fill_missing=True
        )[:, 0, 1]

        if self.sigma > 0:
            self.paw_signal = ndi.gaussian_filter1d(self.paw_signal, sigma=self.sigma)

        # hpr = hpr[38:]
        maxima, _ = find_peaks(self.paw_signal, prominence=self.min_peak_prominence)
        minima, _ = find_peaks(-self.paw_signal, prominence=self.min_peak_prominence)

        if maxima[0] > minima[0]:
            minima = minima[1:]

        if minima[-1] < maxima[-1]:
            maxima = maxima[:-1]

        assert len(maxima) == len(
            minima
        ), f"maxima vs minima len {len(maxima)} != {len(minima)}"

        self._stride_frames = np.zeros((len(maxima), 3), dtype=np.uint32)
        self._stride_frames[:, 0] = maxima  # stance start
        self._stride_frames[:, 1] = minima  # swing start
        self._stride_frames[:-1, 2] = maxima[1:]  # stride stop

        self._stride_frames = self._stride_frames[:-1]

    def plot(
        self,
        ax,
        slc=None,
        color_stance="gray",
        alpha_stance=0.1,
        color_swing="gray",
        alpha_swing=0.02,
    ):
        if slc is None:
            slc = slice(None)

        frames = np.arange(slc.start, slc.stop)

        sig_min = self.paw_signal[slc].min()
        sig_height = self.paw_signal[slc].max() - self.paw_signal.min()

        ax.plot(
            frames, self.paw_signal[slc], color=color_stance, label=f"{self.paw_part}"
        )

        # plot stance
        for stance_start, swing_start, stride_stop in self.stride_frames:
            if stance_start < slc.start or stride_stop > slc.stop:
                continue
            rect_stance = Rectangle(
                (stance_start, sig_min),
                swing_start - stance_start,
                sig_height,
                color=color_stance,
                alpha=alpha_stance,
            )
            ax.add_patch(rect_stance)

            rect_swing = Rectangle(
                (swing_start, sig_min),
                stride_stop - swing_start,
                sig_height,
                color=color_swing,
                alpha=alpha_swing,
            )
            ax.add_patch(rect_swing)

        ax.set_xlim(slc.start, slc.stop)
        ax.set_ylabel(f"Aligned\n{self.paw_part}")

    def validate_strides(self, other):
        bins = np.r_[self._stride_frames[:, 0], self._stride_frames[-1, 2]]
        rng = (self._stride_frames[0, 0], self._stride_frames[-1, 2])

        other_stance_start = other._stride_frames[:, 0]

        hist, edges = np.histogram(other_stance_start, bins=bins, range=rng)

        self._valid_strides_bind = hist == 1

    @property
    def stride_frames(self):
        if self._valid_strides_bind is None:
            print(
                f"WARNING: Strides from '{self.paw_part}' are not validated yet by symmetric paw strides. Using all strides..."
            )
            strides_bind = np.ones(len(self._stride_frames), dtype=bool)
        else:
            strides_bind = self._valid_strides_bind

        return self._stride_frames[strides_bind]

    def decribe(self):
        print(
            f"Strides for '{self.paw_part}'\n # Strides: {len(self._stride_frames)}\n # Strides (valid): {len(self.stride_frames)}"
        )

    def iter(self, start, stop):
        a = self.stride_frames[:, 0] > start
        b = self.stride_frames[:, 2] < stop
        for stance_start, swing_start, stride_stop in self.stride_frames[a & b]:

            yield stance_start, swing_start, stride_stop

    def strides_in_label(self, mask, min_duration):
        in_label_strides = np.zeros(len(self.stride_frames), dtype=np.uint8)
        for rp in skm.regionprops(skm.label(mask)[:, None]):
            if rp.area > min_duration:
                slc = rp.slice[0]
                in_label_strides += np.logical_and(
                    self.stride_frames[:, 0] >= slc.start,
                    self.stride_frames[:, 2] < slc.stop,
                )

        return self.stride_frames[in_label_strides == 1]


def project_point_on_line(line_p1, line_p2, pnt):

    # distance between p1 and p2
    line_dist = np.sum((line_p1 - line_p2) ** 2)
    if line_dist == 0:
        print("project_point_on_line(): line_p1 and line_p2 are the same points")

    # The line extending the segment is parameterized as p1 + t (p2 - p1).
    # The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

    # if you need the point to project on line extention connecting p1 and p2
    t = np.sum((pnt - line_p1) * (line_p2 - line_p1)) / line_dist

    # if you need to ignore if p3 does not project onto line segment
    # if t > 1 or t < 0:

    #     print(
    #         "project_point_on_line(): pnt does not project onto line_p1-line_p2 line segment"
    #     )

    # if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    # t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))

    pnt_projection = line_p1 + t * (line_p2 - line_p1)
    return pnt_projection


class StrideProperties:
    def __init__(self, mouse, mask, pixel_size, fps, min_duration=30):
        self.mouse = mouse
        self.mask = mask

        self.pixel_size = pixel_size
        self.fps = fps
        self.min_duration = min_duration

    def angular_velocity(self, strides, part_axis):
        av_seq = angular_velocity(self.mouse, part_axis[0], part_axis[1]) * self.fps

        res = []

        for stance_start, swing_start, stride_stop in strides.strides_in_label(
            self.mask, self.min_duration
        ):
            res.append(av_seq[stance_start:stride_stop].mean())

        return np.array(res)

    def stride_length(self, strides):
        locs = self.mouse.locs(parts=(strides.paw_part,)).squeeze()
        res = []

        for stance_start, swing_start, stride_stop in strides.strides_in_label(
            self.mask, self.min_duration
        ):
            dist = (
                np.linalg.norm(locs[stance_start] - locs[stride_stop]) * self.pixel_size
            )
            res.append(dist)

        return np.array(res)

    def stride_speed(self, strides, part_for_speed):
        part_spped = (
            self.mouse.speed(part=part_for_speed, sigma=0, pre_sigma=0)
            * self.pixel_size
            * self.fps
        )

        res = []

        for stance_start, swing_start, stride_stop in strides.strides_in_label(
            self.mask, self.min_duration
        ):
            res.append(part_spped[stance_start:stride_stop].mean())

        return np.array(res)

    def duty_factor(self, strides):
        stride_frames = strides.strides_in_label(self.mask, self.min_duration)

        stance_duration = stride_frames[:, 1] - stride_frames[:, 0]
        stride_duration = stride_frames[:, 2] - stride_frames[:, 0]
        return stance_duration / stride_duration

    def step_distances(self, strides, strides_opposite, debug_plot=8):
        strides_frames_opp = strides_opposite.strides_in_label(
            self.mask, self.min_duration
        )
        strides_frames = strides.strides_in_label(self.mask, self.min_duration)

        # real locs 0 for the stride opposite side, 1 for the side in question
        paw_locs = self.mouse.locs(
            parts=(
                strides_opposite.paw_part,
                strides.paw_part,
            )
        )

        cnt = 0

        step_widths = []
        step_lengths = []

        for opp_stance_t, _, _ in strides_frames_opp:
            found_opp_stance = np.nonzero(
                np.logical_and(
                    opp_stance_t > strides_frames[:, 0],
                    opp_stance_t < strides_frames[:, 2],
                )
            )[0]
            if len(found_opp_stance) == 1:
                stride_mapped_r = strides_frames[found_opp_stance[0]]

                p1 = paw_locs[stride_mapped_r[0], 1]
                p2 = paw_locs[stride_mapped_r[2], 1]

                p_opp = paw_locs[opp_stance_t, 0]

                p_opp_proj = project_point_on_line(p2, p1, p_opp)

                step_length = np.linalg.norm(p2 - p_opp_proj) * self.pixel_size
                step_width = np.linalg.norm(p_opp - p_opp_proj) * self.pixel_size

                step_widths.append(step_width)
                step_lengths.append(step_length)

                if cnt == debug_plot:
                    f, ax = plt.subplots()
                    ax.imshow(self.mouse.image(stride_mapped_r[0]), "Reds", alpha=0.5)
                    ax.imshow(self.mouse.image(stride_mapped_r[2]), "Greens", alpha=0.5)
                    ax.plot((p1[0], p2[0]), (p1[1], p2[1]), "g-o")
                    ax.plot(*p2, "ro")
                    ax.plot((p_opp[0], p_opp_proj[0]), (p_opp[1], p_opp_proj[1]), "b-+")
                    ax.set_aspect(1.0)

                cnt += 1

        return np.array(step_lengths), np.array(step_widths)

    def raw_lateral_displacement(self, strides, part, part_to_proj_on, debug_plot=8):
        def cross2d(x, y):
            return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

        strides_frames = strides.strides_in_label(self.mask, self.min_duration)

        # real locs 0 for the stride opposite side, 1 for the side in question
        part_to_proj_on_locs = self.mouse.locs(parts=(part_to_proj_on,)).squeeze()
        parts_locs = self.mouse.locs(parts=(part,)).squeeze()

        displacements = {}

        for stride_ix, (stance_start, swing_start, stride_end) in enumerate(
            strides_frames
        ):

            p1 = part_to_proj_on_locs[stance_start]
            p2 = part_to_proj_on_locs[stride_end]

            ps = [parts_locs[pt] for pt in range(stance_start, stride_end)]

            ps_proj = [project_point_on_line(p1, p2, p) for p in ps]

            ps_proj_sign = [np.sign(cross2d(p2 - p1, p1 - p)) for p in ps]

            displacements[stride_ix] = np.array(
                [
                    sig * np.linalg.norm(pp - p)
                    for pp, p, sig in zip(ps_proj, ps, ps_proj_sign)
                ]
            )

            if stride_ix == debug_plot:

                f, ax = plt.subplots()
                ax.imshow(self.mouse.image(stance_start), "Reds", alpha=0.5)
                ax.imshow(self.mouse.image(stride_end), "Greens", alpha=0.5)
                ax.plot((p1[0], p2[0]), (p1[1], p2[1]), "g-o")
                ax.plot(*p2, "ro")
                for p, p_proj, p_sign in zip(ps, ps_proj, ps_proj_sign):
                    color = "m"
                    if p_sign < 0:
                        color = "c"
                    ax.plot((p[0], p_proj[0]), (p[1], p_proj[1]), f"{color}-+")
                ax.set_aspect(1.0)
                ax.set_xlim(
                    p2[0] - 256,
                    p2[0] + 256,
                )
                ax.set_ylim(
                    p2[1] + 256,
                    p2[1] - 256,
                )

                # axl.plot(range(stance_start, stride_end), displacements[stride_ix])

        return displacements

    def lateral_displacements_metrics(
        self, strides, part, part_to_proj_on, sigma_pf=3, debug_plot=-1
    ):
        rld = self.raw_lateral_displacement(
            strides, part, part_to_proj_on, debug_plot=debug_plot
        )

        phase = []
        displ = []
        for k, v in rld.items():
            displ.append(v.max() - v.min())
            x = np.linspace(0, 1, 100)
            y = np.interp(x * (len(v) - 1), np.arange(len(v)), v)

            sy = gaussian_filter1d(y, sigma_pf)

            i_peaks, _ = find_peaks(sy)

            i_max_peak = np.nan
            if len(i_peaks) > 0:
                i_max_peak = i_peaks[np.argmax(sy[i_peaks])]

                phase.append(i_max_peak)

        return np.array(displ) * self.pixel_size, np.array(phase) / 100
