import numpy as np
from .analysis import angular_velocity


from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from matplotlib.patches import Rectangle

from skimage import measure as skm


class Stride:
    def __init__(self, mouse, paw_part, min_peak_prominence_px, sigma=0, track_idx=0):
        """
        Initializes the gait analysis object for a specific mouse and paw part.
        Args:
            mouse: The mouse object associated with the gait analysis.
            paw_part: The specific paw part (e.g., 'left_hind', 'right_fore') to analyze.
            min_peak_prominence_px (float): Minimum peak prominence in pixels for stride detection.
            sigma (float, optional): Standard deviation for Gaussian smoothing of the paw signal. Defaults to 0 (no smoothing).
            track_idx (int, optional): Index of the tracking data to use. Defaults to 0.
        """
        self.mouse = mouse
        self.paw_part = paw_part
        self.min_peak_prominence = min_peak_prominence_px
        self.sigma = sigma
        self.track_idx = track_idx

        self._stride_frames = None
        self.paw_signal = None
        self._valid_strides_bind = None

    def find_strides(self):
        """
        Detects stride cycles based on the paw position signal.
        This method processes the paw position signal for a specified paw part,
        optionally applies Gaussian smoothing, and identifies local maxima and minima
        as stance and swing phase transitions, respectively. It then constructs an array
        of stride frames, where each row contains the indices for stance start, swing start,
        and stride stop for each detected stride.

        Sets the following attributes:
            - self.paw_signal: The processed paw position signal.
            - self._stride_frames: A (N-1, 3) array of stride frame indices, where each row is
                [stance_start, swing_start, stride_stop].
        """
        self.paw_signal = self.mouse.ego_locs(
            parts=(self.paw_part,), fill_missing=True, track_idx=self.track_idx
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

        assert len(maxima) == len(minima), (
            f"maxima vs minima len {len(maxima)} != {len(minima)}"
        )

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
            slc = slice(0, self.mouse.nframes)

        frames = np.arange(slc.start, slc.stop)

        sig_min = self.paw_signal[slc].min()
        sig_height = self.paw_signal[slc].max() - self.paw_signal[slc].min()

        ax.plot(
            frames, self.paw_signal[slc], color=color_stance, label=f"{self.paw_part}"
        )

        # plot stance
        first_time = True
        for stance_start, swing_start, stride_stop in self.stride_frames:
            if stance_start < slc.start or stride_stop > slc.stop:
                continue
            rect_stance = Rectangle(
                (stance_start, sig_min),
                swing_start - stance_start,
                sig_height,
                color=color_stance,
                alpha=alpha_stance,
                label="Stance" if first_time else None,
            )
            ax.add_patch(rect_stance)

            rect_swing = Rectangle(
                (swing_start, sig_min),
                stride_stop - swing_start,
                sig_height,
                color=color_swing,
                alpha=alpha_swing,
                label="Swing" if first_time else None,
            )
            ax.add_patch(rect_swing)

            first_time = False

        ax.set_xlim(slc.start, slc.stop)
        ax.set_ylabel(f"Aligned\n{self.paw_part}")

    def validate_strides(self, other):
        """
        Validates the strides of another object by checking if each stride interval in the current object
        contains at least one stance start from the other object's stride frames.
        """
        bins = np.r_[self._stride_frames[:, 0], self._stride_frames[-1, 2]]

        other_stance_start = other._stride_frames[:, 0]

        hist, edges = np.histogram(other_stance_start, bins=bins)

        self._valid_strides_bind = hist >= 1

    @property
    def stride_frames(self):
        if self._valid_strides_bind is None:
            strides_bind = np.ones(len(self._stride_frames), dtype=bool)
        else:
            strides_bind = self._valid_strides_bind

        return self._stride_frames[strides_bind]

    def describe(self):
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
                    self.stride_frames[:, 2] <= slc.stop,
                )

        return self.stride_frames[in_label_strides == 1]

    def strides_in_label_slice_iter(self, mask, min_duration):
        for rp in skm.regionprops(skm.label(mask)[:, None]):
            if rp.area > min_duration:
                yield rp.slice[0]

    def stride_idx_of_frame(self, frame):
        sidx = np.nonzero(
            np.logical_and(
                self.stride_frames[:, 0] <= frame,
                self.stride_frames[:, 2] > frame,
            )
        )[0]

        if len(sidx) == 0:
            return -1
        return sidx[0]


def project_point_on_line(line_p1, line_p2, pnt):
    line_dist = np.sum((line_p1 - line_p2) ** 2)
    if line_dist == 0:
        raise AttributeError(
            "project_point_on_line(): line_p1 and line_p2 are the same points"
        )

    t = np.sum((pnt - line_p1) * (line_p2 - line_p1)) / line_dist

    pnt_projection = line_p1 + t * (line_p2 - line_p1)
    return pnt_projection


class StrideProperties:
    def __init__(self, mouse, mask, pixel_size, min_duration_sec, track_idx):
        """
        Initializes the gait analysis object with mouse data, mask, and analysis parameters.
        Args:
            mouse: An object representing the mouse, expected to have a 'video_fps' attribute.
            mask: The mask to be applied for gait analysis.
            pixel_size: The size of a pixel in real-world units.
            min_duration_sec: Minimum duration (in seconds) for a valid gait event.
            track_idx: Index of the track to be analyzed.
        """
        self.mouse = mouse
        self.mask = mask

        self.pixel_size = pixel_size
        self.fps = mouse.video_fps
        self.min_duration = min_duration_sec * self.fps
        self.track_idx = track_idx

    def angular_velocity(self, strides, part_axis):
        """
        Computes the mean angular velocity for each stride segment defined in the input.
        Parameters:
            strides: An object providing stride segmentation, expected to have a `strides_in_label` method.
            part_axis: A tuple or list specifying the body part axes to compute angular velocity between.
        Returns:
            np.ndarray: Array of mean angular velocities for each stride segment.
        Notes:
            - Uses the `angular_velocity` function to compute per-frame angular velocities between specified axes.
            - The result is scaled by the frame rate (`self.fps`).
            - Only stride segments that satisfy `self.mask` and `self.min_duration` are considered.
        """
        av_seq = (
            angular_velocity(
                self.mouse, part_axis[0], part_axis[1], track_idx=self.track_idx
            )
            * self.fps
        )

        res = []

        for stance_start, swing_start, stride_stop in strides.strides_in_label(
            self.mask, self.min_duration
        ):
            res.append(av_seq[stance_start:stride_stop].mean())

        return np.array(res)

    def distance(self, strides, part_a, part_b):
        """
        Calculates the mean Euclidean distance between two specified body parts over each detected stride.
        For each stride segment defined by the input `strides`, this method computes the average distance
        (in physical units, accounting for `self.pixel_size`) between `part_a` and `part_b` using their
        tracked locations.
        Args:
            strides: An object providing stride segmentation, expected to have a `strides_in_label` method
                     that yields (stance_start, swing_start, stride_stop) tuples.
            part_a: The name or index of the first body part to measure.
            part_b: The name or index of the second body part to measure.
        Returns:
            np.ndarray: An array of mean distances (one per stride) between `part_a` and `part_b`.
        """
        locs = self.mouse.locs(
            parts=(part_a, part_b), track_idx=self.track_idx
        ).squeeze()

        res = []

        for stance_start, swing_start, stride_stop in strides.strides_in_label(
            self.mask, self.min_duration
        ):
            dist = (
                np.linalg.norm(
                    locs[stance_start:stride_stop, 0]
                    - locs[stance_start:stride_stop, 1],
                    axis=1,
                ).mean()
                * self.pixel_size
            )
            res.append(dist)

        return np.array(res)

    def stride_length(self, strides):
        """
        Calculates the stride lengths for a series of strides.
        Parameters:
            strides: An object containing stride information, including paw part and stride intervals.
        Returns:
            np.ndarray: An array of stride lengths, where each element corresponds to the distance (in physical units)
            between the stance start and stride stop locations for each stride.
        Notes:
            - Uses the mouse's tracked locations for the specified paw part and track index.
            - Stride intervals are determined using the provided mask and minimum duration.
            - Distances are scaled by the pixel size to convert from pixels to physical units.
        """
        locs = self.mouse.locs(
            parts=(strides.paw_part,), track_idx=self.track_idx
        ).squeeze()
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
        """
        Calculates the average speed of a specified body part during each stride.
        Parameters
        ----------
        strides : object
            An object that provides stride intervals via the `strides_in_label` method.
        part_for_speed : str or int
            The identifier for the body part whose speed is to be calculated.
        Returns
        -------
        np.ndarray
            An array containing the mean speed of the specified body part for each stride interval.
        Notes
        -----
        - The speed is computed using the `mouse.speed` method, scaled by `pixel_size` and `fps`.
        - Only stride intervals that satisfy the mask and minimum duration are considered.
        """
        part_speed = (
            self.mouse.speed(
                part=part_for_speed, sigma=0, pre_sigma=0, track_idx=self.track_idx
            )
            * self.pixel_size
            * self.fps
        )

        res = []

        for stance_start, swing_start, stride_stop in strides.strides_in_label(
            self.mask, self.min_duration
        ):
            res.append(part_speed[stance_start:stride_stop].mean())

        return np.array(res)

    def duty_factor(self, strides):
        """
        Calculates the duty factor for each stride based on stride frame indices.
        The duty factor is defined as the ratio of the stance phase duration to the total stride duration.
        Args:
            strides: An object that provides the `strides_in_label(mask, min_duration)` method,
                which returns a NumPy array of shape (N, 3), where each row contains the start,
                stance, and end frame indices for a stride.
        Returns:
            numpy.ndarray: An array of duty factors for each stride, computed as
                (stance_duration / stride_duration).
        """
        stride_frames = strides.strides_in_label(self.mask, self.min_duration)

        stance_duration = stride_frames[:, 1] - stride_frames[:, 0]
        stride_duration = stride_frames[:, 2] - stride_frames[:, 0]
        return stance_duration / stride_duration

    def step_distances(self, strides, strides_opposite, debug_plot=-1):
        """
        Calculates the step lengths and widths for each stride in a gait cycle.
        For each stride in the provided `strides` object, this method finds the corresponding stance phase
        in the opposite limb (`strides_opposite`), projects the opposite paw location onto the stride line,
        and computes the step length (distance along the stride direction) and step width (perpendicular distance).
        Optionally, a debug plot can be generated for a specific stride.
        Args:
            strides: An object representing the strides of the limb in question, containing stride frames and paw part information.
            strides_opposite: An object representing the strides of the opposite limb, containing stride frames and paw part information.
            debug_plot (int, optional): The index of the stride for which to generate a debug plot. Defaults to -1 (no plot).
        Returns:
            np.ndarray: An array of shape (N, 2), where N is the number of strides. Each row contains
                [step_length, step_width] for a stride. Values are in physical units (e.g., mm).
        """
        paw_locs = self.mouse.locs(
            parts=(
                strides_opposite.paw_part,
                strides.paw_part,
            ),
            track_idx=self.track_idx,
        )

        # strides_frames
        strides_frames = strides.strides_in_label(self.mask, self.min_duration)

        #
        strides_frames_opp = strides_opposite._stride_frames

        # real locs 0 for the stride opposite side, 1 for the side in question

        cnt = 0

        step_widths = []
        step_lengths = []

        for stride in strides_frames:
            match_index = np.nonzero(
                np.logical_and(
                    stride[0] <= strides_frames_opp[:, 0],
                    strides_frames_opp[:, 0] < stride[2],
                )
            )[0]
            if len(match_index) == 0:
                print(
                    "WARNING: No match index found. That should not happen. Strides validated??"
                )
                step_widths.append(np.nan)
                step_lengths.append(np.nan)
                continue

            stride_opp_stance = strides_frames_opp[match_index[0], 0]

            p1 = paw_locs[stride[0], 1]
            p2 = paw_locs[stride[2], 1]

            p_opp = paw_locs[stride_opp_stance, 0]

            p_opp_proj = project_point_on_line(p2, p1, p_opp)

            step_length = np.linalg.norm(p2 - p_opp_proj) * self.pixel_size
            step_width = np.linalg.norm(p_opp - p_opp_proj) * self.pixel_size

            step_widths.append(step_width)
            step_lengths.append(step_length)

            if cnt == debug_plot:
                f, (ax, ax_p) = plt.subplots(1, 2, figsize=(12, 5))
                ax.imshow(
                    self.mouse.image(stride[0]),
                    "Reds",
                    alpha=0.5,
                )
                ax.imshow(
                    self.mouse.image(stride[2]),
                    "Greens",
                    alpha=0.5,
                )
                ax.plot((p1[0], p2[0]), (p1[1], p2[1]), "g-", label="Stride length")
                ax.plot(*p1, "g.", label="Stride Start")
                ax.plot(*p2, "r.", label="Stride End")
                ax.plot(
                    (p_opp[0], p_opp_proj[0]),
                    (p_opp[1], p_opp_proj[1]),
                    "b-",
                    label="Step width",
                )
                ax.set_aspect(1.0)
                ax.set_xlim(p2[0] - 180, p2[0] + 180)
                ax.set_ylim(p2[1] + 180, p2[1] - 180)

                ax_p.plot(strides.paw_signal)
                ax_p.axvline(stride[0], color="g", label="Stride Start")
                ax_p.axvline(stride[2], color="r", label="Stride End")
                ax_p.set_xlim(int(stride[0]) - 100, int(stride[2]) + 100)
                ax_p.set_xlabel("Time (frames)")
                ax_p.set_label("Aligned paw y-location")

                ax.legend()

            cnt += 1

        return np.stack((step_lengths, step_widths), axis=-1)

    def raw_lateral_displacement(self, strides, part, part_to_proj_on, debug_plot=8):
        def cross2d(x, y):
            return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

        strides_frames = strides.strides_in_label(self.mask, self.min_duration)

        # real locs 0 for the stride opposite side, 1 for the side in question
        part_to_proj_on_locs = self.mouse.locs(
            parts=(part_to_proj_on,), track_idx=self.track_idx
        ).squeeze()
        parts_locs = self.mouse.locs(parts=(part,), track_idx=self.track_idx).squeeze()

        displacements = {}

        for stride_ix, (stance_start, swing_start, stride_end) in enumerate(
            strides_frames
        ):
            p1 = part_to_proj_on_locs[stance_start]
            p2 = part_to_proj_on_locs[stride_end + 1]

            ps = [parts_locs[pt] for pt in range(stance_start, stride_end + 1)]

            ps_proj = [project_point_on_line(p1, p2, p) for p in ps]

            ps_proj_sign = [np.sign(cross2d(p2 - p1, p1 - p)) for p in ps]

            displacements[stride_ix] = np.array(
                [
                    sig * np.linalg.norm(pp - p)
                    for pp, p, sig in zip(ps_proj, ps, ps_proj_sign)
                ]
            )

            if stride_ix == debug_plot:
                fig = plt.figure(figsize=(6, 10)).subplot_mosaic(
                    mosaic="""
                            AAAAAA
                            AAAAAA
                            AAAAAA
                            BBBBBB
                            """,
                )
                ax1 = fig["A"]
                ax2 = fig["B"]

                ax1.imshow(
                    self.mouse.image(stance_start),
                    "Reds",
                    alpha=0.5,
                )
                ax1.imshow(
                    self.mouse.image(stride_end),
                    "Greens",
                    alpha=0.5,
                )
                ax1.axline(
                    p1,
                    p2,
                    marker=".",
                    color="darkblue",
                    ls="--",
                    label="Stride Body Axis",
                )
                ax1.plot(*p1, "go", label="Center Stride Start")
                ax1.plot(*p2, "ro", label="Center Stride End")
                for p, p_proj, p_sign in zip(ps, ps_proj, ps_proj_sign):
                    color = "m"
                    if p_sign < 0:
                        color = "c"
                    ax1.plot((p[0], p_proj[0]), (p[1], p_proj[1]), f"{color}")
                ax1.plot(
                    *np.array(ps).T, "-", label=f"{part} Trajectory", color="darkred"
                )
                ax1.set_aspect(1.0)
                ax1.set_xlim(
                    p1[0] - 180,
                    p1[0] + 180,
                )
                ax1.set_ylim(
                    p1[1] + 180,
                    p1[1] - 180,
                )
                ax1.set_axis_off()
                ax1.legend()
                y = displacements[stride_ix]
                x = np.linspace(0, 100, len(y))
                ax2.plot(x, y, color="darkred")
                ax2.hlines(0, xmin=0, xmax=100, color="darkblue", ls="--")
                ax2.set_xlabel("Time (%Stride)")
                ax2.set_ylabel("Lateral Displacement (px)")

                # axl.plot(range(stance_start, stride_end), displacements[stride_ix])
                plt.tight_layout()

        return displacements

    def lateral_displacements_metrics(
        self, strides, part, part_to_proj_on, sigma_pf=3, debug_plot=-1
    ):
        """
        Computes lateral displacement metrics for a specified body part across multiple strides.
        For each stride, calculates the maximum lateral displacement and the phase (as a fraction of stride duration)
        at which the maximum peak occurs after smoothing the displacement signal.
        Args:
            strides (iterable): Collection of stride data to analyze.
            part (str or int): Identifier for the body part whose displacement is measured.
            part_to_proj_on (str or int): Identifier for the body part or axis onto which the displacement is projected.
            sigma_pf (float, optional): Standard deviation for Gaussian smoothing of the displacement signal. Default is 3.
            debug_plot (int, optional): If non-negative, enables debug plotting for the specified stride index. Default is -1.
        Returns:
            np.ndarray: Array of shape (N, 2), where N is the number of strides. Each row contains:
                - Maximum lateral displacement (in physical units, e.g., microns or mm)
                - Phase (fraction of stride, between 0 and 1) at which the maximum peak occurs
        Notes:
            - Uses Gaussian smoothing to reduce noise in the displacement signal.
            - The phase is computed relative to a normalized stride duration.
        """

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

        return np.stack(
            (np.array(displ) * self.pixel_size, np.array(phase) / 100), axis=-1
        )
