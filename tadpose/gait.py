import numpy as np
import pandas as pd
from .analysis import angular_velocity


from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

from matplotlib.patches import Rectangle

from skimage import measure as skm

from .alignment import RotationalAligner
from .tadpose import Tadpole


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

        if len(maxima) == 0 or len(minima) == 0:
            raise ValueError(
                f"Stride.find_strides(): no peaks found for '{self.paw_part}'. "
                "Lower min_peak_prominence_px or check the paw signal."
            )

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
        """
        Plot the paw signal with shaded stance and swing phases.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on.
        slc : slice, optional
            Frame range to display. Defaults to the full recording.
        color_stance : str
            Colour for the paw signal line and stance-phase patches.
        alpha_stance : float
            Opacity of stance-phase patches.
        color_swing : str
            Colour for swing-phase patches.
        alpha_swing : float
            Opacity of swing-phase patches.
        """
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
        """
        Validated stride frames as an (N, 3) array.

        Returns all detected strides if ``validate_strides`` has not been
        called yet, otherwise only strides that contain at least one
        contralateral stance start (i.e. bilateral strides).

        Returns
        -------
        np.ndarray of shape (N, 3)
            Each row is ``[stance_start, swing_start, stride_stop]`` in
            frame indices.
        """
        if self._valid_strides_bind is None:
            strides_bind = np.ones(len(self._stride_frames), dtype=bool)
        else:
            strides_bind = self._valid_strides_bind

        return self._stride_frames[strides_bind]

    def describe(self):
        """Print a short summary of total and validated stride counts."""
        print(
            f"Strides for '{self.paw_part}'\n # Strides: {len(self._stride_frames)}\n # Strides (valid): {len(self.stride_frames)}"
        )

    def iter(self, start, stop):
        """
        Yield validated strides whose full extent falls within [start, stop].

        Parameters
        ----------
        start : int
            First frame of the window (exclusive for stance start).
        stop : int
            Last frame of the window (exclusive for stride stop).

        Yields
        ------
        tuple of int
            ``(stance_start, swing_start, stride_stop)`` for each qualifying stride.
        """
        a = self.stride_frames[:, 0] > start
        b = self.stride_frames[:, 2] < stop
        for stance_start, swing_start, stride_stop in self.stride_frames[a & b]:
            yield stance_start, swing_start, stride_stop

    def strides_in_label(self, mask, min_duration):
        """
        Return validated strides that fall entirely within labelled mask regions.

        Regions shorter than ``min_duration`` frames are ignored. A stride is
        included only if it belongs to exactly one qualifying region (strides
        that straddle region boundaries are excluded).

        Parameters
        ----------
        mask : np.ndarray of bool, shape (nframes,)
            Boolean mask marking frames of interest (e.g. moving bouts).
        min_duration : float
            Minimum region length in frames.

        Returns
        -------
        np.ndarray of shape (N, 3)
            Subset of ``stride_frames`` whose strides lie within a valid region.
        """
        self._mask = np.zeros_like(mask)
        in_label_strides = np.zeros(len(self.stride_frames), dtype=np.uint8)
        for rp in skm.regionprops(skm.label(mask)[:, None]):
            if rp.area > min_duration:
                slc = rp.slice[0]
                self._mask[slc] = True
                in_label_strides += np.logical_and(
                    self.stride_frames[:, 0] >= slc.start,
                    self.stride_frames[:, 2] <= slc.stop,
                )

        return self.stride_frames[in_label_strides == 1]

    def strides_in_label_slice_iter(self, mask, min_duration):
        """
        Yield frame slices for each mask region longer than ``min_duration``.

        Parameters
        ----------
        mask : np.ndarray of bool, shape (nframes,)
            Boolean mask marking frames of interest.
        min_duration : float
            Minimum region length in frames; shorter regions are skipped.

        Yields
        ------
        slice
            Frame slice ``[start, stop)`` of each qualifying region.
        """
        for rp in skm.regionprops(skm.label(mask)[:, None]):
            if rp.area > min_duration:
                yield rp.slice[0]

    def stride_idx_of_frame(self, frame):
        """
        Return the index of the stride that contains ``frame``.

        Parameters
        ----------
        frame : int
            Frame number to look up.

        Returns
        -------
        int
            Index into ``stride_frames`` of the containing stride, or ``-1``
            if ``frame`` does not fall within any validated stride.
        """
        sidx = np.nonzero(
            np.logical_and(
                self.stride_frames[:, 0] <= frame,
                self.stride_frames[:, 2] > frame,
            )
        )[0]

        if len(sidx) == 0:
            return -1
        return sidx[0]


def project_point_on_line(line_p1, line_p2, pnt, must_be_on_line=False):
    """
    Project a 2-D point onto the line defined by two anchor points.

    Parameters
    ----------
    line_p1, line_p2 : array-like of shape (2,)
        Two distinct points defining the line.
    pnt : array-like of shape (2,)
        The point to project.
    must_be_on_line : bool
        If ``True``, raise ``AttributeError`` when the projection falls
        outside the segment ``[line_p1, line_p2]`` (i.e. ``t`` not in (0, 1)).

    Returns
    -------
    np.ndarray of shape (2,)
        Coordinates of the projected point on the line.

    Raises
    ------
    AttributeError
        If ``line_p1 == line_p2`` (degenerate line), or if
        ``must_be_on_line`` is ``True`` and the projection is off-segment.
    """
    line_dist = np.sum((line_p1 - line_p2) ** 2)
    if line_dist == 0:
        raise AttributeError(
            "project_point_on_line(): line_p1 and line_p2 are the same points"
        )

    t = np.sum((pnt - line_p1) * (line_p2 - line_p1)) / line_dist
    if must_be_on_line and not (0 < t < 1):
        raise AttributeError("project_point_on_line(): pnt not in line")

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
        self.fps = mouse.video_fps

        self.mask = mask
        self.min_duration = min_duration_sec * self.fps

        self.pixel_size = pixel_size

        self.track_idx = track_idx

        self.mask_for_strides = np.zeros_like(mask)
        for rp in skm.regionprops(skm.label(mask)[:, None]):
            if rp.area > self.min_duration:
                slc = rp.slice[0]
                self.mask_for_strides[slc] = True

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
            res.append(np.nanmean(part_speed[stance_start:stride_stop]))

        return np.array(res)

    def stride_acceleration(self, strides, part_for_acc):
        """
        Calculates the average acceleration of a specified body part during each stride.
        Parameters
        ----------
        strides : object
            An object that provides stride intervals via the `strides_in_label` method.
        part_for_acc : str or int
            The identifier for the body part whose acceleration is to be calculated.
        Returns
        -------
        np.ndarray
            An array containing the mean acceleration of the specified body part for each stride interval.
        Notes
        -----
        - The acceleration is computed using the `mouse.acceleration` method, scaled by `pixel_size` and `fps`.
        - Only stride intervals that satisfy the mask and minimum duration are considered.
        """
        part_acc = (
            self.mouse.acceleration(part=part_for_acc, track_idx=self.track_idx)
            * self.pixel_size
            * self.fps**2
        )

        res = []

        for stance_start, swing_start, stride_stop in strides.strides_in_label(
            self.mask, self.min_duration
        ):
            res.append(np.nanmean(part_acc[stance_start:stride_stop]))

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

    def stride_duration(self, strides):
        """
        Calculate stance and stride durations for given strides.
        Parameters:
            strides: An object with a `strides_in_label` method that returns an array of stride frame indices.
                     The method is called with `self.mask` and `self.min_duration` as arguments.
        Returns:
            np.ndarray: A stacked array of shape (N, 3), where N is the number of strides.
                - The first column contains stance durations (frames).
                - The second column contains stride durations (frames).
                - The third column contains the starting frame index of each stride.
        """
        stride_frames = strides.strides_in_label(self.mask, self.min_duration)

        stance_duration = stride_frames[:, 1] - stride_frames[:, 0]
        stride_duration = stride_frames[:, 2] - stride_frames[:, 0]
        return np.stack(
            [stance_duration, stride_duration, stride_frames[:, 0]], axis=-1
        )

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
            np.ndarray: An array of shape (N, 3), where N is the number of strides. Each row contains
                [step_length, step_width, step_phase]. Lengths and widths are in physical units
                (e.g., cm). Step phase is the fraction of the stride cycle (0–1) at which the
                contralateral stance starts.
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
        step_phases = []

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
                step_phases.append(np.nan)
                continue

            stride_opp_stance = strides_frames_opp[match_index[0], 0]

            p1 = paw_locs[stride[0], 1]
            p2 = paw_locs[stride[2], 1]

            p_opp = paw_locs[stride_opp_stance, 0]

            try:
                p_opp_proj = project_point_on_line(p2, p1, p_opp, must_be_on_line=True)
            except AttributeError:
                step_widths.append(np.nan)
                step_lengths.append(np.nan)
                step_phases.append(np.nan)
                continue

            step_length = np.linalg.norm(p2 - p_opp_proj) * self.pixel_size
            step_width = np.linalg.norm(p_opp - p_opp_proj) * self.pixel_size
            step_phase = (stride_opp_stance - stride[0]) / (stride[2] - stride[0])

            step_phases.append(step_phase)
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
                ax_p.set_ylabel("Aligned paw y-location")

                ax.legend()

            cnt += 1

        return np.stack((step_lengths, step_widths, step_phases), axis=-1)

    def raw_lateral_displacement(self, strides, part, part_to_proj_on, debug_plot=8):
        """
        Compute signed lateral displacement of a body part relative to the stride axis.

        For each stride the stride axis is defined by the trajectory of
        ``part_to_proj_on`` from stance start to stride end. Each frame's
        location of ``part`` is projected onto that axis; the signed
        perpendicular distance is returned (positive = left of the axis,
        negative = right).

        Parameters
        ----------
        strides : Stride
            Stride object whose ``strides_in_label`` method defines the
            stride epochs to analyse.
        part : str
            Body part whose lateral displacement is measured.
        part_to_proj_on : str
            Body part whose trajectory defines the reference (stride) axis
            (e.g. ``"Spine_Center"``).
        debug_plot : int
            Index of the stride for which to generate a diagnostic figure.
            Pass ``-1`` to suppress all plots.

        Returns
        -------
        dict
            Maps stride index (int) to a 1-D ``np.ndarray`` of signed
            lateral displacements in pixels, one value per frame within
            the stride.
        """

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
            p2 = part_to_proj_on_locs[
                min(stride_end + 1, len(part_to_proj_on_locs) - 1)
            ]

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
                fig = plt.figure(figsize=(6, 10))
                axs = fig.subplot_mosaic(
                    mosaic="""
                            AAAAAA
                            AAAAAA
                            AAAAAA
                            BBBBBB
                            """,
                )
                ax1 = axs["A"]
                ax2 = axs["B"]

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
            tuple:
                - **np.ndarray of shape (N, 2)**: One row per stride; column 0 is the peak-to-peak
                  lateral displacement in physical units; column 1 is the phase (0–1) of the
                  maximum peak.
                - **list of np.ndarray**: Raw (unscaled) per-frame displacement vectors, one 1-D
                  array per stride.
        Notes:
            - Uses Gaussian smoothing to reduce noise in the displacement signal.
            - The phase is computed relative to a normalized stride duration.
        """

        rld = self.raw_lateral_displacement(
            strides, part, part_to_proj_on, debug_plot=debug_plot
        )

        phase = []
        displ = []
        raw_vec = []
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
            raw_vec.append(v)

        return np.stack(
            (np.array(displ) * self.pixel_size, np.array(phase) / 100), axis=-1
        ), raw_vec

    def stride_x_corr(self, strides, xcorr_part, debug_plot=1):
        """
        Compute the Pearson correlation between the paw signal and a second body part per stride.

        Both signals are taken from ego-centric (aligned) coordinates,
        z-scored within each stride, then correlated. Strides whose stance
        phase is shorter than 2 frames or whose signal has zero variance are
        returned as ``np.nan``.

        Parameters
        ----------
        strides : Stride
            Stride object defining the paw and the epochs to analyse.
        xcorr_part : str
            Name of the body part to cross-correlate against the paw signal.
        debug_plot : int
            Index of the stride for which to show a diagnostic plot.
            Pass ``-1`` to suppress all plots.

        Returns
        -------
        tuple
            - **np.ndarray of shape (N,)** : Pearson correlation coefficient for each stride
              (``nan`` for strides that could not be computed).
            - **list of tuple** : Raw signal pairs ``(sig, opp_sig)`` for each stride, where each
              element is a pair of 1-D arrays (z-scored paw signal and z-scored partner signal).
        """
        strides_frames = strides.strides_in_label(self.mask, self.min_duration)

        # real locs 0 for the stride opposite side, 1 for the side in question
        xcorr_ego_locs = self.mouse.ego_locs(
            parts=(
                strides.paw_part,
                xcorr_part,
            ),
            track_idx=self.track_idx,
        )

        res = []
        vec = []
        for stride_ix, (stance_start, swing_start, stride_end) in enumerate(
            strides_frames
        ):
            # if swing_start - stance_start < 2:
            #     res.append(np.nan)
            #     vec.append((np.nan, np.nan))
            #     continue

            sig = xcorr_ego_locs[stance_start:stride_end, 0, 1]
            opp_sig = xcorr_ego_locs[stance_start:stride_end, 1, 1]

            vec.append((sig, opp_sig))


            # if sig.std() == 0 or opp_sig.std() == 0:
            #     res.append(np.nan)
            #     vec.append((np.nan, np.nan))
            #     continue

            sig = (sig - sig.mean()) / sig.std()
            opp_sig = (opp_sig - opp_sig.mean()) / opp_sig.std()

            sig_corr_stride = pearsonr(sig, opp_sig).statistic

            # sig_corr_stance = pearsonr(
            #     sig[: swing_start - stance_start], opp_sig[: swing_start - stance_start]
            # ).statistic

            # sig_corr_swing = pearsonr(
            #     sig[swing_start - stance_start :], opp_sig[swing_start - stance_start :]
            # ).statistic

            res.append(sig_corr_stride)
            # res.append((sig_corr_stride, sig_corr_stance, sig_corr_swing))

            # cc = np.correlate(sig, opp_sig, mode="full")

            if stride_ix == debug_plot:
                f, ax = plt.subplots()
                ax.plot(
                    sig,
                    label=strides.paw_part,
                )
                ax.plot(opp_sig, label=xcorr_part)
                # ax.plot(cc, label="corr")
                ax.axvline(
                    swing_start - stance_start,
                    color="gray",
                    ls="--",
                    label="Swing Start",
                )
                # ax.set_title(
                #     f"Stride {stride_ix} xcorr:\nstride: {sig_corr_stride:.3f}\nstance: {sig_corr_stance:.3f}\nswing{sig_corr_swing:.3f}"
                # )
                ax.set_title(f"Stride {stride_ix} xcorr: {sig_corr_stride:.3f}")

                ax.legend()

        return np.array(res), vec


def stride_to_dataframe(
    sp,
    strides,
    strides_opposite=None,
    stride_type="",
    part_for_speed=None,
    parts_angular_velocity=None,
    parts_lat_displacement=None,
    part_xcorr=None,
    parts_body_size=None,
):
    """
    Compute all per-stride gait metrics and return them as a tidy DataFrame.

    One row per stride. Multiple paws can be stacked with pd.concat() because
    the 'paw' column distinguishes them.

    Parameters
    ----------
    sp : StrideProperties
        Carries mouse, pixel_size, fps, mask, min_duration, track_idx.
    strides : Stride
        The paw whose strides define the rows.
    strides_opposite : Stride, optional
        Contralateral paw — required for step_length, step_width, step_phase.
    stride_type : str
        Label for the limb pair, e.g. "Hind" or "Fore".
    part_for_speed : str, optional
        Body part used for stride_speed (e.g. "Spine_Center").
    parts_angular_velocity : tuple of str, optional
        Two body parts defining the angular-velocity axis, e.g. ("Neck_Base", "Tail_Base").
    parts_lat_displacement : dict, optional
        Mapping ``label -> (part, part_to_proj_on)`` for lateral displacement metrics,
        e.g. ``{"Nose": ("Nose", "Spine_Center"), "Tail_base": ("Tail_Base", "Spine_Center")}``.
    part_xcorr : str, optional
        Body part to cross-correlate the paw signal against.
    parts_body_size : tuple of str, optional
        Two body-part names (e.g. ``("Neck_Base", "Tail_Base")``) whose mean distance
        is stored as ``stride_body_lengths_cm`` for body-length normalisation.

    Returns
    -------
    pd.DataFrame
        Always-present columns: Video_fn, Track_idx, Stride_type, Paw,
        stance_start_frame, swing_start_frame, stride_stop_frame,
        stance_start_x_cm, stance_start_y_cm, stride_stop_x_cm, stride_stop_y_cm,
        stride_duration_sec, duty_factor, stride_length_cm.
        Optional columns (present when the corresponding argument is supplied):
        [stride_speed_cm_s, stride_acceleration_cm_s2], [angular_vel_deg_s],
        [step_length_cm, step_width_cm, step_phase],
        [xcorr, xcorr_vec, xcorr_vec_opp],
        [lat_displ_dist_<label>_cm, lat_displ_phase_<label>, lat_displ_vec_<label>],
        [stride_body_lengths_cm].
    """
    stride_frames = strides.strides_in_label(sp.mask, sp.min_duration)

    if len(stride_frames) == 0:
        return pd.DataFrame()

    paw_locs = sp.mouse.locs(
        parts=(strides.paw_part,), track_idx=sp.track_idx
    ).squeeze()

    rows = {
        "Video_fn": sp.mouse.video_fn,
        "Track_idx": sp.track_idx,
        "Stride_type": stride_type,
        "Paw": strides.paw_part,
        "stance_start_frame": stride_frames[:, 0].astype(int),
        "swing_start_frame": stride_frames[:, 1].astype(int),
        "stride_stop_frame": stride_frames[:, 2].astype(int),
        "stance_start_x_cm": paw_locs[stride_frames[:, 0], 0] * sp.pixel_size,
        "stance_start_y_cm": paw_locs[stride_frames[:, 0], 1] * sp.pixel_size,
        "stride_stop_x_cm": paw_locs[stride_frames[:, 2], 0] * sp.pixel_size,
        "stride_stop_y_cm": paw_locs[stride_frames[:, 2], 1] * sp.pixel_size,
        "stride_duration_sec": (stride_frames[:, 2] - stride_frames[:, 0]) / sp.fps,
        "duty_factor": sp.duty_factor(strides),
        "stride_length_cm": sp.stride_length(strides),
    }

    if part_for_speed is not None:
        rows["stride_speed_cm_s"] = sp.stride_speed(strides, part_for_speed)
        rows["stride_acceleration_cm_s2"] = sp.stride_acceleration(
            strides, part_for_speed
        )

    if parts_angular_velocity is not None:
        rows["angular_vel_deg_s"] = sp.angular_velocity(strides, parts_angular_velocity)

    if strides_opposite is not None:
        step = sp.step_distances(strides, strides_opposite, debug_plot=-1)
        rows["step_length_cm"] = step[:, 0]
        rows["step_width_cm"] = step[:, 1]
        rows["step_phase"] = step[:, 2]

    if part_xcorr is not None:
        xcorr, vecs = sp.stride_x_corr(strides, part_xcorr, debug_plot=-1)
        vec, vec_opp = zip(*vecs)
        rows["xcorr"] = xcorr
        rows["xcorr_vec"] = vec
        rows["xcorr_vec_opp"] = vec_opp  


    if parts_lat_displacement is not None:
        for label, (part, proj_part) in parts_lat_displacement.items():
            ld, raw = sp.lateral_displacements_metrics(
                strides, part, proj_part, debug_plot=-1
            )
            rows[f"lat_displ_dist_{label}_cm"] = ld[:, 0]
            rows[f"lat_displ_phase_{label}"] = ld[:, 1]
            rows[f"lat_displ_vec_{label}"] = raw

    if parts_body_size is not None:
        rows["stride_body_lengths_cm"] = sp.distance(strides, *parts_body_size)

    return pd.DataFrame(rows)


class GaitAnalysis:
    """
    High-level pipeline for rodent gait analysis from a SLEAP tracking file.

    Loads a video/tracking file, sets up rotational alignment, detects
    moving bouts, finds and validates stride cycles for each defined limb
    pair, and exposes methods to extract and summarise per-stride features.

    Parameters
    ----------
    video_fn : str
        Path to the SLEAP HDF5 file (used by ``Tadpole.from_sleap``).
    track_idx : int
        Which SLEAP track to use when multiple animals are present.
    cfg : dict
        Configuration dictionary with the following keys:

        - ``ALIGN_CENTRAL`` / ``ALIGN_TOP`` : body parts used by
          ``RotationalAligner`` to orient each frame head-up.
        - ``ARENA_SIZE_CM`` / ``ARENA_SIZE_PX`` : arena dimensions used
          to convert pixel distances to centimetres.
        - ``MOVING_SIGMA_SEC`` : Gaussian smoothing sigma (seconds) for
          the forward-speed signal.
        - ``MOVING_THRESH`` : forward-speed threshold (cm/s) above which
          the animal is considered moving.
        - ``MOVING_MIN_DURATION_SEC`` : minimum bout length (seconds) for
          a moving epoch to be included in stride analysis.
        - ``PEAK_MIN_PROMINENCE_PX`` : minimum peak prominence (pixels)
          for stride detection in the paw signal.
        - ``PEAK_SMOOTH_SIGMA_SEC`` : Gaussian smoothing sigma (seconds)
          applied to the paw signal before peak finding.
        - ``STRIDE_DEF`` : dict mapping a stride-type label (e.g.
          ``"Hind"``) to a ``(left_paw_part, right_paw_part)`` tuple.
        - ``STRIDE_SPEED_NODE`` : body-part name used for stride speed.
        - ``STRIDE_ANG_VEL_AXIS`` : two body-part names defining the
          angular-velocity axis.
        - ``STRIDE_LAT_DISPL_DEF`` : dict passed to
          ``stride_to_dataframe`` for lateral displacement metrics.
        - ``ANIMAL_LENGTH_DEF`` : two body-part names whose distance
          defines body length for normalisation.

    Attributes
    ----------
    mouse : Tadpole
        Loaded tracking object with rotational alignment attached.
    pixel_size : float
        Centimetres per pixel.
    mouse_fw_speed : np.ndarray
        Per-frame forward speed in cm/s.
    mouse_is_moving_mask : np.ndarray of bool
        True for frames where forward speed exceeds ``MOVING_THRESH``.
    stride_props : StrideProperties
        Shared property computer (pixel size, mask, fps).
    stride_objs : dict
        Maps each stride-type label to a ``(Stride_left, Stride_right)``
        tuple with validated stride cycles.

    Methods
    -------
    plot()
        Plot paw signals with stance/swing patches and the speed trace.
    compute_raw_features()
        Return a tidy DataFrame with one row per stride and one column
        per raw metric (lengths, speed, duty factor, etc.).
    compute_features()
        Like ``compute_raw_features`` but also adds body-length-normalised
        columns and the duty-factor temporal-symmetry index.
    """

    def __init__(self, mouse: Tadpole, track_idx: int, cfg: dict):
        self.cfg = cfg
        self.mouse = mouse

        self.mouse.aligner = RotationalAligner(
            central_part=self.cfg["ALIGN_CENTRAL"],
            aligned_part=self.cfg["ALIGN_TOP"],
            align_to=(0, 1),
        )

        self.pixel_size = self.cfg["ARENA_SIZE_CM"] / self.cfg["ARENA_SIZE_PX"]
        self.speed_calibration_factor = self.pixel_size * self.mouse.video_fps

        self.track_idx = track_idx

        self.mouse_fw_speed = (
            self.mouse.forward_speed(
                part="Spine_Center",
                sigma=self.cfg["MOVING_SIGMA_SEC"],
                track_idx=self.track_idx,
            )
            * self.speed_calibration_factor
        )

        self.mouse_is_moving_mask = self.mouse_fw_speed > self.cfg["MOVING_THRESH"]

        self.stride_props = StrideProperties(
            self.mouse,
            self.mouse_is_moving_mask,
            pixel_size=self.pixel_size,
            min_duration_sec=self.cfg["MOVING_MIN_DURATION_SEC"],
            track_idx=self.track_idx,
        )

        self.stride_objs = {}

        for stride_type, (paw_L, paw_R) in self.cfg["STRIDE_DEF"].items():
            strides_L = Stride(
                self.mouse,
                paw_L,
                min_peak_prominence_px=self.cfg["PEAK_MIN_PROMINENCE_PX"],
                sigma=self.cfg["PEAK_SMOOTH_SIGMA_SEC"],
                track_idx=self.track_idx,
            )

            strides_R = Stride(
                self.mouse,
                paw_R,
                min_peak_prominence_px=self.cfg["PEAK_MIN_PROMINENCE_PX"],
                sigma=self.cfg["PEAK_SMOOTH_SIGMA_SEC"],
                track_idx=self.track_idx,
            )

            strides_L.find_strides()
            strides_R.find_strides()

            strides_L.validate_strides(strides_R)
            strides_R.validate_strides(strides_L)

            self.stride_objs[stride_type] = (strides_L, strides_R)

    def plot(self):
        """
        Plot paw signals, stance/swing phases, and the forward-speed trace.

        Produces one figure per stride type defined in ``cfg["STRIDE_DEF"]``.
        Each figure has three vertically stacked subplots sharing the x-axis:
        the left-paw signal (red stance patches), the right-paw signal (green
        stance patches), and the forward speed with the moving-bout mask
        overlaid.
        """
        from matplotlib import pyplot as plt

        for stride_type, (strides_L, strides_R) in self.stride_objs.items():
            f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            strides_L.plot(
                ax=ax1,
                color_stance="red",
                alpha_swing=0.2,
            )

            strides_R.plot(
                ax=ax2,
                color_stance="green",
                alpha_swing=0.2,
            )

            ax3.plot(self.mouse_fw_speed)
            ax3.plot(self.stride_props.mask_for_strides * self.cfg["MOVING_THRESH"])

            ax1.set_xlim(0, 400)
            ax1.legend()
            ax2.legend()

    def compute_raw_features(self):
        """
        Compute per-stride gait metrics for all defined limb pairs.

        For each stride type in ``cfg["STRIDE_DEF"]`` both the left and
        right paw are processed with ``stride_to_dataframe``, using the
        contralateral paw as the opposite-side reference.

        Returns
        -------
        pd.DataFrame
            Tidy table with one row per stride. Columns include frame
            indices, spatial coordinates, duty factor, stride length,
            speed, angular velocity, step distances, lateral displacement
            metrics, cross-correlation, and body-length normalisation
            inputs. See ``stride_to_dataframe`` for the full column list.
        """
        result_tab = []
        for stride_type, (strides_L, strides_R) in self.stride_objs.items():
            props_L = stride_to_dataframe(
                self.stride_props,
                strides_L,
                strides_R,
                stride_type=stride_type,
                part_for_speed=self.cfg["STRIDE_SPEED_NODE"],
                parts_angular_velocity=self.cfg["STRIDE_ANG_VEL_AXIS"],
                parts_lat_displacement=self.cfg["STRIDE_LAT_DISPL_DEF"],
                part_xcorr=strides_R.paw_part,
                parts_body_size=self.cfg["ANIMAL_LENGTH_DEF"],
            )

            props_R = stride_to_dataframe(
                self.stride_props,
                strides_R,
                strides_L,
                stride_type=stride_type,
                part_for_speed=self.cfg["STRIDE_SPEED_NODE"],
                parts_angular_velocity=self.cfg["STRIDE_ANG_VEL_AXIS"],
                parts_lat_displacement=self.cfg["STRIDE_LAT_DISPL_DEF"],
                part_xcorr=strides_L.paw_part,
                parts_body_size=self.cfg["ANIMAL_LENGTH_DEF"],
            )

            result_tab.extend([props_L, props_R])

        return pd.concat(result_tab, ignore_index=True)

    def compute_features(self):
        """
        Compute raw gait features and add derived summary metrics.

        Extends ``compute_raw_features`` with:

        - **Body-length-normalised columns** — if ``cfg["ANIMAL_LENGTH_DEF"]``
          is set, spatial metrics (stride length, speed, step length/width,
          lateral displacement distances) are divided by the mean body length
          across all strides and stored in ``relative_*`` columns.
        - **Duty-factor temporal symmetry** — for each stride type, the
          symmetry index ``(L - R) / (0.5 * (L + R))`` is computed from
          the mean duty factors of the left and right paw and broadcast to
          every row of that stride type.

        Returns
        -------
        pd.DataFrame
            All columns from ``compute_raw_features`` plus ``average_speed_cm_s``
            (whole-session mean speed), ``relative_*`` body-length-normalised
            columns, and ``duty_factor_temporal_symmetry``.
        """
        tab = self.compute_raw_features()

        ### avg spped
        tab["average_speed_cm_s"] = (
            np.nanmean(self.mouse.speed(part=self.cfg["STRIDE_SPEED_NODE"]))
            * self.speed_calibration_factor
        )

        ### Body length normalization
        if "stride_body_lengths_cm" in tab.columns:
            body_length = tab.stride_body_lengths_cm.mean()
            cols_to_norm = [
                "stride_length_cm",
                "stride_speed_cm_s",
                "step_length_cm",
                "step_width_cm",
                "stride_body_lengths_cm",
                "average_speed_cm_s",
            ]

            for c in tab.columns:
                if c.startswith("lat_displ_dist_"):
                    cols_to_norm.append(c)

            for col in cols_to_norm:
                if col in tab.columns:
                    tab[f"relative_{col.replace('_cm', '')}"] = tab[col] / body_length

        ### temporal symmetry for duty factor

        stride_types = tab.Stride_type.unique()

        tab["duty_factor_temporal_symmetry"] = np.nan
        for st in stride_types:
            l_dtf, r_dtf = tab.groupby(["Stride_type", "Paw"]).duty_factor.mean()[st]
            temp_symm_per_st = (l_dtf - r_dtf) / (0.5 * (l_dtf + r_dtf))
            tab.loc[tab.Stride_type == st, "duty_factor_temporal_symmetry"] = (
                temp_symm_per_st
            )

        return tab
