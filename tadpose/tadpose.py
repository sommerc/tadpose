import os
import cv2
import h5py
import json
import warnings
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm.auto import tqdm

from . import utils, analysis


SplineClass = analysis.ReparametrizedCSAPSSplineFit


class Tadpole:
    def __init__(self, video_fn, bodyparts_cmap):
        assert os.path.exists(video_fn), f"Movie file '{video_fn}' does not exist"

        self.video_fn = os.path.abspath(video_fn)
        self.vid_path = os.path.dirname(video_fn)
        self.vid_fn = os.path.basename(video_fn)
        self.vid_fn, self.vid_ext = os.path.splitext(self.vid_fn)

        self.bodyparts_cmap = bodyparts_cmap
        self._aligner = None

        self._vid_handle = None

        self.info = {}
        self.json_fn = f"{self.vid_path}/{self.vid_fn}.json"
        if os.path.exists(self.json_fn):
            with open(self.json_fn) as f:
                self.info = json.load(f)

    @staticmethod
    def from_dlc(video_fn, bodyparts_cmap="jet"):
        return DeeplabcutTadpole(video_fn=video_fn, bodyparts_cmap=bodyparts_cmap)

    @staticmethod
    def from_sleap(video_fn, bodyparts_cmap="rainbow"):
        return SleapTadpole(video_fn=video_fn, bodyparts_cmap=bodyparts_cmap)

    @property
    def analysis_file(self):
        pass

    def split_detection_and_likelihood(self):
        tmp = self.locations[self.bodyparts].to_np().reshape(-1, len(self.bodyparts), 3)
        return tmp[..., :2], tmp[..., 2]

    @property
    def bodyparts(self):
        return self._bodyparts

    @property
    def bodypart_colors(self):
        colorclass = matplotlib.cm.ScalarMappable(cmap=self.bodyparts_cmap)
        C = colorclass.to_rgba(np.linspace(0, 1, len(self.bodyparts)))
        return (C[:, :3] * 255).astype(np.uint8)

    @property
    def bodypart_color(self):
        return dict(
            [(k, v / 255.0) for k, v in zip(self.bodyparts, self.bodypart_colors)]
        )

    @property
    def aligner(self):
        return self._aligner

    @aligner.setter
    def aligner(self, ta):
        if hasattr(ta, "tracks_to_align"):
            tids = ta.tracks_to_align
        else:
            tids = range(len(self))

        # for track_idx in tqdm(tids, desc="Aligning animals"):
        for track_idx in tids:
            all_locations = self.locs(track_idx=track_idx)
            ta.fit(track_idx, self.bodyparts, all_locations)
        self._aligner = ta


class SleapTadpole(Tadpole):
    def __init__(self, video_fn, bodyparts_cmap, **kwargs):
        super().__init__(video_fn, bodyparts_cmap)
        self.scorer = "sleap"

        with h5py.File(self.analysis_file, "r") as hf:
            self.tracks = hf["tracks"][:].T
            self._bodyparts = list(map(lambda x: x.decode(), hf["node_names"]))

        self.current_track = 0

        if "end_frame" in self.info:
            print(f"INFO: end frame set to {self.info['end_frame']} for {self.vid_fn}")
            self.tracks = self.tracks[: self.info["end_frame"]]

        if "mutant_side" in self.info and self.info["mutant_side"] == "right":
            print("INFO: Mirror Left-Right for {self.vid_fn}")

            cap = cv2.VideoCapture(self.video_fn)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            cap.release()

            self.tracks[..., 0, :] *= -1
            self.tracks[..., 0, :] += width

            mirrored_body_parts = []
            for bp in self._bodyparts:
                bp_mirror = bp
                if "Right" in bp:
                    bp_mirror = bp.replace("Right", "Left")

                if "Left" in bp:
                    bp_mirror = bp.replace("Left", "Right")

                mirrored_body_parts.append(bp_mirror)

            self._bodyparts = mirrored_body_parts

    def __len__(self):
        return self.tracks.shape[-1]

    @property
    def nframes(self):
        return len(self.tracks)

    def check_frames(self, frames):
        if frames is None:
            frames = range(self.nframes)
        elif isinstance(frames, range):
            pass
        elif isinstance(frames, (list, tuple)) and len(frames) == 2:
            frames = range(frames[0], frames[1])
        elif isinstance(frames, (list, tuple)) and len(frames) == 3:
            frames = range(frames[0], frames[1], frames[2])
        elif isinstance(frames, (int,)):
            frames = range(frames, frames + 1)

        else:
            raise RuntimeError(
                "Frames must be integer, a list = [start, end) or [stard, end, step]"
            )
        return frames

    def locs_table(self, frames, track_idx=0, fill_missing=False):
        frames = self.check_frames(frames)

        locs = self.locs(fill_missing=fill_missing, track_idx=track_idx)[frames]
        coords = ["x", "y"]

        df = pd.DataFrame(locs.reshape(len(locs), -1))
        df.columns = pd.MultiIndex.from_product([self.bodyparts, coords])

        return df

    def ego_table(self, frames, track_idx=0, fill_missing=False):
        frames = self.check_frames(frames)

        locs = self.ego_locs(fill_missing=fill_missing, track_idx=track_idx)[frames]
        coords = ["x", "y"]

        df = pd.DataFrame(locs.reshape(len(locs), -1))
        df.columns = pd.MultiIndex.from_product(
            [[f"{bp}_aligned" for bp in self.bodyparts], coords]
        )

        return df

    def locs(self, track_idx=0, parts=None, fill_missing=True):
        tracks = self.tracks[..., track_idx]

        if parts is None:
            parts = self.bodyparts

        part_idx = [self.bodyparts.index(p) for p in parts]
        tracks = tracks[:, part_idx, ...]

        if fill_missing:
            return utils.fill_missing(tracks)

        return tracks

    def ego_locs(self, track_idx=0, parts=None, fill_missing=True):
        locations = self.locs(
            track_idx=track_idx, parts=parts, fill_missing=fill_missing
        )

        return self.aligner.transform(track_idx, locations)

    def ego_bbox(self, frame, track_idx=0, dest_height=100, dest_width=100):
        trans = self.aligner.transformations[track_idx][frame]

        dest_shape = [
            [-dest_width // 2, -dest_height // 2],
            [-dest_width // 2, dest_height // 2],
            [dest_width // 2, dest_height // 2],
            [dest_width // 2, -dest_height // 2],
        ]
        sbb_coords = trans.inverse(dest_shape)

        return sbb_coords

    @property
    def video_fps(self):
        fps = None
        if not self._vid_handle:
            self._vid_handle = cv2.VideoCapture(self.video_fn)
        fps = self._vid_handle.get(cv2.CAP_PROP_FPS)
        return fps

    def ego_image(self, frame, track_idx=0, dest_height=100, dest_width=100, **kwargs):
        if isinstance(frame, int):
            frames = range(frame, frame + 1)
        elif isinstance(frame, (list, tuple)) and len(frame) == 2:
            frames = range(frame[0], frame[1])
        else:
            raise RuntimeError("frame must be integer or list of [start, end]")

        if not self._vid_handle:
            self._vid_handle = cv2.VideoCapture(self.video_fn)

        self._vid_handle.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

        out_img = np.zeros((len(frames), dest_height, dest_width, 3), dtype="uint8")

        for k, frame in enumerate(frames):
            trans = self.aligner.transformations[track_idx][frame]
            _, in_img = self._vid_handle.read()

            if "mutant_side" in self.info and self.info["mutant_side"] == "right":
                print(
                    "DEBUG: Mutant side 'right' found in info file. Switching to left..."
                )
                # in_img = np.flipud(in_img)
                in_img = np.fliplr(in_img)

            out_img[k] = self.aligner.warp_image(
                in_img,
                trans,
                dest_height,
                dest_width,
            )

        return out_img.squeeze()

    def ego_plot(
        self, frame, parts=None, track_idx=0, dest_height=320, dest_width=160, ax=None
    ):
        ego_img = self.ego_image(
            frame=frame,
            track_idx=track_idx,
            dest_height=dest_height,
            dest_width=dest_width,
        )

        if ax is None:
            _, ax = matplotlib.pyplot.subplots()
        ax.imshow(
            ego_img,
            "gray",
            origin="lower",
            extent=[-dest_width / 2, dest_width / 2, -dest_height / 2, dest_height / 2],
        )
        if parts is not None:
            ego_locs = self.ego_locs(
                track_idx=track_idx, parts=tuple(parts), fill_missing=False
            )[frame]
            ax.plot(*ego_locs.T, ".-")
        return ax

    def ego_image_gen(self, frames=None, track_idx=0, dest_height=100, dest_width=100):
        frames = self.check_frames(frames)

        _vid_handle = cv2.VideoCapture(self.video_fn)

        _vid_handle.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

        for frame in frames:
            trans = self.aligner.transformations[track_idx][frame]
            _, in_img = _vid_handle.read()

            if "mutant_side" in self.info and self.info["mutant_side"] == "right":
                print(
                    "DEBUG: Mutant side 'right' found in info file. Switching to left..."
                )
                # in_img = np.flipud(in_img)
                in_img = np.fliplr(in_img)

            yield (
                frame,
                self.aligner.warp_image(
                    in_img,
                    trans,
                    dest_height,
                    dest_width,
                ),
            )

        _vid_handle.release()

    def bbox_image_gen(
        self,
        center_part,
        frames=None,
        track_idx=0,
        dest_height=100,
        dest_width=100,
        smooth_sigma=0,
    ):
        frames = self.check_frames(frames)

        _vid_handle = cv2.VideoCapture(self.video_fn)

        _vid_handle.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

        center_part_loc = self.locs(
            track_idx=track_idx,
            parts=(center_part,),
            fill_missing=True,
        )[:, 0, :]

        if smooth_sigma > 0:
            center_part_loc = utils.gaussian_filter1d(
                center_part_loc, smooth_sigma, axis=0
            )

        center_part_loc = np.round(center_part_loc).astype(int)

        h2 = dest_height // 2
        w2 = dest_width // 2

        for frame in frames:
            _, in_img = _vid_handle.read()

            if in_img.shape[-1] == 3:
                # BGR -> RGB
                in_img = in_img[..., ::-1]

            if "mutant_side" in self.info and self.info["mutant_side"] == "right":
                print(
                    "DEBUG: Mutant side 'right' found in info file. Switching to left..."
                )
                # in_img = np.flipud(in_img)
                in_img = np.fliplr(in_img)

            x, y = center_part_loc[frame]

            xa, xb = max(0, x - w2), min(in_img.shape[1] - 1, x + w2)
            ya, yb = max(0, y - h2), min(in_img.shape[0] - 1, y + h2)

            bbox_img = in_img[ya:yb, xa:xb]

            if bbox_img.shape[:2] == (dest_height, dest_width):
                yield frame, bbox_img
            else:
                bbox_img_pad = np.zeros(
                    (dest_height, dest_width) + in_img.shape[2:], dtype=in_img.dtype
                )
                bbox_img_pad[
                    max(0, h2 - y) : max(0, h2 - y) + bbox_img.shape[0],
                    max(0, w2 - x) : max(0, w2 - x) + bbox_img.shape[1],
                ] = bbox_img
                yield frame, bbox_img_pad

        _vid_handle.release()

    def image(self, frame, rgb=False):
        if not self._vid_handle:
            self._vid_handle = cv2.VideoCapture(self.video_fn)

        self._vid_handle.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, out_img = self._vid_handle.read()

        ### FIXME
        if "mutant_side" in self.info and self.info["mutant_side"] == "right":
            print("DEBUG: Mutant side 'right' found in info file. Switching to left...")
            out_img = np.fliplr(out_img)

        if not rgb:
            out_img = out_img[..., 0]

        return out_img

    def video_gif(self, frames=None, dest_width=150, dest_height=300, track_idx=0):
        import imageio
        from IPython.display import Image, display

        with imageio.get_writer(".temp.gif", mode="I") as writer:
            for _, image in self.ego_image_gen(
                frames,
                dest_width=dest_width,
                dest_height=dest_height,
                track_idx=track_idx,
            ):
                writer.append_data(image)

        with open(".temp.gif", "rb") as file:
            display(Image(file.read(), format="png"))

    def speed(self, part, frames=None, track_idx=0, pre_sigma=0, sigma=0):
        frames = self.check_frames(frames)

        part_loc = self.locs(parts=(part,), track_idx=track_idx).squeeze()[frames]

        if pre_sigma > 0:
            for c in range(2):
                part_loc[:, c] = utils.gaussian_filter1d(part_loc[:, c], pre_sigma)

        part_disp = np.gradient(part_loc, axis=0)
        speed = np.linalg.norm(part_disp, axis=1)

        if sigma > 0:
            speed = utils.gaussian_filter1d(speed, sigma)

        return speed

    def acceleration(self, part, frames=None, track_idx=0, sigma=0):
        frames = self.check_frames(frames)

        part_loc = self.locs(parts=(part,), track_idx=track_idx).squeeze()[frames]
        part_disp = np.gradient(part_loc, axis=0)
        part_disp2 = np.gradient(part_disp, axis=0)
        speed = np.linalg.norm(part_disp2, axis=1)

        if sigma > 0:
            speed = utils.gaussian_filter1d(speed, sigma)

        return speed

    def angles_from_segment(self, parts, frames=None, track_idx=0, win=None):
        angles = []
        for a, b, c in zip(parts, parts[1:], parts[2:]):
            ang = analysis.angles(
                self, (a, b), (b, c), win=win, track_idx=track_idx, frames=frames
            )
            angles.append(ang)

        return np.stack(angles, -1)

    def spline_curvature(
        self, parts, frames=None, track_idx=0, n_interpolants=64, spline_smooth=1
    ):
        frames = self.check_frames(frames)

        parts_positions = self.ego_locs(parts=tuple(parts), track_idx=track_idx)[frames]

        get_curvature = lambda points: SplineClass(
            points, n_interpolants, spline_smooth
        ).singed_curvature()

        return np.stack(list(map(get_curvature, parts_positions)))

    def spline_interpolate(
        self, parts, frames=None, track_idx=0, n_interpolants=64, spline_smooth=1
    ):
        frames = self.check_frames(frames)

        parts_positions = self.ego_locs(parts=tuple(parts), track_idx=track_idx)[frames]

        get_curvature = lambda points: SplineClass(
            points, n_interpolants, spline_smooth
        ).interpolate()

        return np.stack(list(map(get_curvature, parts_positions)))

    def parts_detected(self, parts=None, frames=None, track_idx=0):
        frames = self.check_frames(frames)

        if parts is None:
            parts = self.bodyparts

        parts_idx = [self.bodyparts.index(p) for p in parts]

        if track_idx is None:
            res = np.logical_not(
                np.isnan(self.tracks[frames][:, parts_idx, 0, :])
            ).squeeze()
        else:
            res = np.logical_not(
                np.isnan(self.tracks[frames][:, parts_idx, 0, track_idx])
            ).squeeze()

        return res

    @property
    def locations(self):
        warnings.warn(
            "\n\nPlease DO NOT use 'aligned_locations' and 'locations' anymore.\nWill be removed soon.\n Use 'tad.ego_locs() to get np array of aligned locations\n\n'"
        )
        tracks = self.tracks
        liklihoods = 1.0 - np.isnan(tracks[:, :, 0, self.current_track])

        tracks = tracks[..., self.current_track]

        parts = self.bodyparts
        coords = ["x", "y", "likelihood"]

        liklihoods = liklihoods[..., None]

        tracks = np.concatenate([tracks, liklihoods], axis=2)
        df = pd.DataFrame(tracks.reshape(len(tracks), -1))
        df.columns = pd.MultiIndex.from_product([parts, coords])

        return df

    @property
    def analysis_file(self):
        return f"{self.video_fn}.predictions.analysis.h5"

    def export_ego_movie(
        self,
        frames=None,
        shape=(224, 244),
        track_idx=0,
        suffix="aligned",
        out_fn=None,
        fps=30,
    ):
        from .utils import VideoProcessorCV as vp

        frames = self.check_frames(frames)

        if out_fn is None:
            out_mov = f"{self.vid_path}/{self.vid_fn}_{track_idx:02}_{suffix}.mp4"
        else:
            out_mov = out_fn
        print(f"Export to: {out_mov}")

        dest_height, dest_width = shape
        clip = vp(sname=out_mov, codec="mp4v", sw=dest_width, sh=dest_height, fps=fps)

        for _, img in tqdm(
            self.ego_image_gen(frames, track_idx, dest_height, dest_width),
            total=len(frames),
        ):
            clip.save_frame(np.rot90(img, k=2))
        clip.close()

    def social_receptive_field(
        self,
        focal_track_idx=0,
        srf_bins=None,
        srf_size=None,
        other_part=None,
    ):
        if len(self) == 1:
            raise RuntimeError("SRF analysis requires at least 2 animals.")
        if self.aligner is None:
            raise RuntimeError(
                "SRF analysis requires an aligner. Please set the aligner first."
            )
        from tadpose.analysis import SocialReceptiveField

        return SocialReceptiveField(
            self, focal_track_idx, srf_bins, srf_size, other_part
        )


class DeeplabcutTadpole(SleapTadpole):
    def __init__(self, video_fn, bodyparts_cmap):
        Tadpole.__init__(self, video_fn, bodyparts_cmap)

        vid_p = Path(video_fn)
        dlc_h5_fn = str(next((vid_p.parent).glob(f"{vid_p.stem}*.h5")))

        dlc_tab = pd.read_hdf(dlc_h5_fn)
        nframes = len(dlc_tab)

        self.scorer = dlc_tab.columns.get_level_values(0).unique().to_list()[0]

        self.track_names = dlc_tab.columns.get_level_values(1).unique().to_list()

        self._bodyparts = dlc_tab.columns.get_level_values(2).unique().to_list()

        self.tracks = np.zeros(
            (nframes, len(self._bodyparts), 2, len(self.track_names)), dtype=np.float32
        )

        for ti, t in enumerate(self.track_names):
            for bi, b in enumerate(self._bodyparts):
                x_coords = dlc_tab[self.scorer][t][b]["x"]
                y_coords = dlc_tab[self.scorer][t][b]["y"]
                self.tracks[:, bi, 0, ti] = x_coords
                self.tracks[:, bi, 1, ti] = y_coords

        self.current_track = 0


class BatchGrouper:
    def __init__(self, exp_table, aligner, output_grouped_by="Stage"):
        if isinstance(exp_table, str):
            self.exp_table = pd.read_csv(exp_table, sep="\t")
        else:
            self.exp_table = exp_table

        self.aligner = aligner
        self.output_grouped_by = output_grouped_by

    def __len__(self):
        return len(self.exp_table)

    def __iter__(self):
        for grp, df_grp in self.exp_table.groupby(self.output_grouped_by):
            for ind, row in df_grp.iterrows():
                tadpole = Tadpole.from_sleap(
                    row["FullPath"],
                )
                tadpole.aligner = self.aligner

                yield tadpole, grp, ind, row
