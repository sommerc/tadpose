import os
import cv2
import h5py
import warnings
import matplotlib
import numpy as np
import pandas as pd

from tqdm.auto import tqdm, trange
from functools import lru_cache

from . import utils


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

    @staticmethod
    def from_dlc(video_fn, scorer, bodyparts_cmap="jet"):
        return DeeplabcutTadpole(
            video_fn=video_fn, scorer=scorer, bodyparts_cmap=bodyparts_cmap
        )

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
        for track_idx in trange(len(self), desc="Aligning animals"):
            all_locations = self.locs(track_idx=track_idx)
            ta.fit(track_idx, self.bodyparts, all_locations)
        self._aligner = ta

    @property
    @lru_cache()  ### TOTO proper cache for multi-animal
    def aligned_locations(self):
        warnings.warn(
            "\n\nPlease DO NOT use 'aligned_locations' and 'locations' anymore.\nWill be removed soon.\n Use 'tad.ego_locs() to get np array of aligned locations\n\n'"
        )
        return self._aligner.align(self)


class SleapTadpole(Tadpole):
    def __init__(self, video_fn, bodyparts_cmap, **kwargs):
        super().__init__(video_fn, bodyparts_cmap)
        self.scorer = ""

        with h5py.File(self.analysis_file, "r") as hf:
            self.tracks = hf["tracks"][:].T
            self._bodyparts = list(map(lambda x: x.decode(), hf["node_names"]))

        self.current_track = 0

    def __len__(self):
        return self.tracks.shape[-1]

    @property
    def nframes(self):
        return len(self.tracks)

    @lru_cache()
    def __getitem__(self, track_idx):
        assert track_idx < len(self), "track does not exist, go away"
        tracks = self.tracks[..., track_idx]
        liklihoods = 1.0 - np.isnan(self.tracks[:, :, 0, track_idx])

        coords = ["x", "y", "likelihood"]

        liklihoods = liklihoods[..., None]

        tracks = np.concatenate([tracks, liklihoods], axis=2)
        df = pd.DataFrame(tracks.reshape(len(tracks), -1))
        df.columns = pd.MultiIndex.from_product([self.bodyparts, coords])

        return df

    @lru_cache()
    def locs(self, track_idx=0, parts=None, fill_missing=True):
        tracks = self.tracks[..., track_idx]

        if parts is None:
            parts = self.bodyparts

        part_idx = [self.bodyparts.index(p) for p in parts]
        tracks = tracks[:, part_idx, ...]

        if fill_missing:
            return utils.fill_missing(tracks)

        return tracks

    @lru_cache()
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

    def ego_image(self, frame, track_idx=0, dest_height=100, dest_width=100, **kwargs):
        if isinstance(frame, int):
            frames = range(frame, frame + 1)
        elif isinstance(frame, (list, tuple)) and len(frame) == 2:
            frames = range(frame[0], frame[1])
        else:
            raise RuntimeError("frame must be integer or list of [start, end]")

        if not self._vid_handle:
            self._vid_handle = cv2.VideoCapture(self.video_fn)

        self._vid_handle.set(cv2.cv2.CAP_PROP_POS_FRAMES, frames[0])

        out_img = np.zeros((len(frames), dest_height, dest_width, 3), dtype="uint8")

        for k, frame in enumerate(frames):
            trans = self.aligner.transformations[track_idx][frame]
            _, in_img = self._vid_handle.read()

            out_img[k] = self.aligner.warp_image(
                in_img, trans, dest_height, dest_width,
            )

        return out_img.squeeze()
        return np.rot90(out_img.squeeze(), k=2)

    def ego_image_gen(self, frames=None, track_idx=0, dest_height=100, dest_width=100):
        if frames is None:
            frames = range(self.nframes)
        elif isinstance(frames, (list, tuple)) and len(frames) == 2:
            frames = range(frames[0], frames[1])
        else:
            raise RuntimeError("Frames must be integer or list of [start, end]")

        if not self._vid_handle:
            self._vid_handle = cv2.VideoCapture(self.video_fn)

        self._vid_handle.set(cv2.cv2.CAP_PROP_POS_FRAMES, frames[0])

        for frame in frames:
            trans = self.aligner.transformations[track_idx][frame]
            _, in_img = self._vid_handle.read()

            yield self.aligner.warp_image(
                in_img, trans, dest_height, dest_width,
            )

    def image(self, frame, rgb=False):
        if not self._vid_handle:
            self._vid_handle = cv2.VideoCapture(self.video_fn)

        self._vid_handle.set(cv2.cv2.CAP_PROP_POS_FRAMES, frame)
        res, out_img = self._vid_handle.read()

        if not rgb:
            out_img = out_img[..., 0]

        return out_img

    def video_gif(self, frames=None, dest_width=150, dest_height=300):
        import imageio
        from IPython.display import Image, display

        with imageio.get_writer(".temp.gif", mode="I") as writer:
            for image in self.ego_image_gen(
                frames, dest_width=dest_width, dest_height=dest_height
            ):
                writer.append_data(image)

        with open(".temp.gif", "rb") as file:
            display(Image(file.read(), format="png"))

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
        self, frames=None, shape=(224, 244), track_idx=0, suffix="aligned",
    ):
        from .utils import VideoProcessorCV as vp

        out_mov = f"{self.vid_path}/{self.vid_fn}_{track_idx:02}_{suffix}.mp4"
        print(f"Export to: {out_mov}")

        dest_height, dest_width = shape
        clip = vp(sname=out_mov, codec="mp4v", sw=dest_width, sh=dest_height)

        if frames is None:
            frames = [0, len(self.tracks)]

        for img in tqdm(
            self.ego_image_gen(frames, track_idx, dest_height, dest_width),
            total=frames[1] - frames[0],
        ):
            clip.save_frame(np.rot90(img, k=2))
        clip.close()


class DeeplabcutTadpole(Tadpole):
    def __init__(self, video_fn, bodyparts_cmap, scorer, **kwargs):
        super().__init__(video_fn, bodyparts_cmap)
        self.scorer = scorer

    @property
    @lru_cache()
    def locations(self):
        return pd.read_hdf(self.analysis_file)[self.scorer]

    @property
    @lru_cache()
    def bodyparts(self):
        return self.locations.columns.get_level_values(0).unique().tolist()

    @property
    def analysis_file(self):
        return f"{self.vid_path}/{self.vid_fn}{self.scorer}.h5"

