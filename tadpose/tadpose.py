import os
import cv2
import h5py
import numpy
import pandas
import matplotlib

from . import visu
from . import utils
from . import analysis
from . import alignment

from functools import lru_cache






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
        tmp = (
            self.locations[self.bodyparts]
            .to_numpy()
            .reshape(-1, len(self.bodyparts), 3)
        )
        return tmp[..., :2], tmp[..., 2]

    @property
    def bodyparts(self):
        pass

    @bodyparts.setter
    def bodyparts(self, bodyparts):
        self._bodyparts = bodyparts

    @bodyparts.getter
    def bodyparts(self):
        return self._bodyparts

    @property
    def bodypart_colors(self):
        colorclass = matplotlib.cm.ScalarMappable(cmap=self.bodyparts_cmap)
        C = colorclass.to_rgba(numpy.linspace(0, 1, len(self.bodyparts)))
        return (C[:, :3] * 255).astype(numpy.uint8)

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
        self._aligner = ta

    @property
    @lru_cache()  ### TOTO proper cache for multi-animal
    def aligned_locations(self):
        return self._aligner.align(self)

    def export_aligned_movie(
        self, dest_height, dest_width, aligned_suffix="aligned", **kwargs
    ):
        out_mov = f"{self.vid_path}/{self.vid_fn}{self.scorer}_{aligned_suffix}.mp4"
        print(f"Export to: {out_mov}")
        self.aligner.export_movie(
            self,
            self.video_fn,
            out_mov,
            dest_height=dest_height,
            dest_width=dest_width,
            **kwargs,
        )

    # @lru_cache()
    def aligned_image(self, frame, dest_height=100, dest_width=100, rgb=False):
        Cs, Rs, Ts = self.aligner.estimate_alignment(self, frame=frame)

        trans = self.aligner._get_transformation(Cs[0], Rs[0], Ts[0])

        if not self._vid_handle:
            self._vid_handle = cv2.VideoCapture(self.video_fn)

        self._vid_handle.set(cv2.cv2.CAP_PROP_POS_FRAMES, frame)
        res, in_img = self._vid_handle.read()

        out_img = self.aligner.warp_image(
            in_img, trans, (dest_height, dest_width), rgb=rgb
        )

        if not rgb:
            out_img = out_img[..., 0]

        return numpy.rot90(out_img, k=2)


class SleapTadpole(Tadpole):
    def __init__(self, video_fn, bodyparts_cmap, **kwargs):
        super().__init__(video_fn, bodyparts_cmap)
        self.scorer = ""

        with h5py.File(self.analysis_file, "r") as hf:
            self.tracks = hf["tracks"][:].T
            self.bodyparts = list(map(lambda x: x.decode(), hf["node_names"]))

        self.current_track = 0

    def __len__(self):
        return self.tracks.shape[-1]

    @lru_cache()
    def __getitem__(self, track_idx):
        assert track_idx < len(self), "track does not exist, go away"
        tracks = self.tracks[..., track_idx]
        liklihoods = 1.0 - numpy.isnan(self.tracks[:, :, 0, track_idx])

        coords = ["x", "y", "likelihood"]

        liklihoods = liklihoods[..., None]

        tracks = numpy.concatenate([tracks, liklihoods], axis=2)
        df = pandas.DataFrame(tracks.reshape(len(tracks), -1))
        df.columns = pandas.MultiIndex.from_product([self.bodyparts, coords])

        return df

    @lru_cache()
    def locs(self, track_idx=0, parts=None, fill_missing=True):
        tracks = self.tracks

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
        all_locations = self.locs(
            track_idx=track_idx, parts=None, fill_missing=fill_missing
        )  # get all parts

        all_aligned_locations = self.aligner.new_align(self.bodyparts, all_locations)

        if parts is None:
            parts = self.bodyparts
        part_idx = [self.bodyparts.index(p) for p in parts]
        return all_aligned_locations[:, part_idx, ...]

    @lru_cache()
    def ego_image(self, frame, track_idx=0, dest_height=100, dest_width=100, rgb=False):
        location = self.locs(track_idx)[frame : frame + 1, ...]
        Cs, Rs, Ts = self.aligner.compute_alignment_matrices(self.bodyparts, location)

        trans = self.aligner._get_transformation(Cs[0], Rs[0], Ts[0])

        if not self._vid_handle:
            self._vid_handle = cv2.VideoCapture(self.video_fn)

        self._vid_handle.set(cv2.cv2.CAP_PROP_POS_FRAMES, frame)
        res, in_img = self._vid_handle.read()

        out_img = self.aligner.warp_image(
            in_img, trans, (dest_height, dest_width), rgb=rgb
        )

        if not rgb:
            out_img = out_img[..., 0]

        return numpy.rot90(out_img, k=2)

    @property
    def locations(self):
        tracks = self.tracks
        liklihoods = 1.0 - numpy.isnan(tracks[:, :, 0, self.current_track])

        tracks = tracks[..., self.current_track]

        parts = self.bodyparts
        coords = ["x", "y", "likelihood"]

        liklihoods = liklihoods[..., None]

        tracks = numpy.concatenate([tracks, liklihoods], axis=2)
        df = pandas.DataFrame(tracks.reshape(len(tracks), -1))
        df.columns = pandas.MultiIndex.from_product([parts, coords])

        return df

    @property
    def analysis_file(self):
        return f"{self.video_fn}.predictions.analysis.h5"

    def export_aligned_movie(
        self, dest_height, dest_width, aligned_suffix="aligned", **kwargs
    ):
        out_mov = f"{self.vid_path}/{self.vid_fn}_{self.current_track:02}_{aligned_suffix}.mp4"
        print(f"Export to: {out_mov}")
        self.aligner.export_movie(
            self,
            self.video_fn,
            out_mov,
            dest_height=dest_height,
            dest_width=dest_width,
            **kwargs,
        )


class DeeplabcutTadpole(Tadpole):
    def __init__(self, video_fn, bodyparts_cmap, scorer, **kwargs):
        super().__init__(video_fn, bodyparts_cmap)
        self.scorer = scorer

    @property
    @lru_cache()
    def locations(self):
        return pandas.read_hdf(self.analysis_file)[self.scorer]

    @property
    @lru_cache()
    def bodyparts(self):
        return self.locations.columns.get_level_values(0).unique().tolist()

    @property
    def analysis_file(self):
        return f"{self.vid_path}/{self.vid_fn}{self.scorer}.h5"

