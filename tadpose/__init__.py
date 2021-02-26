import os
import h5py
import numpy
import pandas
import matplotlib
from functools import lru_cache

from . import alignment
from . import analysis
from . import visu
from . import utils


def file_select_dialog(ext=".yaml"):
    import os
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_name = filedialog.askopenfilename(
        defaultextension=ext, filetypes=[(f"{ext} files", f"*{ext}")], parent=root,
    )
    root.destroy()

    if len(file_name) == 0:
        print("No DLC config file selected")

    return file_name


class TadpoleAnalysis:
    BACKENDS = ["deeplabcut", "sleap"]

    def __init__(self, backend):
        if backend not in self.BACKENDS:
            raise RuntimeError(
                f"Backend '{backend}' not supported. Valid backands are {BACKENDS}"
            )
        self.backend = backend

    def __call__(self, *args, **kwargs):
        if self.backend == "deeplabcut":
            return DeeplabcutTadpole(*args, **kwargs)
        if self.backend == "sleap":
            return SleapTadpole(*args, **kwargs)


class Tadpole:
    def __init__(self, video_fn, bodyparts, bodyparts_cmap="jet"):
        assert os.path.exists(video_fn), f"Movie file '{video_fn}' does not exist"

        self.video_fn = os.path.abspath(video_fn)
        self.vid_path = os.path.dirname(video_fn)
        self.vid_fn = os.path.basename(video_fn)
        self.vid_fn, self.vid_ext = os.path.splitext(self.vid_fn)

        self.bodyparts_cmap = bodyparts_cmap
        self._aligner = None

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
    @lru_cache()
    def aligned_locations(self):
        return self._aligner.align(self)

    def export_aligned_movie(
        self, dest_height, dest_width, aligned_suffix="aligned", **kwargs
    ):
        self.aligner.export_movie(
            self.locations,
            self.video_fn,
            f"{self.vid_path}/{self.vid_fn}{self.scorer}_{aligned_suffix}.mp4",
            dest_height=dest_height,
            dest_width=dest_width,
            **kwargs,
        )


class SleapTadpole(Tadpole):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

    @property
    @lru_cache()
    def locations(self):
        with h5py.File(self.analysis_file, "r") as hf:
            tracks = hf["tracks"][:].T
            liklihoods = 1.0 - numpy.isnan(tracks[:, :, 0, 0])

            tracks = utils.fill_missing(tracks)[..., 0]

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

    @property
    @lru_cache()
    def bodyparts(self):
        with h5py.File(self.analysis_file, "r") as hf:
            return list(map(lambda x: x.decode(), hf["node_names"]))


class DeeplabcutTadpole(Tadpole):
    def __init__(self, *args, scorer, **kwargs):
        super().__init__(*args)
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

