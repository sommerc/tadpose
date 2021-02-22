from . import alignment
from . import analysis
from . import visu
from . import utils

import os
import numpy
import pandas
import matplotlib
from functools import lru_cache


def select_dlc_config(ext=".yaml"):
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


class Tadpole:
    def __init__(self, video_fn, scorer, dlc_config):
        assert os.path.exists(video_fn), f"Movie file '{video_fn}' does not exist"
        self.video_fn = os.path.abspath(video_fn)

        self.vid_path = os.path.dirname(video_fn)
        self.vid_fn = os.path.basename(video_fn)
        self.vid_fn, self.vid_ext = os.path.splitext(self.vid_fn)

        self.scorer = scorer
        self.dlc_config = dlc_config

        self.bodyparts = dlc_config["bodyparts"]

        colorclass = matplotlib.cm.ScalarMappable(cmap="jet")
        C = colorclass.to_rgba(numpy.linspace(0, 1, len(self.bodyparts)))
        self.bodypart_colors = (C[:, :3] * 255).astype(numpy.uint8)
        self.bodypart_color = dict(
            [(k, v / 255.0) for k, v in zip(self.bodyparts, self.bodypart_colors)]
        )

        self.aligner = None

    @property
    def aligner(self):
        return self._aligner

    @aligner.setter
    def aligner(self, ta):
        self._aligner = ta

    @property
    @lru_cache()
    def locations(self):
        return pandas.read_hdf(f"{self.vid_path}/{self.vid_fn}{self.scorer}.h5")[
            self.scorer
        ]

    @property
    @lru_cache()
    def aligned_locations(self):
        Cs, Rs, Ts = self._aligner.estimate_allign(self.locations)
        return self.aligner.allign(self.locations, Cs, Rs, Ts)

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
