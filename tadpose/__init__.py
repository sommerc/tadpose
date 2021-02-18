from . import alignment
from . import analysis
from . import visu
from . import utils

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

    if len(file_name) == 0: print("No DLC config file selected")

    return file_name


class Tadpole:
    def __init__(self, path, vid_fn, scorer, dlc_config):
        self.path = path
        self.vid_fn = vid_fn
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
        return pandas.read_hdf(f"{self.path}/{self.vid_fn}{self.scorer}.h5")[
            self.scorer
        ]

    @property
    @lru_cache()
    def aligned_locations(self):
        Cs, Rs, Ts = self._aligner.estimate_allign(self.locations)
        return self.aligner.allign(self.locations, Cs, Rs, Ts)

