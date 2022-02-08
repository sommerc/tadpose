import os
import numpy as np
import pandas as pd
import seaborn as sns
from tadpose import utils, analysis
from matplotlib import pyplot as plt
from matplotlib.colors import is_color_like, Normalize
from matplotlib import cm
import os
import re
import glob
import pandas
import traceback
import tkinter as tk
from datetime import datetime
from functools import partial
from tkinter import filedialog
from dataclasses import dataclass, field
from typing import Union, Tuple, List
from itertools import count

coords = ["x", "y"]


def dish(
    tadpole,
    part=None,
    alpha=1.0,
    color=None,
    ax=None,
    ls="-",
    size=1,
    lw=1.0,
    speed_cmap="jet",
):

    if ax is None:
        _, ax = plt.subplots()

    if part is None:
        part = tadpole.bodyparts[0]

    xy = tadpole.locs(parts=(part,)).squeeze()

    if color is None:
        color = tadpole.bodypart_color[part]

    if is_color_like(color):
        ax.plot(*xy.T, ls=ls, color=color, alpha=alpha, lw=lw)
    elif color == "speed":
        speed = analysis.speeds(tadpole, parts=(part,))
        ax.scatter(
            xy[:, 0], xy[:, 1], c=speed.to_numpy(), s=size, cmap=speed_cmap, alpha=alpha
        )

    ax.set_aspect(1.0)

    return ax


def bodyparts(
    tadpole,
    parts=None,
    alpha=0.02,
    N_max=None,
    ax=None,
    contour=False,
    scatter=True,
    lines=True,
    bbox=None,
    bbox_perc=99.42,
):
    if ax is None:
        _, ax = plt.subplots()

    if parts is None:
        parts = tadpole.bodyparts

    xy = tadpole.ego_locs(parts=tuple(parts))

    if N_max is not None:
        xy = xy[np.random.permutation(xy.shape[0])]
        xy = xy[:N_max, ...]

    for pi, p in enumerate(parts):

        if scatter:
            ax.plot(*xy[:, pi].T, ".", color=tadpole.bodypart_color[p], alpha=alpha)

        if lines:
            ax.plot(*xy[:, pi].T, "-", color=tadpole.bodypart_color[p], alpha=alpha)

        if contour:
            sns.kdeplot(*xy[:, pi].T, color=tadpole.bodypart_color[p], levels=5)

        if True:
            ax.plot(
                np.median(xy[:, pi, 0]),
                np.median(xy[:, pi, 1]),
                ".",
                color=tadpole.bodypart_color[p],
                markersize=8,
                markeredgecolor="k",
                markeredgewidth=0.3,
            )

    max_perc = bbox_perc
    min_perc = 100 - max_perc

    if bbox is None:
        bbox = (
            np.percentile(xy[..., 0].flatten(), min_perc),
            np.percentile(xy[..., 0].flatten(), max_perc),
            np.percentile(xy[..., 1].flatten(), min_perc),
            np.percentile(xy[..., 1].flatten(), max_perc),
        )
    x1, x2, y1, y2 = bbox

    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    ax.hlines(0, xmin=x1, xmax=x2, color="gray", linestyle=":")
    ax.vlines(0, ymin=y1, ymax=y2, color="gray", linestyle=":")

    ax.invert_xaxis()
    ax.set_aspect(1.0)
    ax.set_xticklabels([])
    sns.despine(ax=ax)

    plt.tight_layout()

    return ax


@dataclass
class HeatMap:
    id: int = field(default_factory=count().__next__, init=False)
    filetype: str = ".png"
    bodyparts_cmap: str = None
    parts: Tuple[str] = None
    N_max: int = None
    alpha: float = 0.02
    contour: bool = False
    scatter: bool = True
    lines: bool = True
    bbox_perc: float = 100
    bbox: Tuple[int, int, int, int] = None

    def __call__(self, tad, row_info, outdir):
        if self.bodyparts_cmap is not None:
            tad.bodyparts_cmap = self.bodyparts_cmap

        if self.parts is not None:
            self.parts = tuple(self.parts)

        ax = bodyparts(
            tad,
            parts=self.parts,
            N_max=self.N_max,
            alpha=self.alpha,
            contour=self.contour,
            scatter=self.scatter,
            lines=self.lines,
            bbox_perc=self.bbox_perc,
            bbox=self.bbox,
        )
        ax.get_figure().savefig(
            f"{outdir}/HeatMap{self.id}_{row_info['File']}{self.filetype}"
        )


@dataclass
class SkeletonMap:
    id: int = field(default_factory=count().__next__, init=False)
    filetype: str = ".png"
    color: str = "r"
    parts: Tuple[str] = None
    N_max: int = 500
    alpha: float = 0.02
    ls: str = "-"
    lw: float = 1.0
    ego: bool = True
    vmin: int = None
    vmax: int = None

    def __call__(self, tad, row_info, outdir):
        if self.parts is not None:
            self.parts = tuple(self.parts)

        if self.ego:
            xy = tad.ego_locs(parts=self.parts)
        else:
            xy = tad.locs(parts=self.parts)

        perm = np.random.permutation(xy.shape[0])
        xy = xy[perm]
        xy = xy[: self.N_max, ...]

        f, ax = plt.subplots()

        if self.color == "speed":
            if self.ego:
                node_velocities = analysis.ego_speeds(tad, parts=tuple(self.parts))
            else:
                node_velocities = analysis.speeds(tad, parts=tuple(self.parts))

            node_velocities = node_velocities.iloc[perm].iloc[: self.N_max].to_numpy()

            if self.vmin is None:
                self.vmin = node_velocities.min()

            if self.vmax is None:
                self.vmax = node_velocities.max()

            norm = Normalize(vmin=self.vmin, vmax=self.vmax)

        for i, skel in enumerate(xy):
            if self.color == "speed":
                ax.plot(
                    skel[:, 0],
                    skel[:, 1],
                    ls=self.ls,
                    lw=self.lw,
                    alpha=self.alpha,
                    color=cm.jet(norm(node_velocities[i].max())),
                )

            else:
                ax.plot(
                    skel[:, 0],
                    skel[:, 1],
                    ls=self.ls,
                    lw=self.lw,
                    alpha=self.alpha,
                    color=self.color,
                )

        ax.set_aspect(1.0)

        ax.get_figure().savefig(
            f"{outdir}/SkeletonMap{self.id}_{row_info['File']}{self.filetype}"
        )


@dataclass
class DishMovement:
    part: str
    id: int = field(default_factory=count().__next__, init=False)
    filetype: str = ".png"
    size: float = 1.0
    color: str = "orange"
    ls: str = "-"
    lw: float = 1.0
    alpha: float = 1.0
    speed_cmap: str = "jet"

    def __call__(self, tad, row_info, outdir):
        ax = dish(
            tad,
            part=self.part,
            color=self.color,
            ls=self.ls,
            size=self.size,
            alpha=self.alpha,
            lw=self.lw,
            speed_cmap=self.speed_cmap,
        )
        ax.get_figure().savefig(
            f"{outdir}/DishMovement{self.id}_{row_info['File']}{self.filetype}"
        )


@dataclass
class SpeedBox:
    id: int = field(default_factory=count().__next__, init=False)
    parts: Tuple[str] = None
    filetype: str = ".png"
    cmap: str = None
    ego: bool = True
    figsize: Tuple[int, int] = None
    vmin: int = None
    vmax: int = None
    xlim: Tuple[int, int] = (None, None)

    def __call__(self, tad, row_info, outdir):

        if self.parts is not None:
            self.parts = tuple(self.parts)

        if self.ego:
            node_velocities = analysis.ego_speeds(tad, parts=tuple(self.parts))
        else:
            node_velocities = analysis.speeds(tad, parts=tuple(self.parts))

        node_velocities = node_velocities[self.xlim[0] : self.xlim[1]]

        if self.parts is None:
            node_names = tad.bodyparts
        else:
            node_names = self.parts

        if self.cmap is None:
            cmap = tad.bodyparts_cmap

        f, ax = plt.subplots(figsize=self.figsize)

        sns.boxplot(data=node_velocities, palette=self.cmap, ax=ax, showfliers=False)

        ax.set_ylim((self.vmin, self.vmax))

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        plt.tight_layout()

        ax.get_figure().savefig(
            f"{outdir}/SpeedBox{self.id}_{row_info['File']}{self.filetype}"
        )


@dataclass
class SpeedMap:

    id: int = field(default_factory=count().__next__, init=False)
    parts: Tuple[str] = None
    filetype: str = ".png"
    cmap: str = "gray_r"
    figsize: Tuple[int, int] = None
    vmin: int = None
    vmax: int = None
    xlim: Tuple[int, int] = (None, None)
    ego: bool = True

    def __call__(self, tad, row_info, outdir):

        if self.parts is not None:
            self.parts = tuple(self.parts)

        if self.ego:
            node_velocities = analysis.ego_speeds(tad, parts=self.parts)
        else:
            node_velocities = analysis.speeds(tad, parts=self.parts)

        if self.parts is None:
            node_names = tad.bodyparts
        else:
            node_names = self.parts
        node_count = len(node_names)

        f, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(
            node_velocities.T,
            aspect="auto",
            vmin=self.vmin,
            vmax=self.vmax,
            interpolation="nearest",
            cmap=self.cmap,
        )
        ax.set_xlabel("Time (Frames)")
        ax.set_ylabel("Parts")
        ax.set_yticks(np.arange(node_count))
        ax.set_yticklabels(node_names)
        ax.set_xlim(self.xlim)

        f.colorbar(im)
        plt.tight_layout()

        ax.get_figure().savefig(
            f"{outdir}/SpeedMap{self.id}_{row_info['File']}{self.filetype}"
        )


if __name__ == "__main__":
    pass
