import os
import glob
import pandas
import numpy

np = numpy


from tqdm.auto import tqdm
from skimage.draw import disk, line
from skimage import transform as st
from skimage import io
from matplotlib import pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.ndimage import map_coordinates

coords = ["x", "y"]


def plot_parts(
    tadpole,
    parts=None,
    alpha=0.02,
    ax=None,
    contour=False,
    scatter=True,
    lines=True,
    bbox=(-150, 150, -300, 100),
):
    if ax is None:
        _, ax = plt.subplots()

    if parts is None:
        parts = tadpole.bodyparts

    for p in parts:
        xy = tadpole.ego_locs(parts=(p,)).squeeze()

        if scatter:
            ax.plot(*xy.T, ".", color=tadpole.bodypart_color[p], alpha=alpha)

        if lines:
            ax.plot(*xy.T, "-", color=tadpole.bodypart_color[p], alpha=alpha)

        if contour:
            sns.kdeplot(*xy.T, color=tadpole.bodypart_color[p], levels=5)

        if True:
            ax.plot(
                numpy.median(xy[:, 0]),
                numpy.median(xy[:, 1]),
                ".",
                color=tadpole.bodypart_color[p],
                markersize=8,
                markeredgecolor="k",
                markeredgewidth=0.3,
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
