import os
import glob
import pandas
import numpy

np = numpy

from deeplabcut.utils import read_config
from deeplabcut.utils.video_processor import VideoProcessorCV as vp

from tqdm.auto import tqdm
from skimage.draw import disk, line
from skimage import transform as st
from skimage import io
from matplotlib import pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.ndimage import map_coordinates

coords = ["x", "y"]

def plot_limbs(df, ta, row=None, alpha=0.02, ax=None, contour=False, scatter=True):
    if ax is None:
        f, ax = plt.subplots()

    xR, yR = df["LimbR"][coords].to_numpy().T
    xL, yL = df["LimbL"][coords].to_numpy().T

    if scatter:
        ax.plot(xR, yR, ".", color=ta.bodypart_color["LimbR"], alpha=alpha)
        ax.plot(xL, yL, ".", color=ta.bodypart_color["LimbL"], alpha=alpha)

    if contour:
        sns.kdeplot(x=xR, y=yR, color=ta.bodypart_color["LimbR"], levels=5)
        sns.kdeplot(x=xL, y=yL, color=ta.bodypart_color["LimbL"], levels=5)

    x1, x2, y1, y2 = -120, 120, -220, 40

    if row is not None:
        ax.set_title(f"Limbs: Movie {row.id} ({row.genotype})")
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    ax.hlines(0, xmin=x1, xmax=x2, color="gray", linestyle=":")
    ax.vlines(0, ymin=y1, ymax=y2, color="gray", linestyle=":")

    for bi, bp in enumerate(ta.bodyparts):
        p = df[bp][["x", "y"]].median()
        ax.plot(
            p.x, p.y, ".", color=tuple(ta.bodypart_colors[bi] / 255.0), markersize=8, markeredgecolor="k", markeredgewidth=0.3
        )

    ax.invert_xaxis()
    ax.set_aspect(1.0)
    ax.set_xticklabels([])
    sns.despine(ax=ax)

    plt.tight_layout()

    return ax


def plot_tail(df, ta, row, alpha=0.01, ax=None, scatter=True, contour=True):
    if ax is None:
        f, ax = plt.subplots()

    xR, yR = df["TailCenter"][coords].to_numpy().T
    xL, yL = df["TailTip"][coords].to_numpy().T

    if scatter:
        ax.plot(xR, yR, ".", color=ta.bodypart_color["TailCenter"], alpha=alpha)
        ax.plot(xL, yL, ".", color=ta.bodypart_color["TailTip"], alpha=alpha)

    # ax.plot(xR, yR, "g-", alpha=0.5)
    # ax.plot(xL, yL, "r-", alpha=0.5)

    if contour:
        sns.kdeplot(x=xR, y=yR, color=ta.bodypart_color["TailCenter"], levels=5)
        sns.kdeplot(x=xL, y=yL, color=ta.bodypart_color["TailTip"], levels=5)

    x1, x2, y1, y2 = -250, 250, -600, 40

    if row is not None:
        ax.set_title(f"Tail: Movie {row.id} ({row.genotype})")
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    ax.hlines(0, xmin=x1, xmax=x2, color="gray", linestyle=":")
    ax.vlines(0, ymin=y1, ymax=y2, color="gray", linestyle=":")

    for bi, bp in enumerate(ta.bodyparts):
        p = df[bp][["x", "y"]].median()
        ax.plot(
            p.x, p.y, ".", color=tuple(ta.bodypart_colors[bi] / 255.0), markersize=10, markeredgecolor="k", markeredgewidth=0.3,
        )

    ax.invert_xaxis()
    ax.set_aspect(1.0)
    ax.set_xticklabels([])
    sns.despine(ax=ax)

    plt.tight_layout()

    return ax