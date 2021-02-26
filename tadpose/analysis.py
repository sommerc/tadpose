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

from functools import reduce

coords = ["x", "y"]


def get_valid_alignment_idx(df, max_std=1):
    return numpy.abs(stats.zscore(df["Center"].y)) < max_std


def get_valid_alignment_idx(df, max_std=1):
    return numpy.abs(stats.zscore(df["Center"].y)) < max_std


def get_displacements(df, part, coords=("x", "y")):
    return (df[part][list(coords)].diff(1) ** 2).sum(1, skipna=False).apply(numpy.sqrt)


def get_displacements(df, part, coords=("x", "y")):
    return (df[part][list(coords)].diff(1) ** 2).sum(1, skipna=False).apply(numpy.sqrt)


def get_speed(df, part, coords=("x", "y"), per="frame"):
    displ = get_displacements(df, part, coords)
    dt = df["Time"][per].diff(1)
    res = displ / dt
    res.iloc[0] = res.iloc[1]
    return res


def get_angle(df_flt_tail, part1="TailTip", part2="TailCenter"):
    v0 = df_flt_tail[part1][coords].to_numpy() - df_flt_tail[part2][coords].to_numpy()
    v1 = df_flt_tail[part2][coords].to_numpy()

    v0 = v0 / numpy.linalg.norm(v0, axis=1)[:, None]
    v1 = v1 / numpy.linalg.norm(v1, axis=1)[:, None]

    c = (v0 * v1).sum(1)
    angles = numpy.arccos(numpy.clip(c, -1, 1))
    return numpy.rad2deg(angles)


def filter_likelihood(df, parts, min_likelihood):
    sel = reduce(numpy.logical_and, [df[p].likelihood > min_likelihood for p in parts])

    return df[sel]


def filter_misdetections_limbs(df, likelihood=0.01, max_displ=50):
    # by likelihood
    df_res = df[
        (df["Center"].likelihood > likelihood)
        & (df["LimbR"].likelihood > likelihood)
        & (df["LimbL"].likelihood > likelihood)
    ]

    # by displacements
    displL = get_displacements(df_res, "LimbL")
    displR = get_displacements(df_res, "LimbR")
    df_res = df_res[(displL < max_displ) & (displR < max_displ)]

    return df_res


def filter_misdetections_tail(df, likelihood=0.01, max_displ=150):
    # by likelihood
    df_res = df[
        (df["Center"].likelihood > likelihood)
        & (df["TailCenter"].likelihood > likelihood)
        & (df["TailTip"].likelihood > likelihood)
    ]

    # by displacements
    displL = get_displacements(df_res, "TailCenter")
    displR = get_displacements(df_res, "TailTip")
    df_res = df_res[(displL < max_displ) & (displR < max_displ)]

    return df_res


def get_rotation_angle(Rs):
    return numpy.rad2deg(numpy.arctan2(Rs[:, 1, 0], -Rs[:, 0, 0],))


def get_singed_angular_speed(Rs):
    return numpy.array(
        list(
            map(
                lambda a: a if abs(a) < 180 else 360 - abs(a),
                numpy.diff(get_rotation_angle(Rs)),
            )
        )
    )

