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

from .utils import smooth_diff


def get_displacements(df, part, coords=("x", "y")):
    return (df[part][list(coords)].diff(1) ** 2).sum(1, skipna=False).apply(numpy.sqrt)


def get_displacements(df, part, coords=("x", "y")):
    return (df[part][list(coords)].diff(1) ** 2).sum(1, skipna=False).apply(numpy.sqrt)


# def get_speed(df, part, coords=("x", "y"), per="frame"):
#     displ = get_displacements(df, part, coords)
#     dt = df["Time"][per].diff(1)
#     res = displ / dt
#     res.iloc[0] = res.iloc[1]
#     return res


def ego_speeds(tadpole, parts=None):
    if parts is None:
        parts = tadpole.bodyparts

    locs = tadpole.ego_locs(parts=tuple(parts))

    speeds = {}
    for pi, p in enumerate(parts):
        speeds[p] = smooth_diff(locs[:, pi, :])

    return pandas.DataFrame(speeds)


def speeds(tadpole, parts=None):
    if parts is None:
        parts = tadpole.bodyparts

    locs = tadpole.locs(parts=tuple(parts))

    speeds = {}
    for pi, p in enumerate(parts):
        speeds[p] = smooth_diff(locs[:, pi, :])

    return pandas.DataFrame(speeds, columns=parts)


# def get_angle(df_flt_tail, part1="TailTip", part2="TailCenter"):
#     v0 = df_flt_tail[part1][coords].to_numpy() - df_flt_tail[part2][coords].to_numpy()
#     v1 = df_flt_tail[part2][coords].to_numpy()

#     v0 = v0 / numpy.linalg.norm(v0, axis=1)[:, None]
#     v1 = v1 / numpy.linalg.norm(v1, axis=1)[:, None]

#     c = (v0 * v1).sum(1)
#     angles = numpy.arccos(numpy.clip(c, -1, 1))
#     return numpy.rad2deg(angles)


# def filter_likelihood(df, parts, min_likelihood):
#     sel = reduce(numpy.logical_and, [df[p].likelihood > min_likelihood for p in parts])

#     return df[sel]


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

