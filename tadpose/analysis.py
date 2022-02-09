import numpy as np
import pandas as pd

from .tadpose import Tadpole
from .utils import smooth_diff

coords = ["x", "y"]


class BatchGrouper:
    def __init__(self, exp_table, aligner, output_grouped_by="Stage", output_path=None):
        if isinstance(exp_table, str):
            self.exp_table = pd.read_csv(exp_table, sep="\t")
        else:
            self.exp_table = exp_table

        self.aligner = aligner
        self.output_grouped_by = output_grouped_by

    def __len__(self):
        return len(self.exp_table)

    def __iter__(self):
        for grp, df_grp in self.exp_table.groupby(self.output_grouped_by):
            for ind, row in df_grp.iterrows():
                tadpole = Tadpole.from_sleap(row["FullPath"],)
                tadpole.aligner = self.aligner

                yield tadpole, grp, ind, row


# def get_displacements(df, part, coords=("x", "y")):
#     return (df[part][list(coords)].diff(1) ** 2).sum(1, skipna=False).apply(np.sqrt)


# def get_displacements(df, part, coords=("x", "y")):
#     return (df[part][list(coords)].diff(1) ** 2).sum(1, skipna=False).apply(np.sqrt)


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

    return pd.DataFrame(speeds)


def speeds(tadpole, parts=None):
    if parts is None:
        parts = tadpole.bodyparts

    locs = tadpole.locs(parts=tuple(parts))

    speeds = {}
    for pi, p in enumerate(parts):
        speeds[p] = smooth_diff(locs[:, pi, :])

    return pd.DataFrame(speeds, columns=parts)


# def get_angle(df_flt_tail, part1="TailTip", part2="TailCenter"):
#     v0 = df_flt_tail[part1][coords].to_numpy() - df_flt_tail[part2][coords].to_numpy()
#     v1 = df_flt_tail[part2][coords].to_numpy()

#     v0 = v0 / np.linalg.norm(v0, axis=1)[:, None]
#     v1 = v1 / np.linalg.norm(v1, axis=1)[:, None]

#     c = (v0 * v1).sum(1)
#     angles = np.arccos(np.clip(c, -1, 1))
#     return np.rad2deg(angles)


# def filter_likelihood(df, parts, min_likelihood):
#     sel = reduce(np.logical_and, [df[p].likelihood > min_likelihood for p in parts])

#     return df[sel]


def get_rotation_angle(Rs):
    return np.rad2deg(np.arctan2(Rs[:, 1, 0], -Rs[:, 0, 0],))


def get_singed_angular_speed(Rs):
    return np.array(
        list(
            map(
                lambda a: a if abs(a) < 180 else 360 - abs(a),
                np.diff(get_rotation_angle(Rs)),
            )
        )
    )

