import numpy as np
import pandas as pd

from .tadpose import Tadpole
from .utils import smooth_diff, smooth


class BatchGrouper:
    def __init__(self, exp_table, aligner, output_grouped_by="Stage"):
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


def angles(tad, part_tuple1, part_tuple2, win=5, track_idx=0):
    elocs = tad.ego_locs(track_idx=track_idx).copy()

    elocs[..., 0] *= -1

    parts1_idx = [tad.bodyparts.index(p) for p in part_tuple1]
    parts2_idx = [tad.bodyparts.index(p) for p in part_tuple2]

    parts1 = elocs[:, parts1_idx, :]
    parts2 = elocs[:, parts2_idx, :]

    for p in range(parts1.shape[1]):
        parts1[:, p, :] = smooth(parts1[:, p, :], win=win)
        parts2[:, p, :] = smooth(parts2[:, p, :], win=win)

    vec1 = np.diff(parts1, axis=1).squeeze()
    vec2 = np.diff(parts2, axis=1).squeeze()

    vec1 = (vec1.T / np.linalg.norm(vec1, axis=1)).T
    vec2 = (vec2.T / np.linalg.norm(vec2, axis=1)).T

    ortho_vec1 = np.c_[-vec1[:, 1], vec1[:, 0]]
    sign = np.sign(np.sum(ortho_vec1 * vec2, axis=1))

    c = np.sum(vec1 * vec2, axis=1)
    angles = sign * np.rad2deg(np.arccos(np.clip(c, -1, 1)))
    return angles


# def get_rotation_angle(Rs):
#     return np.rad2deg(np.arctan2(Rs[:, 1, 0], -Rs[:, 0, 0],))


# def get_singed_angular_speed(Rs):
#     return np.array(
#         list(
#             map(
#                 lambda a: a if abs(a) < 180 else 360 - abs(a),
#                 np.diff(get_rotation_angle(Rs)),
#             )
#         )
#     )

