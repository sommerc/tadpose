import numpy as np
from tqdm.auto import tqdm
from skimage import transform as st
from scipy.ndimage import map_coordinates


from tadpose import utils


class TadpoleAligner:
    def __init__(self, alignment_dict, scale=False, smooth_sigma=None):
        self.alignment_dict = alignment_dict
        self.scale = scale

        self.Q = np.stack(list(alignment_dict.values()), axis=0)
        self.bodyparts_to_align = list(alignment_dict.keys())

        self.smooth_sigma = smooth_sigma

        self.alignment_matrices = {}
        self.transformations = {}

    def fit(self, track_idx, all_bodyparts, all_locations):
        part_idx = [all_bodyparts.index(p) for p in self.bodyparts_to_align]
        Ps = all_locations[:, part_idx, ...]

        if self.smooth_sigma is not None:
            for p in range(Ps.shape[1]):
                Ps[:, p] = utils.smooth_gaussian(Ps[:, p], sigma=self.smooth_sigma)

        n = len(Ps)

        Cs = np.empty((n,))
        Rs = np.empty_like(Ps)
        Ts = np.empty((n, 2))

        transformations = [None] * n

        for i, P in enumerate(Ps):
            Cs[i], Rs[i], Ts[i] = umeyama(P, self.Q)

            hom = np.zeros((3, 3))
            hom[2, 2] = 1

            hom[:2, :2] = Rs[i].T
            hom[:2, 2] = Ts[i] / Cs[i]

            transformations[i] = st.EuclideanTransform(matrix=hom)

        self.alignment_matrices[track_idx] = (Cs, Rs, Ts)
        self.transformations[track_idx] = transformations

    def transform(self, track_idx, locations):
        Cs, Rs, Ts = self.alignment_matrices[track_idx]

        return ((locations @ Rs).T).T + (Ts[:, None, :].T / Cs).T

    def warp_image(
        self, image, trans, dest_height, dest_width,
    ):
        # xy coords (needed for LA)
        dbb_coords = np.meshgrid(
            np.linspace(-dest_width // 2, dest_width // 2, dest_width),
            np.linspace(-dest_height // 2, dest_height // 2, dest_height),
        )

        dbb_coords = np.stack([dbb_coords[0].ravel(), dbb_coords[1].ravel()], axis=1)

        sbb_coords = trans.inverse(dbb_coords).T.reshape(2, dest_height, dest_width)

        # flip xy coords to ij, for image extraction
        sbb_coords = sbb_coords[::-1, ...]

        if image.ndim == 2:
            image = image[..., None]

        nchan = image.shape[2]

        image_trans = np.zeros(sbb_coords.shape[1:] + (nchan,), dtype="uint8")
        for c in range(nchan):
            image_trans[..., c] = map_coordinates(
                image[:, :, nchan - c - 1], sbb_coords, order=1, prefilter=False
            )  # BGR -> RGB

        return image_trans

    # def export_movie(
    #     self,
    #     tadpole,4
    #     movie_in,
    #     movie_out,
    #     dest_height=740,
    #     dest_width=280,
    #     min_lh=0.33,
    #     trail_len=24,
    #     trail_parts=None,
    #     dot_radius=5,
    #     just_frames=False,
    #     parts=None,
    #     skeletons=None,
    #     skeleton_colors=None,
    #     rgb=False,
    # ):
    #     from .utils import VideoProcessorCV as vp

    #     df = tadpole.aligned_locations
    #     n = len(df)

    #     Cs, Rs, Ts = self.estimate_alignment(tadpole)

    #     df_aligned = self.do_alignment(tadpole, Cs, Rs, Ts)

    #     parts_to_trans, part_probs = tadpole.split_detection_and_likelihood()

    #     clip = vp(movie_in, movie_out, codec="mp4v", sw=dest_width, sh=dest_height)

    #     dest_shape = np.array([dest_height, dest_width])
    #     dest_offset = dest_shape // 2

    #     for i, (c, R, T, Ps, Plh) in tqdm(
    #         enumerate(zip(Cs, Rs, Ts, parts_to_trans, part_probs)), total=n
    #     ):
    #         image = clip.load_frame()

    #         trans = self._get_transformation(c, R, T)
    #         image_trans = self.warp_image(image, trans, dest_shape, rgb)

    #         # if i > 500:
    #         #     break

    #         if just_frames:
    #             clip.save_frame(np.rot90(image_trans, k=2))
    #             continue

    #         # paint axis
    #         image_trans[dest_offset[0], :, :] = 128
    #         image_trans[:, dest_offset[1], :] = 128

    #         # paint trail detection trail
    #         start_trail = max(0, i - trail_len)
    #         end_trail = i

    #         if trail_parts is None:
    #             trail_parts = tadpole.bodyparts
    #             trail_colors = tadpole.bodypart_colors
    #         else:
    #             trail_colors = [
    #                 tadpole.bodypart_colors[tadpole.bodyparts.index(tp)]
    #                 for tp in trail_parts
    #             ]

    #         for bp, bc in zip(trail_parts, trail_colors):

    #             trail_df = df_aligned[bp].loc[start_trail:end_trail]
    #             trail_df = trail_df[trail_df.likelihood > min_lh]

    #             trail_df[["x", "y"]] = trail_df[["x", "y"]] + dest_offset[::-1]

    #             trail_df["x"] = trail_df["x"].clip(0, dest_width - 1)
    #             trail_df["y"] = trail_df["y"].clip(0, dest_height - 1)

    #             for (_, row1), (_, row2) in zip(
    #                 trail_df.iloc[:-1].iterrows(), trail_df[1:].iterrows()
    #             ):
    #                 rr, cc = line(int(row1.y), int(row1.x), int(row2.y), int(row2.x))
    #                 image_trans[rr, cc, :] = bc

    #         if parts is None:
    #             parts = tadpole.bodyparts

    #         # paint skeletons
    #         aligned_locs = trans(Ps)
    #         parts_idx = dict([(p, i) for i, p in enumerate(tadpole.bodyparts)])

    #         if skeletons is not None:
    #             if skeleton_colors is None:
    #                 skeleton_colors = dict(
    #                     [
    #                         (i, tadpole.bodypart_color[skel[-1]])
    #                         for (i, skel) in enumerate(skeletons)
    #                     ]
    #                 )
    #             for skel_i, skel in enumerate(skeletons):
    #                 for pair in zip(skel, skel[1:]):
    #                     p1 = parts_idx[pair[0]]
    #                     p2 = parts_idx[pair[1]]
    #                     if (Plh[p1] > min_lh) and (Plh[p2] > min_lh):
    #                         nP1 = (aligned_locs[p1] + dest_offset[::-1] + 0.5).astype(
    #                             "int32"
    #                         )
    #                         nP2 = (aligned_locs[p2] + dest_offset[::-1] + 0.5).astype(
    #                             "int32"
    #                         )
    #                         rr, cc, val = line_aa(
    #                             int(np.clip(nP1[1], 0, dest_height - 1)),
    #                             int(np.clip(nP1[0], 0, dest_width - 1)),
    #                             int(np.clip(nP2[1], 1, dest_height - 1)),
    #                             int(np.clip(nP2[0], 1, dest_width - 1)),
    #                         )
    #                         image_trans[rr, cc, :] = skeleton_colors[skel_i] * 255.0

    #         # paint current detection

    #         for ip, (nP, lh) in enumerate(zip(aligned_locs, Plh)):
    #             # flip dest_offset into xy
    #             if lh > min_lh and tadpole.bodyparts[ip] in parts:
    #                 nP = (nP + dest_offset[::-1] + 0.5).astype("int32")
    #                 rr, cc = disk((nP[1], nP[0]), dot_radius, shape=dest_shape)
    #                 image_trans[rr, cc, :] = tadpole.bodypart_colors[ip]

    #         clip.save_frame(np.rot90(image_trans, k=2))

    #     clip.close()


def umeyama(P, Q):
    """
    Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
    with least-squares error.
    Returns (scale factor c, rotation matrix R, translation vector t) such that
      Q = P*cR + t
    if they align perfectly, or such that
      SUM over point i ( | P_i*cR + t - Q_i |^2 )
    is minimised if they don't align perfectly."""
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1 / varP * np.sum(S)  # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c * R)

    return c, R, t
