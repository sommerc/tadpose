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


def add_time_column(df, fps=24):
    df["Time", "frame"] = list(df.index)
    df["Time", "sec"] = df["Time", "frame"] / fps


def mirror_df(in_df):
    df = in_df.copy()
    for bp in ["Eye", "Limb"]:
        tmp = df[f"{bp}L"].copy()
        df[f"{bp}L"] = df[f"{bp}R"]
        df[f"{bp}R"] = tmp

    for c in df.columns:
        if c[1] == "x":
            df[c] = -df[c]

    return df


class TadpoleAligner:
    def __init__(self, cfg, alignment_dict, scale=False):
        self.cfg = cfg
        self.bodyparts = list(cfg["bodyparts"])
        self.alignment_dict = alignment_dict
        self.scale = scale
        self.fps = 24.0

        self.Q = numpy.stack(list(alignment_dict.values()), axis=0)

    def estimate_allign(self, df):
        n = df.shape[0]
        bodyparts = self.cfg["bodyparts"]

        Ps = (
            df[list(self.alignment_dict.keys())]
            .to_numpy()
            .reshape(-1, len(self.alignment_dict), 3)[:, :, :2]
        )
        Cs = numpy.empty((n,))
        Rs = numpy.empty_like(Ps)
        Ts = numpy.empty((n, 2))
        for i, P in enumerate(Ps):
            Cs[i], Rs[i], Ts[i] = umeyama(P, self.Q)

        return Cs, Rs, Ts

    def allign(self, df, Cs, Rs, Ts):
        parts = self.get_detection_points(df)
        part_probs = self.get_detection_probs(df)

        parts_aligned = ((parts @ Rs).T).T + (Ts[:, None, :].T / Cs).T

        df_aligned = pandas.DataFrame(
            numpy.concatenate([parts_aligned, part_probs[..., None]], axis=-1).reshape(
                parts.shape[0], -1
            )
        )
        df_aligned.columns = df.columns

        add_time_column(df, self.fps)
        add_time_column(df_aligned, self.fps)

        # insert time column

        return df_aligned

    def get_detection_points(self, df):
        return (
            df[self.bodyparts].to_numpy().reshape(-1, len(self.bodyparts), 3)[:, :, :2]
        )

    def get_detection_probs(self, df):
        return (
            df[self.bodyparts].to_numpy().reshape(-1, len(self.bodyparts), 3)[:, :, 2]
        )

    def _destination_bb(self, dest_height, dest_width):
        dest_shape = numpy.array([dest_height, dest_width])

        dest_offset = dest_shape // 2

        # xy coords (needed for LA)
        dbb_coords = np.meshgrid(
            np.linspace(-dest_offset[1], dest_offset[1], dest_width),
            np.linspace(-dest_offset[0], dest_offset[0], dest_height),
        )

        dbb_coords = numpy.stack([dbb_coords[0].ravel(), dbb_coords[1].ravel()], axis=1)
        return dbb_coords

    def _get_transformation(self, c, R, T):
        hom = numpy.zeros((3, 3))
        hom[2, 2] = 1

        hom[:2, :2] = R.T
        hom[:2, 2] = T / c

        trans = st.EuclideanTransform(matrix=hom)
        return trans

    def _transform(self, image, trans, dbb_coords, dest_shape):
        sbb_coords = trans.inverse(dbb_coords).T.reshape(2, *dest_shape)

        # flip xy coords to ij, for image extraction
        image_trans_gray = map_coordinates(image[:, :, 0], sbb_coords[::-1, ...])

        image_trans = np.repeat(image_trans_gray[:, :, np.newaxis], 3, axis=2).astype(
            "uint8"
        )

        return image_trans

    def export_movie(
        self,
        df,
        movie_in,
        movie_out,
        dest_height=740,
        dest_width=280,
        min_lh=0.33,
        trail_len=24,
        trail_parts=None,
        dot_radius=5,
        just_frames=False,
    ):
        n = len(df)

        Cs, Rs, Ts = self.estimate_allign(df)

        df_aligned = self.allign(df, Cs, Rs, Ts)

        parts_to_trans = self.get_detection_points(df)
        part_probs = self.get_detection_probs(df)

        clip = vp(movie_in, movie_out, codec="mp4v", sw=dest_width, sh=dest_height)

        dest_shape = numpy.array([dest_height, dest_width])
        dest_offset = dest_shape // 2

        dbb_coords = self._destination_bb(dest_height, dest_width)

        for i, (c, R, T, Ps, Plh) in tqdm(
            enumerate(zip(Cs, Rs, Ts, parts_to_trans, part_probs)), total=n
        ):
            image = clip.load_frame()

            trans = self._get_transformation(c, R, T)
            image_trans = self._transform(image, trans, dbb_coords, dest_shape)

            if just_frames:
                clip.save_frame(numpy.rot90(image_trans, k=2))
                continue

            # paint axis
            image_trans[dest_offset[0], :, :] = 128
            image_trans[:, dest_offset[1], :] = 128

            # paint trail detection trail
            start_trail = max(0, i - trail_len)
            end_trail = i

            if trail_parts is None:
                trail_parts = self.bodyparts
                trail_colors = self.bodypart_colors
            else:
                trail_colors = [
                    self.bodypart_colors[self.bodyparts.index(tp)] for tp in trail_parts
                ]

            for bp, bc in zip(trail_parts, trail_colors):

                trail_df = df_aligned[bp].loc[start_trail:end_trail]
                trail_df = trail_df[trail_df.likelihood > min_lh]

                trail_df[["x", "y"]] = trail_df[["x", "y"]] + dest_offset[::-1]

                trail_df["x"] = trail_df["x"].clip(0, dest_width - 1)
                trail_df["y"] = trail_df["y"].clip(0, dest_height - 1)

                for (_, row1), (_, row2) in zip(
                    trail_df.iloc[:-1].iterrows(), trail_df[1:].iterrows()
                ):
                    rr, cc = line(int(row1.y), int(row1.x), int(row2.y), int(row2.x))
                    image_trans[rr, cc, :] = bc

            # paint current detection
            for ip, (nP, lh) in enumerate(zip(trans(Ps), Plh)):

                # flip dest_offset into xy
                if lh > min_lh:
                    nP = (nP + dest_offset[::-1] + 0.5).astype("int32")
                    rr, cc = disk((nP[1], nP[0]), dot_radius, shape=dest_shape)
                    image_trans[rr, cc, :] = self.bodypart_colors[ip]

            clip.save_frame(numpy.rot90(image_trans, k=2))

            # if i > 300:
            #     break

        clip.close()

    def export_screenshots(
        self, df, movie_in, file_out, frames, dest_height=740, dest_width=280,
    ):
        n = len(df)

        Cs, Rs, Ts = self.estimate_allign(df)
        df_aligned = self.allign(df, Cs, Rs, Ts)

        clip = vp(movie_in, "", codec="mp4v", sw=dest_width, sh=dest_height)

        dest_shape = numpy.array([dest_height, dest_width])
        dest_offset = dest_shape // 2

        dbb_coords = self._destination_bb(dest_height, dest_width)

        parts_to_trans = self.get_detection_points(df)

        plt.ioff()

        for i, (c, R, T, Ps) in enumerate(zip(Cs, Rs, Ts, parts_to_trans)):
            image = clip.load_frame()
            if i not in frames:
                continue

            trans = self._get_transformation(c, R, T)

            image_trans = self._transform(image, trans, dbb_coords, dest_shape)
            image_trans2 = image_trans.copy()

            # io.imsave(file_out + f"image_frame_{i:04d}.png", numpy.rot90(image_trans, k=2))

            # paint current detection
            for ip, nP in enumerate(trans(Ps)):

                # flip dest_offset into xy
                nP = (nP + dest_offset[::-1] + 0.5).astype("int32")
                rr, cc = disk((nP[1], nP[0]), 8, shape=dest_shape)
                image_trans[rr, cc, :] = self.bodypart_colors[ip]

            image_trans = numpy.rot90(image_trans, k=2)
            image_trans2 = numpy.rot90(image_trans2, k=2)

            # io.imsave(file_out + f"alligned_image_frame_{i:04d}_pred.png", image_trans)

            f, ax = plt.subplots()
            ax.imshow(
                image_trans2,
                extent=[
                    dest_width // 2,
                    -dest_width // 2,
                    -dest_height // 2,
                    dest_height // 2,
                ],
            )
            ax.axhline(
                0,
                xmin=-dest_width // 2,
                xmax=dest_width // 2,
                color="white",
                linestyle=":",
                linewidth=0.5,
            )
            ax.axvline(
                0,
                ymin=-dest_height // 2,
                ymax=dest_height // 2,
                color="white",
                linestyle=":",
                linewidth=0.5,
            )

            for ip, nP in enumerate(trans(Ps)):
                ax.plot(nP[0], nP[1], ".", color=self.bodypart_colors[ip] / 255.0)

            # ax.set_xticklabels([str(-int(xxx.get_text())) for xxx in ax.get_xticklabels()])

            sns.despine(ax=ax)

            plt.savefig(
                file_out + f"alligned_image_frame_{i:04d}_with_axes.pdf", dpi=120
            )
            plt.close(f)

        clip.vid.release()


def explort_aligned_movies(
    path_config_file,
    orig_movies,
    scorer,
    out_suffix="aligned",
    overwrite=False,
    **kwargs,
):
    cfg = read_config(path_config_file)
    bodyparts = cfg["bodyparts"]

    for mov_fn in tqdm(orig_movies):
        print(mov_fn)
        mov_base_fn = os.path.splitext(mov_fn)[0]

        out_fn = f"{mov_base_fn}{scorer}_{out_suffix}.mp4"

        if os.path.exists(out_fn) and not overwrite:
            print("already there... skipping (use overwrite=True) to overwrite")
            continue

        df = pandas.read_hdf(f"{mov_base_fn}{scorer}.h5")
        df = df[scorer]
        ta = TadpoleAligner(
            cfg, {"TailStem": numpy.array([0, 0.0]), "Center": numpy.array([0, 1.0])}
        )

        ta.export_movie(df, mov_fn, out_fn, **kwargs)


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
