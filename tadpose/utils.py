import os
import cv2
import numpy
import pandas as pd

np = numpy

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def file_select_dialog(ext=".*"):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_name = filedialog.askopenfilename(
        defaultextension=ext, filetypes=[(f"{ext} files", f"*{ext}")], parent=root,
    )
    root.destroy()

    if len(file_name) == 0:
        print("No file specified")

    return file_name


def outfile(main_path):
    def tmp(fn):
        os.makedirs(os.path.join(main_path, os.path.dirname(fn)), exist_ok=True)
        return os.path.join(main_path, fn)

    return tmp


def add_time_column(df, fps):
    df = df.copy()
    df["Time", "frame"] = list(df.index)
    df["Time", "sec"] = df["Time", "frame"] / fps

    return df


def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))

        if len(x) < 2:
            import warnings

            warnings.warn(
                "WARNING: all locations of a bodypart are None. Cannot fill missing.... skipping"
            )
            Y[:, i] = y
            continue

        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore  to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def cart2pol(xy):
    rho = np.linalg.norm(xy, axis=-1)
    phi = np.arctan2(xy[..., 1], xy[..., 0])
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def smooth_diff(node_loc, win=25, poly=3, deriv=1):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = numpy.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=deriv)

    node_vel = numpy.linalg.norm(node_loc_vel, axis=1)

    return node_vel


def smooth(node_loc, win=25, poly=3, deriv=0):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = numpy.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=deriv)

    return node_loc_vel


def corr_roll(datax, datay, win):
    """
    datax, datay are the two timeseries to find correlations between

    win sets the number of frames over which the covariance is computed

    """

    s1 = pd.Series(datax)
    s2 = pd.Series(datay)

    return numpy.array(s2.rolling(win, center=True).corr(s1))


class VideoProcessor(object):
    """
    Base class for a video processing unit, implementation is required for video loading and saving
    sh and sw are the output height and width respectively.
    """

    def __init__(
        self, fname="", sname="", nframes=-1, fps=30, codec="X264", sh="", sw=""
    ):
        self.fname = fname
        self.sname = sname
        self.nframes = nframes
        self.codec = codec
        self.h = 0
        self.w = 0
        self.FPS = fps
        self.nc = 3
        self.i = 0

        try:
            if self.fname != "":
                self.vid = self.get_video()
                self.get_info()
                self.sh = 0
                self.sw = 0
            if self.sname != "":
                if sh == "" and sw == "":
                    self.sh = self.h
                    self.sw = self.w
                else:
                    self.sw = sw
                    self.sh = sh
                self.svid = self.create_video()

        except Exception as ex:
            print("Error: %s", ex)

    def load_frame(self):
        try:
            frame = self._read_frame()
            self.i += 1
            return frame
        except Exception as ex:
            print("Error: %s", ex)

    def height(self):
        return self.h

    def width(self):
        return self.w

    def fps(self):
        return self.FPS

    def counter(self):
        return self.i

    def frame_count(self):
        return self.nframes

    def get_video(self):
        """
        implement your own
        """
        pass

    def get_info(self):
        """
        implement your own
        """
        pass

    def create_video(self):
        """
        implement your own
        """
        pass

    def _read_frame(self):
        """
        implement your own
        """
        pass

    def save_frame(self, frame):
        """
        implement your own
        """
        pass

    def close(self):
        """
        implement your own
        """
        self.mywriter.close()


class VideoProcessorCV(VideoProcessor):
    """
    OpenCV implementation of VideoProcessor
    requires opencv-python==3.4.0.12
    """

    def __init__(self, *args, **kwargs):
        super(VideoProcessorCV, self).__init__(*args, **kwargs)

    def get_video(self):
        return cv2.VideoCapture(self.fname)

    def get_info(self):
        self.w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = self.vid.get(cv2.CAP_PROP_FPS)
        self.nc = 3
        if self.nframes == -1 or self.nframes > all_frames:
            self.nframes = all_frames

    def create_video(self):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        return cv2.VideoWriter(self.sname, fourcc, self.FPS, (self.sw, self.sh), True)

    def _read_frame(self):  # return RGB (rather than BGR)!
        # return cv2.cvtColor(np.flip(self.vid.read()[1],2), cv2.COLOR_BGR2RGB)
        return np.flip(self.vid.read()[1], 2)

    def save_frame(self, frame):
        self.svid.write(np.flip(frame, 2))

    def close(self):
        self.svid.release()
        self.vid.release()


class VideoProcessorFFMPEG(VideoProcessor):
    """
    OpenCV implementation of VideoProcessor
    requires opencv-python==3.4.0.12
    """

    def __init__(self, *args, **kwargs):
        super(VideoProcessorFFMPEG, self).__init__(*args, **kwargs)

    def get_video(self):
        return cv2.VideoCapture(self.fname)

    def get_info(self):
        self.w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = self.vid.get(cv2.CAP_PROP_FPS)
        self.nc = 3
        if self.nframes == -1 or self.nframes > all_frames:
            self.nframes = all_frames

        print("FPS", self.FPS)

    def create_video(self):
        import skvideo.io

        print("using ffmpeg writer")

        self.mywriter = skvideo.io.FFmpegWriter(
            self.sname,
            outputdict={
                "-vcodec": "libx264",  # use the h.264 codec
                "-crf": "18",  # set the constant rate factor to 0, which is lossless
                "-preset": "fast",  # the slower the better compression, in princple, try,
                "-framerate": str(self.FPS)
                # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
            },
        )

        return self.mywriter

    def _read_frame(self):  # return RGB (rather than BGR)!
        # return cv2.cvtColor(np.flip(self.vid.read()[1],2), cv2.COLOR_BGR2RGB)
        return np.flip(self.vid.read()[1], 2)

    def save_frame(self, frame):
        self.mywriter.writeFrame(frame)

    def close(self):
        self.mywriter.close()
        self.vid.release()
