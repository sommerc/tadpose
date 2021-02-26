import os
import numpy


def outfile(main_path):
    def tmp(fn):
        os.makedirs(os.path.join(main_path, os.path.dirname(fn)), exist_ok=True)
        return os.path.join(main_path, fn)

    return tmp


from scipy.interpolate import interp1d


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
        x = numpy.flatnonzero(~numpy.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=numpy.nan, bounds_error=False)

        # Fill missing
        xq = numpy.flatnonzero(numpy.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = numpy.isnan(y)
        y[mask] = numpy.interp(
            numpy.flatnonzero(mask), numpy.flatnonzero(~mask), y[~mask]
        )

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


from scipy.signal import savgol_filter


def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = numpy.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = numpy.linalg.norm(node_loc_vel, axis=1)

    return node_vel


import pandas as pd


def corr_roll(datax, datay, win):
    """
    datax, datay are the two timeseries to find correlations between

    win sets the number of frames over which the covariance is computed

    """

    s1 = pd.Series(datax)
    s2 = pd.Series(datay)

    return numpy.array(s2.rolling(win).corr(s1))
