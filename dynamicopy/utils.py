# -*- coding: utf-8 -*-

# Utilitary functions to use inside the module

import numpy as np

### ======================================================= ###
###          Utilitary data manipulation functions          ###
### ======================================================= ###


def idx_closest(A, x):
    """Returns the index of the closest value to x in array A

    Parameters
    ----------
    A : list or 1D np.ndarray
        The array is which to find the closest value to x
    x : float or int
        The value we want to find the closest entry to in A

    Returns
    -------
    int
        The coordinate of the closest value to x in A
    """
    return np.argmin(abs(np.array(A) - x))


def sign_change_detect(A):
    """Indicates the index of the sign change in A

    Parameters
    ----------
    A : list or 1D array
        The data in which we want to detect the sign change

    Returns
    -------
    int
        The index of the first value whose sign is different from the sign of the first value.
    """
    sign = np.sign(A)
    change = sign != sign[0]
    return np.min(np.where(change == True))

def hist2d(bdd, n=780):
    H, X, Y = np.histogram2d(bdd.lon, bdd.lat, bins=[360/4,180/4], range=((0, 360), (-90, 90)))

    hist = xr.DataArray(data=H.T/n, coords=(Y[:-1], X[:-1]), dims = ("lat", "lon"))
    hist.to_dataset(name="H")
    hist.lat["units"] = "degrees"
    hist.lon["units"] = "degrees"

    return hist

if __name__ == "__main__":
    pass
