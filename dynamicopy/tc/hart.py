import numpy as np



def theta(x0=120,x1=130,y0=12,y1=10): # TODO : Gérer différemment SH ?
    """
    Computes the angular direction between to points.
    0° corresponds to eastward, 90° northward, 180° westward and 270° southward.


    Parameters
    ----------
    x0: longitude coordinate of the current point
    x1: longitude coordinate of the next point
    y0: latitude coordinate of the current point
    y1: longitude coordinate of the next point

    Returns
    -------
    The directional angle between the current and the next point.
    """
    u = [1,0]
    v = [x1-x0,y1-y0]
    if np.linalg.norm(v) != 0 :
        cos = (x1-x0) / (np.linalg.norm(u) * np.linalg.norm(v)) # Simplification due to u's coordinates
        Θ = np.sign(y1-y0) * np.arccos(cos) * 180 / np.pi
    else :
        Θ = np.nan
    return Θ

def theta_track(track):
    """
    Computes the angular direction for each points along a track.
    Handling the track altogether allows for treating stationnary cases as well as the end of the track.

    Parameters
    ----------
    track: The Dataframe with the lon/lat positions of the point along the track.

    Returns
    -------
    Θ (np.ndarray): values of Θ along the track.
    """
    Θ = []
    track=track.reset_index()
    for i in track.index[:-1]:  # On calcule la direction de chaque point vers son successeur
        Θ.append(theta(track.loc[i, 'lon'], track.loc[i + 1, 'lon'], track.loc[i, 'lat'], track.loc[i + 1, 'lat']))
        if np.isnan(Θ[-1]):  # S'il est indentique, on prend la direction précédente
            if i != 0 :
                Θ[-1] = Θ[-2]
    if len(track) > 1 : Θ.append(Θ[-1]);
    else : Θ = [np.nan]
    return Θ

def right_left(field, Θ):
    """
    Separate field into left and right of the Θ line.

    Parameters
    ----------
    field
    Θ

    Returns
    -------

    """
    if Θ <= 180 :
        right = field.where((field.az<=Θ) | (field.az>180+Θ))
        left = field.where((field.az>Θ) & (field.az<=180+Θ))
    else :
        right = field.where((field.az<=Θ) & (field.az>Θ-180))
        left = field.where((field.az>Θ) | (field.az<=Θ-180))
    return right, left

def area_weights(field) :
    """
    Computes the weights needed for the weighted mean of polar field.

    Parameters
    ----------
    field

    Returns
    -------

    """
    δ=(field.r[1] - field.r[0])/2
    w = (field.r +δ) ** 2 - (field.r - δ) ** 2
    return w

def B(Θ, snap, SH = False, names=["snap_z900", "snap_z600"]): # TODO : Vectoriser
    """
    Computes the B parameter for a point, with the corresponding snapshot of geopt at 600hPa and 900hPa

    Parameters
    ----------
    point
    snap

    Returns
    -------

    """
    z900 = snap[names[0]]
    z600 = snap[names[1]]
    z900_R, z900_L = right_left(z900, Θ)
    z600_R, z600_L = right_left(z600, Θ)
    ΔZ_R = z900_R - z600_R
    ΔZ_L = z900_L - z600_L
    if SH : h = -1;
    else : h=1;
    return  h * (ΔZ_R.weighted(area_weights(ΔZ_R)).mean() - ΔZ_L.weighted(area_weights(ΔZ_L)).mean())

def VT(snap, names=["snap_z900", "snap_z600", "snap_z300"]):
    """
    Computes V_T^U and V_T^L parameters for the given snapshot of geopt at 300, 600 and 900 hPa

    Parameters
    ----------
    snap

    Returns
    -------

    """
    z900 = snap[names[0]]
    z600 = snap[names[1]]
    z300 = snap[names[2]]
    δz300 = np.abs(z300.max() - z300.min())
    δz600 = np.abs(z600.max() - z600.min())
    δz900 = np.abs(z900.max() - z900.min())
    VTL = 750 * (δz900 - δz600) / (900 - 600)
    VTU = 450 * (δz600 - δz300) / (600 - 300)
    return VTL, VTU