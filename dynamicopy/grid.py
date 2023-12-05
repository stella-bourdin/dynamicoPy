import numpy as np
import matplotlib.pyplot as plt

R_terre = 6371e3
S_terre = 4 * np.pi * R_terre ** 2

def dy(N_lat):
    return R_terre * np.pi / N_lat
def r(lat, deg = True) :
    if deg : lat = lat * np.pi / 180;
    return R_terre * (np.sin(np.pi / 2 - lat))
def dx(N_lon, lat, deg = True) :
    r_phi = r(lat, deg)
    return r_phi * 2 * np.pi / N_lon
def A(N_lon, N_lat, lat, deg = True) :
    return dx(N_lon, lat, deg) * dy(N_lat)

def nbp2cells(nbp) :
    return 10*nbp**2 + 2
def area_ico(nbp) :
    return S_terre / nbp2cells(nbp)

def res_ico(nbp):
    f = 2 / np.sqrt(np.pi)
    area = area_ico(nbp)
    return f*np.sqrt(area)


def compare(N_lon, N_lat, nbp):
    lats = np.linspace(-90, 90, N_lat)
    lons = np.arange(-180, 180, 360 / N_lon)
    δxs = dx(N_lon, lats)
    δy = dy(N_lat)
    areas = A(N_lon, N_lat, lats)

    N_cell = nbp2cells(nbp)
    A_ico = area_ico(nbp)
    dm = res_ico(nbp)

    lat_eq = lats[np.argmin(np.abs(areas - A_ico))]
    print("latitude eq.", lat_eq)

    fig, axs = plt.subplots(1, 2, figsize=[10, 3])
    axs[0].plot(lats, δxs / 1000, color="k", label="dx")
    axs[0].axhline(y=δy / 1000, color="grey", label="dy")
    axs[0].axhline(y=dm / 1000, color="red", label="ico")
    axs[0].legend()
    axs[0].set_xlabel("Latitude / °")
    axs[0].set_ylabel("Grid cell dimension / km")

    axs[1].plot(lats, np.array(areas) * 1e-6, color="k", label="lon-lat grid")
    axs[1].axhline(y=A_ico * 1e-6, color="red", label="ICO grid")
    axs[1].axvline(x=lat_eq, color="grey", linestyle="--")
    axs[1].axvline(x=-lat_eq, color="grey", linestyle="--")
    axs[1].legend()
    axs[1].set_xlabel("Latitude / °")
    axs[1].set_ylabel("Cell area / km²")

    plt.tight_layout()