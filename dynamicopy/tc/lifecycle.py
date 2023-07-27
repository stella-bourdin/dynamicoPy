from .metrics import storm_stats
import pandas as pd

def identify_lifecycle(tracks, ss = None,):
    if type(ss) != pd.core.frame.DataFrame : # Compute ss if not provided
        ss = storm_stats(tracks)
    # For each point indicate time of max wind and min slp
    t_interm = tracks.merge(ss[["track_id", "time_slp", "time_wind"]], on = "track_id", how= "left")
    #
    tracks = tracks.assign(lag_climax_slp = (t_interm.time - t_interm.time_slp).astype('timedelta64[h]').values)
    tracks = tracks.assign(lag_climax_wind = (t_interm.time - t_interm.time_wind).astype('timedelta64[h]').values)
    return tracks

if __name__ == "__main__":
    from dynamicopy.tc import load_TEtracks
    tracks = load_TEtracks("tests/ICO-HR_UZ_STJ.csv")
    tracks_lc = identify_lifecycle(tracks)