from .metrics import storm_stats

def identify_lifecycle(tracks, ss = None,):
    if ss == None :
        ss = storm_stats(tracks)
    t_interm = tracks.merge(ss[["track_id", "time_slp", "time_wind"]], on = "track_id")
    tracks = tracks.assign(lag_climax_slp = t_interm.time - t_interm.time_slp)
    tracks = tracks.assign(lag_climax_wind = t_interm.time - t_interm.time_wind)
    return tracks

if __name__ == "__main__":
    from dynamicopy.tc import load_TEtracks
    tracks = load_TEtracks("tests/ICO-HR_UZ_STJ.csv")
    tracks_lc = identify_lifecycle(tracks)