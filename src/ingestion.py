import pandas as pd
import glob
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) #get to the root directory
PATH = os.path.join(BASE_DIR, "my_spotify_data", "Spotify Extended Streaming History") #join path names
FILE_PATTERN = "Streaming_History_Audio_*.json" #all the needed files have this prefix


def load_data() -> pd.DataFrame:
    files = glob.glob(f"{PATH}/{FILE_PATTERN}")
    print(f"Found {len(files)} file(s).") #checking if the files are selected correctly (6 in this case)

    if not files:
        raise FileNotFoundError(f"No JSON files found at: {PATH}") #error message

    frames = [pd.read_json(f) for f in files] #reading the json files into dataframes, putting them in a list
    df = pd.concat(frames, ignore_index=True) #concatenating the dataframes into one big one
    print(f"Raw rows across all files: {len(df):,}")

    # Drop columns that aren't useful for analysis
    drop_cols = [c for c in ["ip_addr", "offline_timestamp"] if c in df.columns] #dropping columns that aren't useful/privacy concerns
    df = df.drop(columns=drop_cols)

    # Drop podcast/episode rows — they have no track metadata
    df = df.dropna(subset=[ #these columns are only populated for music tracks, so dropping rows where they're missing gets rid of podcasts/episodes
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
    ])

    # Parse timestamps
    df["ts"]      = pd.to_datetime(df["ts"], utc=True)
    df["year"]    = df["ts"].dt.year
    df["month"]   = df["ts"].dt.month
    df["hour"]    = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.day_name()


    # Skip detection — use the dedicated boolean field, NOT reason_end == "fwdbtn"
    # reason_end=="fwdbtn" only catches ~13% of actual skips in this export format.
    if "skipped" in df.columns:
        df["was_skipped"] = df["skipped"].fillna(False).astype(bool)
    else:
        # Fallback for older export formats without the boolean field
        df["was_skipped"] = df["reason_end"] == "fwdbtn"


    # Listening duration (converting milliseconds to minutes played)
    df["minutes_played"] = df.apply(
    lambda row: round(row["ms_played"] / 60000, 2) if not row["was_skipped"] else 0, 
    axis=1
    )
    #if the track was skipped, we count it as 0 minutes played, otherwise we convert ms_played to minutes

    # Time-of-day bucket
    def time_of_day(h):
        if 5  <= h < 12: return "morning" # (5am - 12pm)
        if 12 <= h < 17: return "afternoon" # (12pm - 5pm)
        if 17 <= h < 21: return "evening" # (5pm - 9pm)
        return "night" #everything else (9pm - 5am)

    df["time_of_day"] = df["hour"].apply(time_of_day) #applying function to create time_of_day column

    # Drop exact duplicate log entries
    df = df.drop_duplicates(subset=[  #if all of these are the same, the entries are duplicates
        "ts",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
    ])

    print(f"Clean rows after filtering & dedup: {len(df):,}")
    return df