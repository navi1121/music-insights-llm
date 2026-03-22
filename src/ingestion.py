import pandas as pd
import glob
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PATH = os.path.join(BASE_DIR, "my_spotify_data", "Spotify Extended Streaming History")
FILE_PATTERN = "Streaming_History_Audio_*.json"


def load_data() -> pd.DataFrame:
    files = glob.glob(f"{PATH}/{FILE_PATTERN}")
    print(f"Found {len(files)} file(s).")

    if not files:
        raise FileNotFoundError(f"No JSON files found at: {PATH}")

    frames = [pd.read_json(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Raw rows across all files: {len(df):,}")

    # Drop columns that aren't useful for analysis
    drop_cols = [c for c in ["ip_addr", "offline_timestamp"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Drop podcast/episode rows — they have no track metadata
    df = df.dropna(subset=[
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
    ])

    # Parse timestamps
    df["ts"]      = pd.to_datetime(df["ts"], utc=True)
    df["year"]    = df["ts"].dt.year
    df["month"]   = df["ts"].dt.month
    df["hour"]    = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.day_name()

    # Listening duration
    df["minutes_played"] = (df["ms_played"] / 60000).round(2)

    # Skip detection — use the dedicated boolean field, NOT reason_end == "fwdbtn"
    # reason_end=="fwdbtn" only catches ~13% of actual skips in this export format.
    if "skipped" in df.columns:
        df["was_skipped"] = df["skipped"].fillna(False).astype(bool)
    else:
        # Fallback for older export formats without the boolean field
        df["was_skipped"] = df["reason_end"] == "fwdbtn"

    # Time-of-day bucket
    def time_of_day(h):
        if 5  <= h < 12: return "morning"
        if 12 <= h < 17: return "afternoon"
        if 17 <= h < 21: return "evening"
        return "night"

    df["time_of_day"] = df["hour"].apply(time_of_day)

    # Drop exact duplicate log entries
    df = df.drop_duplicates(subset=[
        "ts",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
    ])

    print(f"Clean rows after filtering & dedup: {len(df):,}")
    return df