import pandas as pd

def listening_trends_by_year(df: pd.DataFrame):
    """
    Returns total listening time per year (in minutes)
    """
    trends = (
        df.groupby("year")["minutes_played"]
        .sum()
        .sort_index()
    )

    return trends.to_dict()