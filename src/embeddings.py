import pandas as pd
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE  = 64


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    decided to group songs played by (artist, track) and create one document per unique song.
    reducing around 50k+ rows to ~2k-5k documents — the main speed fix.
    """
    docs = []
    grouped = df.groupby([ #grouping by artist and track to get unique songs
        "master_metadata_album_artist_name",
        "master_metadata_track_name",
    ])

    for (artist, track), group in tqdm(grouped, desc="Building documents"):
        play_count   = len(group)
        total_min    = group["minutes_played"].sum().round(1)
        skip_count   = int(group["was_skipped"].sum())
        skip_rate    = group["was_skipped"].mean()
        years        = sorted(group["year"].unique().tolist())
        top_time     = group["time_of_day"].mode()[0]
        first_played = group["ts"].min().strftime("%Y-%m-%d")
        last_played  = group["ts"].max().strftime("%Y-%m-%d")

        album_col = "master_metadata_album_album_name"
        album = group[album_col].mode()[0] if album_col in group.columns else "unknown"

        # Extra fields available from the real export
        platform_col = "platform"
        top_platform = group[platform_col].mode()[0] if platform_col in group.columns else None

        shuffle_col = "shuffle"
        shuffle_pct = group[shuffle_col].mean() if shuffle_col in group.columns else None

        # Natural-language summary — this is what gets embedded
        content = (
            f"Track: {track} by {artist} (Album: {album}). "
            f"Played {play_count} times, totalling {total_min} minutes. "
            f"Skipped {skip_count} times ({skip_rate:.0%} skip rate). "
            f"Most often listened to in the {top_time}. "
            f"Active years: {', '.join(map(str, years))}. "
            f"First played {first_played}, last played {last_played}."
            + (f" Usually played on {top_platform}." if top_platform else "")
            + (f" Played on shuffle {shuffle_pct:.0%} of the time." if shuffle_pct is not None else "")
        )

        # Structured metadata — for filtering in tool calls without extra embedding cost
        metadata = {
            "artist":        artist,
            "track":         track,
            "album":         album,
            "play_count":    int(play_count),
            "total_minutes": float(total_min),
            "skip_count":    skip_count,
            "skip_rate":     round(float(skip_rate), 3),
            "time_of_day":   top_time,
            "first_year":    int(years[0]),
            "last_year":     int(years[-1]),
            "first_played":  first_played,
            "last_played":   last_played,
        }
        if top_platform:
            metadata["top_platform"] = top_platform
        if shuffle_pct is not None:
            metadata["shuffle_rate"] = round(float(shuffle_pct), 3)

        docs.append(Document(page_content=content, metadata=metadata))

    print(f"Built {len(docs):,} documents from {len(df):,} play events.")
    return docs


def create_vector_store(df: pd.DataFrame) -> FAISS:
    """Embed documents in batches and build a FAISS index."""
    docs = build_documents(df)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print(f"Embedding {len(docs):,} documents in batches of {BATCH_SIZE}...")

    first_batch = docs[:BATCH_SIZE]
    vectorstore = FAISS.from_documents(first_batch, embeddings)

    for i in tqdm(range(BATCH_SIZE, len(docs), BATCH_SIZE), desc="Embedding batches"):
        vectorstore.add_documents(docs[i : i + BATCH_SIZE])

    print("Embedding complete.")
    return vectorstore