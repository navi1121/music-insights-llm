import os
from ingestion import load_data
from embeddings import create_vector_store
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

VECTORSTORE_PATH = "vectorstore"
EMBED_MODEL      = "nomic-embed-text"

# ── Load or build the vectorstore ────────────────────────────────────────────
# FAISS.load_local requires the same embeddings object used at creation time.
# The original code was missing this argument — it would crash on load.

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

if os.path.exists(VECTORSTORE_PATH):
    print("Loading cached vectorstore from disk...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,                        # <-- was missing, caused crash
        allow_dangerous_deserialization=True
    )
    print("Vectorstore loaded.")
else:
    print("No cached vectorstore found. Building from scratch (runs once)...")
    df = load_data()
    vectorstore = create_vector_store(df)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vectorstore saved to '{VECTORSTORE_PATH}'.")

# ── Retriever ─────────────────────────────────────────────────────────────────
# Pass a plain string, not a dict — retriever.invoke() takes a string query
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

query = "What are some of the most played tracks in my Spotify history?"
docs  = retriever.invoke(query)   # <-- was passing {"query": "..."}, should be a string

print(f"\nTop results for: '{query}'\n")
for i, d in enumerate(docs, 1):
    print(f"[{i}] {d.page_content}")
    print()