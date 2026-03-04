import os
import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key="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")
MODEL = "text-embedding-3-small"

CSV_PATH = "top_300_imdb_movies.csv"
OUT_PATH = "movie_index.npz"

def movie_to_doc(row) -> str:
    parts = [
        f"Title: {row['Title']}",
        f"Year: {row.get('Year', '')}",
        f"Genres: {row.get('Genres', '')}",
        f"Director: {row.get('Director', '')}",
        f"IMDb rating: {row.get('IMDb_Rating', '')}",
        f"Votes: {row.get('Votes', '')}",
        f"Overview: {row.get('Overview', '')}",
    ]
    return "\n".join([p for p in parts if str(p).strip() and "nan" not in str(p).lower()])

def embed_texts(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

df = pd.read_csv(CSV_PATH)

# id estable: usa Rank si existe, si no, el índice
if "Rank" in df.columns:
    ids = df["Rank"].to_numpy()
else:
    ids = df.index.to_numpy()

docs = [movie_to_doc(r) for _, r in df.iterrows()]

BATCH = 128
embs = []
for i in range(0, len(docs), BATCH):
    embs.append(embed_texts(docs[i:i+BATCH]))
embs = np.vstack(embs)

np.savez(
    OUT_PATH,
    embeddings=embs,
    ids=ids,
    titles=df["Title"].to_numpy(),
    years=df.get("Year", pd.Series([""] * len(df))).to_numpy(),
    genres=df.get("Genres", pd.Series([""] * len(df))).to_numpy(),
    directors=df.get("Director", pd.Series([""] * len(df))).to_numpy(),
    overviews=df.get("Overview", pd.Series([""] * len(df))).to_numpy(),
    docs=np.array(docs, dtype=object),
)

print(f"Index guardado en {OUT_PATH} | embeddings shape={embs.shape}")










































