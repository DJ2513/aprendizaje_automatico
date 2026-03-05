import numpy as np

def load_index(path: str):
    idx = np.load(path, allow_pickle=True)
    M = idx["embeddings"].astype(np.float32)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

    return {
        "M": M,
        "M_norm": M_norm,
        "titles": idx["titles"],
        "years": idx.get("years"),
        "genres": idx.get("genres"),
        "directors": idx.get("directors"),
        "overviews": idx.get("overviews"),
    }

def cosine_scores(q: np.ndarray, M_norm: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-12)
    return M_norm @ q