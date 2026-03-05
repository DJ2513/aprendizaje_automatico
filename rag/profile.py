from typing import Optional
import numpy as np

def init_profile(dim: int):
    return np.zeros((dim,), dtype=np.float32)

def update_profile(profile: np.ndarray, movie_vec: np.ndarray, sign: float) -> np.ndarray:
    profile = (profile + sign * movie_vec).astype(np.float32)
    return normalize(profile)

def normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

def blend(profile: Optional[np.ndarray], query_vec: np.ndarray, mix: float) -> np.ndarray:
    if profile is None or mix <= 0:
        return query_vec
    return (mix * profile + (1.0 - mix) * query_vec).astype(np.float32)