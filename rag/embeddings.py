import os
import numpy as np
from openai import OpenAI

MODEL = "text-embedding-3-small"

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def embed_one(text: str) -> np.ndarray:
    resp = _client.embeddings.create(model=MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)