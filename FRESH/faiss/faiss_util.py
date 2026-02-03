# faiss_util.py
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import typing as t

DATA_DIR = r"C:\Users\sathv\OneDrive\Desktop\FRESH\data"
INDEX_PATH = os.path.join(DATA_DIR, "recipe_index.faiss")
IDS_PATH = os.path.join(DATA_DIR, "recipe_ids.npy")   # must match recommender
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy-load embedder to avoid multiple heavy loads
_embed_model = None
def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def load_index():
    """Load FAISS index and ids array from disk."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
    idx = faiss.read_index(INDEX_PATH)
    ids = np.load(IDS_PATH, allow_pickle=True)
    return idx, ids

def normalize(vec: np.ndarray):
    v = vec.astype("float32")
    faiss.normalize_L2(v)
    return v

def text_to_vector(text: str):
    """Return a normalized float32 embedding for the provided text."""
    model = _get_embed_model()
    vec = model.encode(text, convert_to_numpy=True)
    v = vec.astype("float32")
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v

def query_index_by_vector(q_vec, topk=10):
    idx, ids = load_index()
    v = q_vec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(v)
    D, I = idx.search(v, topk)
    results = []
    for score, i in zip(D[0], I[0]):
        if i < 0:
            continue
        results.append((ids[i], float(score)))
    return results

def query_index_by_text(text: str, topk=10):
    vec = text_to_vector(text)
    return query_index_by_vector(vec, topk=topk)
