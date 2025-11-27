# cache_utils.py
"""
Simple caching helpers for embeddings + TF-IDF vectorizer.

Features:
- Save / load embeddings as compressed npz (np.savez_compressed)
- Save / load sklearn TfidfVectorizer via pickle
- Safe atomic write (writes temp file then renames)
- Small helper to check cache freshness (file exists)
"""

from pathlib import Path
import pickle
import numpy as np
import tempfile
import os
from typing import Any

def save_embeddings_npz(path: Path, embeddings: np.ndarray) -> None:
    """
    Save embeddings (2D numpy array) to a compressed .npz file atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
        tmp_name = tmp.name
    try:
        np.savez_compressed(tmp_name, embeddings=embeddings)
        os.replace(tmp_name, str(path))
    finally:
        if os.path.exists(tmp_name):
            try: os.remove(tmp_name)
            except: pass

def load_embeddings_npz(path: Path) -> np.ndarray:
    """
    Load embeddings from .npz. Raises if not found or malformed.
    """
    data = np.load(str(path), allow_pickle=False)
    if "embeddings" not in data:
        raise ValueError("NPZ does not contain 'embeddings' array")
    return data["embeddings"]

def save_object_pickle(path: Path, obj: Any) -> None:
    """
    Save arbitrary python object (e.g. TfidfVectorizer) using pickle atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
        tmp_name = tmp.name
    try:
        with open(tmp_name, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name, str(path))
    finally:
        if os.path.exists(tmp_name):
            try: os.remove(tmp_name)
            except: pass

def load_object_pickle(path: Path) -> Any:
    """
    Load a pickled object. Raises if not found.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def exists(path: Path) -> bool:
    return path.exists()
