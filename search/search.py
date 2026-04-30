"""search.py — search API for the demo to use

All public functions return list[dict] with consistent keys and JSON-safe
(Python float) scores. Missing inputs return [], never raise.

Loaded once at import (slow path; ~3-5s):
  - 3 FAISS indices (recipe text, recipe ingredient-pooled, ingredients)
  - SBERT model (for live query encoding)
  - Word2Vec model (for analogies)
  - Recipe metadata + ID maps
  - In-memory copies of recipe / ingredient vectors
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer


# Paths - change if necssary
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EMB_DIR = _PROJECT_ROOT / "embeddings"
_DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "recipes_clean.pkl"
_SEARCH_DIR = _PROJECT_ROOT / "search"


# Config - model must match 
_SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


# do once
# FAISS indices
_recipe_text_index = faiss.read_index(str(_SEARCH_DIR / "recipe_text_index.faiss"))
_recipe_ing_index = faiss.read_index(str(_SEARCH_DIR / "recipe_ingredient_index.faiss"))
_ingredient_index = faiss.read_index(str(_SEARCH_DIR / "ingredient_index.faiss"))

# Vectors held in memory for direct row access (small enough; <30 MB total)
_recipe_text_vecs = np.load(_EMB_DIR / "recipe_vectors.npy")
_recipe_ing_vecs = np.load(_EMB_DIR / "recipe_vectors_from_ingredients.npy")

# Ingredient vectors: load raw, build a normalized copy mirroring what
# build_indices.py adds to the FAISS index.
_ing_vecs_raw = np.load(_EMB_DIR / "ingredient_vectors.npy")
_norms = np.linalg.norm(_ing_vecs_raw, axis=1, keepdims=True)
_norms[_norms == 0] = 1.0
_ing_vecs_normalized = (_ing_vecs_raw / _norms).astype("float32")

# ID maps. JSON forces dict keys to strings; cast where needed.
with open(_EMB_DIR / "ingredient_id_map.json", encoding="utf-8") as f:
    _ingredient_name_to_id: dict[str, int] = json.load(f)  # name -> row index
with open(_EMB_DIR / "ingredient_id_map_reverse.json", encoding="utf-8") as f:
    _ingredient_id_to_name: dict[int, str] = {int(k): v for k, v in json.load(f).items()}
with open(_EMB_DIR / "recipe_id_map.json", encoding="utf-8") as f:
    _recipe_id_to_row: dict[int, int] = {int(k): int(v) for k, v in json.load(f).items()}
_recipe_row_to_id: dict[int, int] = {v: k for k, v in _recipe_id_to_row.items()}

# Recipe metadata. Row-indexed title list for fast lookup.
_df = pd.read_pickle(_DATA_PATH)
_titles: list[str] = _df["title"].tolist()

# Word2Vec for analogies (gensim handles normalization internally).
_w2v = Word2Vec.load(str(_EMB_DIR / "word2vec.model"))

# SBERT for live query encoding.
_sbert = SentenceTransformer(_SBERT_MODEL_NAME)


# Helpers (private)
_TITLE_NORM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_title(title: str) -> str:
    """Title key for dedup: lowercase, replace non-alnum with space, collapse.

    "Soft Pretzels!", "soft pretzels", "Soft  Pretzels" all map to the same
    key. Aggressive but appropriate for this dataset's near-duplicates.
    """
    return _TITLE_NORM_RE.sub(" ", title.lower()).strip()


def _resolve_ingredient(name: str) -> Optional[str]:
    """Resolve user-typed ingredient name to its canonical vocab key.

    Tries the name verbatim, with spaces -> underscores, and with hyphens
    -> underscores (Agent 1 normalized multi-word ingredients to underscored
    form). Returns the matching key, or None.
    """
    for c in (name, name.replace(" ", "_"), name.replace("-", "_")):
        if c in _ingredient_name_to_id:
            return c
    return None


# api streamlit will call
def is_known_ingredient(name: str) -> Optional[str]:
    """Return canonical form of `name` if it's in the vocab, else None.
    """
    return _resolve_ingredient(name)


def find_similar_recipes_by_text(
    query: str,
    k: int = 10,
    dedup: bool = True,
) -> list[dict]:
    """Encode `query` with SBERT and return top-k recipes.

    Args:
        query: free-text query, e.g. "spicy thai noodles".
        k: number of results to return.
        dedup: if True (default), filter near-duplicate recipe titles.
    Returns:
        list of {'id' (int recipe_id), 'title' (str), 'score' (float)}.
        Score is cosine similarity, roughly in [-1, 1].
    """
    if not query or not query.strip():
        return []
    qvec = _sbert.encode([query], normalize_embeddings=True).astype("float32")
    # Search wider than k when dedup'ing so we have headroom to filter.
    search_k = min(k * 5, _recipe_text_index.ntotal) if dedup else k
    scores, idxs = _recipe_text_index.search(qvec, search_k)

    results: list[dict] = []
    seen: set[str] = set()
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        i = int(i)
        title = _titles[i]
        if dedup:
            key = _normalize_title(title)
            if key in seen:
                continue
            seen.add(key)
        results.append({
            "id": _recipe_row_to_id[i],
            "title": title,
            "score": float(s),
        })
        if len(results) >= k:
            break
    return results


def find_similar_recipes_by_recipe_id(
    recipe_id: int,
    k: int = 10,
    mode: str = "text",
    dedup: bool = True,
) -> list[dict]:
    """Find recipes similar to a known recipe.

    Args:
        recipe_id: must be a valid recipe_id in the dataset.
        k: number of results.
        mode: 'text' uses the SBERT (384-dim) recipe index;
        dedup: filter near-duplicate titles.

    Returns:
        list of {'id', 'title', 'score'}.
    
    bad if not recipe or ingred
    """
    if recipe_id not in _recipe_id_to_row:
        return []
    row = _recipe_id_to_row[recipe_id]

    if mode == "text":
        index, vecs = _recipe_text_index, _recipe_text_vecs
    elif mode == "ingredient":
        index, vecs = _recipe_ing_index, _recipe_ing_vecs
    else:
        raise ValueError(f"mode must be 'text' or 'ingredient', got {mode!r}")

    qvec = vecs[row : row + 1].astype("float32")
    if mode == "ingredient" and float(np.linalg.norm(qvec)) < 1e-6:
        return []

    search_k = min((k + 1) * (5 if dedup else 1), index.ntotal)
    scores, idxs = index.search(qvec, search_k)

    seen: set[str] = {_normalize_title(_titles[row])} if dedup else set()
    results: list[dict] = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        i = int(i)
        if i == row:
            continue
        title = _titles[i]
        if dedup:
            key = _normalize_title(title)
            if key in seen:
                continue
            seen.add(key)
        results.append({
            "id": _recipe_row_to_id[i],
            "title": title,
            "score": float(s),
        })
        if len(results) >= k:
            break
    return results


def find_similar_ingredients(ingredient: str, k: int = 10) -> list[dict]:
    """Top-k ingredients similar to `ingredient` by cosine similarity.
    Name is matched flexibly against the vocab (verbatim, space->underscore,
    hyphen->underscore). Returns [] if not in vocab.
    Returns:
        list of {'name', 'score'}.
    """
    canonical = _resolve_ingredient(ingredient)
    if canonical is None:
        return []
    row = _ingredient_name_to_id[canonical]
    qvec = _ing_vecs_normalized[row : row + 1]
    scores, idxs = _ingredient_index.search(qvec, k + 1)

    results: list[dict] = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        i = int(i)
        if i == row:
            continue
        results.append({
            "name": _ingredient_id_to_name[i],
            "score": float(s),
        })
        if len(results) >= k:
            break
    return results


def find_recipes_by_ingredients(
    ingredients: list[str],
    k: int = 10,
    dedup: bool = True,
) -> list[dict]:
    """Pool the given ingredient vectors and find recipes near the pooled point.
    Ingredients not in vocab are silently skipped. If none of the given
    ingredients are in vocab (or the pooled vector is zero), returns [].
    Args:
        ingredients: list of ingredient names.
        k: number of results.
        dedup: filter near-duplicate titles.
    Returns:
        list of {'id', 'title', 'score'}.
    """
    vecs: list[np.ndarray] = []
    for ing in ingredients:
        canonical = _resolve_ingredient(ing)
        if canonical is None:
            continue
        row = _ingredient_name_to_id[canonical]
        vecs.append(_ing_vecs_normalized[row])
    if not vecs:
        return []

    pooled = np.mean(vecs, axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm < 1e-9:
        return []
    pooled = (pooled / norm).astype("float32").reshape(1, -1)

    search_k = min(k * 5, _recipe_ing_index.ntotal) if dedup else k
    scores, idxs = _recipe_ing_index.search(pooled, search_k)

    seen: set[str] = set()
    results: list[dict] = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        i = int(i)
        title = _titles[i]
        if dedup:
            key = _normalize_title(title)
            if key in seen:
                continue
            seen.add(key)
        results.append({
            "id": _recipe_row_to_id[i],
            "title": title,
            "score": float(s),
        })
        if len(results) >= k:
            break
    return results


def ingredient_analogy(
    positive: list[str],
    negative: Optional[list[str]] = None,
    k: int = 5,
) -> list[dict]:
    """Vector arithmetic: sum(positive) - sum(negative) -> top-k ingredients.
    Args:
        positive: terms to add (must be non-empty).
        negative: terms to subtract (may be empty / None).
        k: number of results.

    Returns:
        list of {'name', 'score'}.
    """
    negative = negative or []
    if not positive:
        return []

    pos_canonical: list[str] = []
    for t in positive:
        c = _resolve_ingredient(t)
        if c is None or c not in _w2v.wv:
            return []
        pos_canonical.append(c)

    neg_canonical: list[str] = []
    for t in negative:
        c = _resolve_ingredient(t)
        if c is None or c not in _w2v.wv:
            return []
        neg_canonical.append(c)

    try:
        results = _w2v.wv.most_similar(
            positive=pos_canonical,
            negative=neg_canonical,
            topn=k,
        )
    except (KeyError, ValueError):
        return []
    return [{"name": str(name), "score": float(score)} for name, score in results]