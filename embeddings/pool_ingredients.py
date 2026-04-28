"""
03_pool_ingredients.py
----------------------
Computes a (N, 100) recipe vector by mean-pooling each recipe's ingredient
Word2Vec vectors. This gives Agent 3 a third axis of comparison ("ingredient
palette" vs SBERT's "text style") and often clusters cuisine better than
SBERT, which is gold for the blog post.

Outputs (in OUTPUT_DIR):
    - recipe_vectors_from_ingredients.npy : (N, 100) float32, L2-normalized

Run AFTER 01_train_word2vec.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# --- CONFIG: adjust these for your local layout -----------------------------
DATA_PATH = Path("./data/processed/recipes_clean.pkl")
OUTPUT_DIR = Path("./embeddings")
# ----------------------------------------------------------------------------


def main() -> None:
    print(f"Loading Word2Vec model from {OUTPUT_DIR / 'word2vec.model'}...")
    w2v_model = Word2Vec.load(str(OUTPUT_DIR / "word2vec.model"))
    vector_size = w2v_model.wv.vector_size

    print(f"Loading recipes from {DATA_PATH}...")
    df = pd.read_pickle(DATA_PATH)
    print(f"  {len(df)} recipes")

    # --- Pool --------------------------------------------------------------
    print("Pooling ingredient vectors per recipe...")
    pooled = np.zeros((len(df), vector_size), dtype=np.float32)
    n_empty = 0
    for i, ingredients in enumerate(df["normalized_ingredients"]):
        vecs = [w2v_model.wv[ing] for ing in ingredients if ing in w2v_model.wv]
        if vecs:
            pooled[i] = np.mean(vecs, axis=0)
        else:
            n_empty += 1
            # leave as zeros; will stay zero after normalization (we guard below)

    if n_empty:
        print(
            f"  WARNING: {n_empty} recipes had zero in-vocab ingredients and "
            "are stored as zero vectors. They will return as 'no match' in "
            "cosine search, which is fine."
        )

    # --- L2 normalize (with zero-vector guard) -----------------------------
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0   # don't divide by zero on empty recipes
    pooled = pooled / norms
    pooled = pooled.astype(np.float32)

    # --- Save --------------------------------------------------------------
    out = OUTPUT_DIR / "recipe_vectors_from_ingredients.npy"
    np.save(out, pooled)
    print(f"\nSaved to {out.resolve()}")
    print(f"  shape={pooled.shape} dtype={pooled.dtype}")
    print(f"  norms: min={np.linalg.norm(pooled, axis=1).min():.4f} "
          f"max={np.linalg.norm(pooled, axis=1).max():.4f} "
          f"(zeros are recipes with no in-vocab ingredients)")


if __name__ == "__main__":
    main()