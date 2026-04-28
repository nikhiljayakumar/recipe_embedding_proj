"""build_indices.py — Build the three FAISS indices used by search.py.

Run once after Agent 2's embeddings land in /embeddings/. Produces:
  - search/recipe_text_index.faiss        (SBERT, 384-dim, IndexFlatIP)
  - search/recipe_ingredient_index.faiss  (pooled W2V, 100-dim, IndexFlatIP)
  - search/ingredient_index.faiss         (W2V ingredients, 100-dim, IndexFlatIP)

All three indices use IndexFlatIP, which on L2-normalized vectors gives
exact cosine similarity. At ~50k vectors, flat is plenty fast — no IVF / HNSW.

Round-trip sanity checks (query the index with its own vectors; top-1 should
be self) are run on each index and printed.

Usage: python build_indices.py
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = PROJECT_ROOT / "embeddings"
SEARCH_DIR = PROJECT_ROOT / "search"
SEARCH_DIR.mkdir(parents=True, exist_ok=True)


def round_trip_check(index: faiss.Index, vecs: np.ndarray, name: str, sample: int = 5) -> None:
    """Query the index with a few of its own vectors; top-1 should be self."""
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(vecs.shape[0], size=min(sample, vecs.shape[0]), replace=False)
    qry = vecs[sample_ids].astype("float32")
    scores, idxs = index.search(qry, 1)
    hits = int((idxs[:, 0] == sample_ids).sum())
    print(
        f"  [{name}] round-trip top-1 self-match: {hits}/{sample}"
        f"  (top-1 score min={scores[:, 0].min():.4f}, max={scores[:, 0].max():.4f})"
    )
    if hits != sample:
        print(f"  WARNING: round-trip check failed for {name} on {sample - hits} samples")


def main() -> None:
    print("Building FAISS indices...")

    # ---- 1. Recipe text (SBERT, 384-dim, already normalized) ----
    print("\n[1/3] recipe_text_index — SBERT recipe vectors")
    recipe_text = np.load(EMB_DIR / "recipe_vectors.npy")
    print(f"  shape: {recipe_text.shape}, dtype: {recipe_text.dtype}")
    n0 = float(np.linalg.norm(recipe_text[0]))
    print(f"  norm of vec[0]: {n0:.4f}  (expected ~1.0; SBERT contract is normalized)")
    if not (0.99 <= n0 <= 1.01):
        print("  WARNING: SBERT vectors do not look L2-normalized. Talk to Agent 2 — do NOT normalize here.")
    recipe_text_f32 = recipe_text.astype("float32")
    text_index = faiss.IndexFlatIP(recipe_text_f32.shape[1])
    text_index.add(recipe_text_f32)
    out_path = SEARCH_DIR / "recipe_text_index.faiss"
    faiss.write_index(text_index, str(out_path))
    print(f"  saved: {out_path}  (ntotal={text_index.ntotal})")
    round_trip_check(text_index, recipe_text_f32, "recipe_text")

    # ---- 2. Recipe ingredient-pooled (W2V, 100-dim, already normalized) ----
    print("\n[2/3] recipe_ingredient_index — pooled W2V recipe vectors")
    recipe_ing = np.load(EMB_DIR / "recipe_vectors_from_ingredients.npy")
    print(f"  shape: {recipe_ing.shape}, dtype: {recipe_ing.dtype}")
    norms = np.linalg.norm(recipe_ing, axis=1)
    n_zero = int((norms < 1e-6).sum())
    print(f"  norm of vec[0]: {norms[0]:.4f}  (expected ~1.0; Agent 2 normalized)")
    print(f"  zero-vector recipes (filtered ingredients): {n_zero}")
    recipe_ing_f32 = recipe_ing.astype("float32")
    ing_pool_index = faiss.IndexFlatIP(recipe_ing_f32.shape[1])
    ing_pool_index.add(recipe_ing_f32)
    out_path = SEARCH_DIR / "recipe_ingredient_index.faiss"
    faiss.write_index(ing_pool_index, str(out_path))
    print(f"  saved: {out_path}  (ntotal={ing_pool_index.ntotal})")
    round_trip_check(ing_pool_index, recipe_ing_f32, "recipe_ingredient")

    # ---- 3. Ingredient (W2V, 100-dim, NOT normalized — we normalize here) ----
    print("\n[3/3] ingredient_index — W2V ingredient vectors")
    ing_raw = np.load(EMB_DIR / "ingredient_vectors.npy")
    print(f"  shape: {ing_raw.shape}, dtype: {ing_raw.dtype}")
    print(f"  norm of vec[0]: {np.linalg.norm(ing_raw[0]):.4f}  (expected NOT ~1.0; raw W2V)")
    norms = np.linalg.norm(ing_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    ing_norm = (ing_raw / norms).astype("float32")
    print(f"  after normalization, norm of vec[0]: {np.linalg.norm(ing_norm[0]):.4f}")
    ing_index = faiss.IndexFlatIP(ing_norm.shape[1])
    ing_index.add(ing_norm)
    out_path = SEARCH_DIR / "ingredient_index.faiss"
    faiss.write_index(ing_index, str(out_path))
    print(f"  saved: {out_path}  (ntotal={ing_index.ntotal})")
    round_trip_check(ing_index, ing_norm, "ingredient")

    print("\nAll three indices built successfully.")


if __name__ == "__main__":
    main()