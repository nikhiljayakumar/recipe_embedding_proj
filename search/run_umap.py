"""run_umap.py — Reduce recipe and ingredient vectors to 2D with UMAP.

outputs:
  - umap_recipes.npy          (N, 2) float32
  - umap_recipes.csv          recipe_id, x, y, title
  - umap_ingredients.npy      (V, 2) float32
  - umap_ingredients.csv      name, x, y

Parameters per the spec:
  - metric='cosine' (matches similarity used everywhere else)
  - n_neighbors=15 for recipes (default; balanced local/global structure)
  - n_neighbors=10 for ingredients (smaller vocab, tighter neighborhoods)
  - random_state=42 for reproducibility
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap

# change these depending on environment/config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = PROJECT_ROOT / "embeddings"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "recipes_clean.pkl"
VIS_DIR = PROJECT_ROOT / "search" / "visualization"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ---- Recipes (SBERT, 384-dim, already normalized) ----
    print("Running UMAP on recipes (SBERT, 384-dim)...")
    recipe_vecs = np.load(EMB_DIR / "recipe_vectors.npy")
    print(f"  input shape: {recipe_vecs.shape}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    recipe_2d = reducer.fit_transform(recipe_vecs).astype("float32")
    np.save(VIS_DIR / "umap_recipes.npy", recipe_2d)

    # CSV with titles for inspection
    df = pd.read_pickle(DATA_PATH)
    assert len(df) == recipe_2d.shape[0], (
        f"recipe count mismatch: df has {len(df)} rows, vectors have {recipe_2d.shape[0]}"
    )
    with open(EMB_DIR / "recipe_id_map.json", encoding="utf-8") as f:
        rid_to_row = {int(k): int(v) for k, v in json.load(f).items()}
    row_to_rid = {v: k for k, v in rid_to_row.items()}
    recipe_ids = [row_to_rid[i] for i in range(recipe_2d.shape[0])]

    pd.DataFrame({
        "recipe_id": recipe_ids,
        "x": recipe_2d[:, 0],
        "y": recipe_2d[:, 1],
        "title": df["title"].tolist(),
    }).to_csv(VIS_DIR / "umap_recipes.csv", index=False)
    print(f"  saved {recipe_2d.shape[0]} recipe coords")

    # ---- Ingredients (W2V, 100-dim, normalize before reducing) ----
    ing_raw = np.load(EMB_DIR / "ingredient_vectors.npy")
    norms = np.linalg.norm(ing_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    ing_norm = (ing_raw / norms).astype("float32")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    ing_2d = reducer.fit_transform(ing_norm).astype("float32")
    np.save(VIS_DIR / "umap_ingredients.npy", ing_2d)

    with open(EMB_DIR / "ingredient_id_map_reverse.json", encoding="utf-8") as f:
        id_to_name = {int(k): v for k, v in json.load(f).items()}
    names = [id_to_name[i] for i in range(ing_2d.shape[0])]

    pd.DataFrame({
        "name": names,
        "x": ing_2d[:, 0],
        "y": ing_2d[:, 1],
    }).to_csv(VIS_DIR / "umap_ingredients.csv", index=False)
    print("\nUMAP done. Outputs in search/visualization/")


if __name__ == "__main__":
    main()