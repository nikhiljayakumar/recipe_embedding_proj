"""
02_encode_sbert.py
------------------
Encodes each recipe (title + ingredients + instructions) with SBERT
(all-MiniLM-L6-v2) into a 384-dim vector.

Vectors are L2-normalized so that Agent 3 can use FAISS IndexFlatIP and
get cosine similarity for free. THIS IS THE SINGLE MOST COMMON FAISS BUG;
do not change `normalize_embeddings=True` without telling Agent 3.

Outputs (in OUTPUT_DIR):
    - recipe_vectors.npy : (N, 384) float32, L2-normalized
    - recipe_id_map.json : {recipe_id: row_index}, identity if Agent 1 was clean

Runtime: ~10-20 min on CPU for 50k recipes, ~1 min on GPU.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- CONFIG: adjust these for your local layout -----------------------------
DATA_PATH = Path("./data/processed/recipes_clean.pkl")
OUTPUT_DIR = Path("./embeddings")

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, well-tested
BATCH_SIZE = 128                   # CPU-friendly; bump to 256 on GPU
INSTRUCTION_CHAR_LIMIT = 500       # SBERT silently truncates at 256 tokens; be explicit
# ----------------------------------------------------------------------------


def build_recipe_text(row) -> str:
    """Combine title + ingredients + instructions into one string for SBERT."""
    title = str(row.title) if pd.notna(row.title) else ""
    ingredients = ", ".join(row.normalized_ingredients) if row.normalized_ingredients else ""
    instructions = str(row.instructions)[:INSTRUCTION_CHAR_LIMIT] if pd.notna(row.instructions) else ""
    return f"{title}. Ingredients: {ingredients}. Instructions: {instructions}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading recipes from {DATA_PATH}...")
    df = pd.read_pickle(DATA_PATH)

    # --- Verify Agent 1's contract (same checks as 01) ---------------------
    assert df["recipe_id"].is_monotonic_increasing, "recipe_id is not sorted"
    assert df["recipe_id"].iloc[0] == 0, "recipe_id should start at 0"
    assert df["recipe_id"].iloc[-1] == len(df) - 1, "recipe_id is not contiguous"
    print(f"  OK -- {len(df)} recipes")

    # --- Build text -------------------------------------------------------
    print("Building recipe texts...")
    texts = [build_recipe_text(row) for row in df.itertuples()]
    print(f"  Sample text: {texts[0][:200]}...")

    # --- Load SBERT (auto-detects CPU/GPU) ---------------------------------
    print(f"Loading {MODEL_NAME}...")
    sbert_model = SentenceTransformer(MODEL_NAME)
    # Note: not calling .to('cuda') -- SentenceTransformer auto-uses GPU if
    # available, otherwise CPU. Leave as-is unless you want to force a device.

    # --- Encode -----------------------------------------------------------
    print(f"Encoding {len(texts)} recipes (batch_size={BATCH_SIZE})...")
    recipe_vectors = sbert_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # CRITICAL: cosine == inner-product after this
    ).astype(np.float32)

    # --- Sanity check normalization ---------------------------------------
    norms = np.linalg.norm(recipe_vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), (
        f"Vectors are not L2-normalized! norms range: {norms.min():.4f} to {norms.max():.4f}"
    )
    print(f"  Verified L2-normalized (all norms ~= 1.0)")

    # --- Build identity recipe_id -> row map ------------------------------
    recipe_id_map = {int(rid): int(rid) for rid in df["recipe_id"]}

    # --- Save -------------------------------------------------------------
    np.save(OUTPUT_DIR / "recipe_vectors.npy", recipe_vectors)
    with open(OUTPUT_DIR / "recipe_id_map.json", "w", encoding="utf-8") as f:
        json.dump(recipe_id_map, f)

    print(f"\nSaved to {OUTPUT_DIR.resolve()}:")
    print(f"  recipe_vectors.npy   shape={recipe_vectors.shape} dtype={recipe_vectors.dtype}")
    print(f"  recipe_id_map.json   {len(recipe_id_map)} entries (identity map)")


if __name__ == "__main__":
    main()