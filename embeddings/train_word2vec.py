"""
train_word2vec.py
--------------------
Trains a skip-gram Word2Vec model on ingredient co-occurrence.

Each recipe is treated as a "sentence" of ingredient tokens. Ingredients
that co-occur in similar recipes end up with similar vectors. This is the
right tool for single-token ingredient embeddings -- it's literally what
Word2Vec was designed for.

Outputs (in OUTPUT_DIR):
    - word2vec.model               : full gensim model 
    - ingredient_vectors.npy       : (V, 100) numpy array, RAW (unnormalized)
    - ingredient_id_map.json       : {ingredient_name: row_index}
    - ingredient_id_map_reverse.json : {row_index: ingredient_name}

Vectors are saved RAW (not L2-normalized). gensim's most_similar expects
unnormalized vectors and handles normalization internally.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# --- CONFIG: adjust these for your local layout -----------------------------
DATA_PATH = Path("./data/processed/recipes_clean.pkl")
OUTPUT_DIR = Path("./embeddings")

VECTOR_SIZE = 100   # standard sweet spot for 5k+ vocab
WINDOW = 10         # ingredient order in a recipe is mostly arbitrary
MIN_COUNT = 5       # safety net; filterd by preprocessing 
SG = 1              # skip-gram, better than CBOW for small-ish vocabs
EPOCHS = 20         # 50k recipes is small by NLP standards; more epochs helps
WORKERS = 4         # bump to 8 if you have a beefy CPU
SEED = 42
# ----------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading recipes from {DATA_PATH}...")
    df = pd.read_pickle(DATA_PATH)

    assert "recipe_id" in df.columns, "Missing recipe_id column"
    assert "normalized_ingredients" in df.columns, "Missing normalized_ingredients"
    assert df["recipe_id"].is_monotonic_increasing, "recipe_id is not sorted"
    assert df["recipe_id"].iloc[0] == 0, "recipe_id should start at 0"
    assert df["recipe_id"].iloc[-1] == len(df) - 1, "recipe_id is not contiguous"
    assert df["normalized_ingredients"].notna().all(), "Null ingredients found"
    sample = df["normalized_ingredients"].iloc[0]
    assert isinstance(sample, list), (
        f"normalized_ingredients must be lists of strings, got {type(sample)}. "
    )
    assert all(isinstance(t, str) for t in sample), "Ingredient tokens must be strings"
    print(f"  OK -- {len(df)} recipes, sample ingredients: {sample[:5]}")

    # --- Train Word2Vec ----------------------------------------------------
    sentences = df["normalized_ingredients"].tolist()
    print(
        f"Training Word2Vec (vector_size={VECTOR_SIZE}, window={WINDOW}, "
        f"sg={SG}, epochs={EPOCHS}, min_count={MIN_COUNT})..."
    )
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
        epochs=EPOCHS,
        seed=SEED,
    )
    print(f"  Done. Vocabulary size: {len(w2v_model.wv.key_to_index)}")

    # --- Extract vectors in deterministic (alphabetical) order -------------
    vocab = sorted(w2v_model.wv.key_to_index.keys())
    ingredient_vectors = np.stack([w2v_model.wv[w] for w in vocab]).astype(np.float32)
    ingredient_id_map = {w: i for i, w in enumerate(vocab)}
    ingredient_id_map_reverse = {str(i): w for i, w in enumerate(vocab)}

    np.save(OUTPUT_DIR / "ingredient_vectors.npy", ingredient_vectors)
    with open(OUTPUT_DIR / "ingredient_id_map.json", "w", encoding="utf-8") as f:
        json.dump(ingredient_id_map, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "ingredient_id_map_reverse.json", "w", encoding="utf-8") as f:
        json.dump(ingredient_id_map_reverse, f, ensure_ascii=False, indent=2)
    w2v_model.save(str(OUTPUT_DIR / "word2vec.model"))

    print(f"\nSaved to {OUTPUT_DIR.resolve()}:")
    print(f"  ingredient_vectors.npy        shape={ingredient_vectors.shape} dtype={ingredient_vectors.dtype}")
    print(f"  ingredient_id_map.json        {len(ingredient_id_map)} entries")
    print(f"  ingredient_id_map_reverse.json")
    print(f"  word2vec.model")

    # --- sanity check --------------------------------------------------
    print("\n--- Quick nearest-neighbor smell test ---")
    for probe in ["basil", "soy sauce", "butter", "garlic", "lime"]:
        if probe in w2v_model.wv:
            neighbors = w2v_model.wv.most_similar(probe, topn=5)
            pretty = ", ".join(f"{w} ({s:.2f})" for w, s in neighbors)
            print(f"  {probe:12s} -> {pretty}")
        else:
            print(f"  {probe:12s} -> [not in vocab]")


if __name__ == "__main__":
    main()