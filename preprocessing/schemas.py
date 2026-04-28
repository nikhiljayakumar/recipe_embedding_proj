"""
Schema contract for recipes_clean.pkl.

This is the binding contract between Agent 1 (data) and Agents 2, 3, 4.
Downstream agents should import constants from this module rather than
hardcoding column name strings, and should call validate(df) immediately
after loading the artifact to fail fast on contract violations.

Bumping SCHEMA_VERSION is a breaking change. Coordinate before bumping.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.0"

# ---- Column names -----------------------------------------------------------
RECIPE_ID = "recipe_id"
TITLE = "title"
NORMALIZED_INGREDIENTS = "normalized_ingredients"
RAW_INGREDIENTS = "raw_ingredients"
INSTRUCTIONS = "instructions"
INGREDIENT_COUNT = "ingredient_count"

ALL_COLUMNS = [
    RECIPE_ID,
    TITLE,
    NORMALIZED_INGREDIENTS,
    RAW_INGREDIENTS,
    INSTRUCTIONS,
    INGREDIENT_COUNT,
]

# ---- Implementer notes ------------------------------------------------------
#
# normalized_ingredients: list[str]. Each element is a complete multi-word
#   ingredient name with spaces preserved (e.g., "olive oil"). DO NOT
#   underscore-join. Gensim's Word2Vec treats each list element as a single
#   token, so "olive oil" stays as one vocabulary entry (correct for the
#   food domain — "olive oil" is semantically distinct from "vegetable oil").
#
# raw_ingredients: list[str]. The original strings before normalization,
#   preserved for human display in the Streamlit demo (Agent 4).
#
# recipe_id: contiguous integer index 0..N-1. Agent 2's embedding matrix
#   rows are aligned to this. DO NOT shuffle, drop, or reindex this column
#   without coordinating with Agent 2.
#
# Minimum recipe size: every recipe has >= 3 normalized ingredients.
# Minimum ingredient document frequency: every ingredient appearing in
#   normalized_ingredients was present in >= 5 recipes (post-dedup).


def validate(df) -> None:
    """
    Validate that a DataFrame matches the v1.0 contract.

    Raises AssertionError on violation. Cheap to call (samples one row
    for type checks rather than iterating the full frame).
    """
    import pandas as pd

    assert isinstance(df, pd.DataFrame), f"expected DataFrame, got {type(df)}"

    # Columns present
    missing = set(ALL_COLUMNS) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"

    n = len(df)
    assert n > 0, "DataFrame is empty"

    # recipe_id contiguous 0..N-1
    ids = df[RECIPE_ID].to_numpy()
    assert ids.dtype.kind in ("i", "u"), (
        f"{RECIPE_ID} must be integer dtype, got {ids.dtype}"
    )
    assert int(ids.min()) == 0, f"{RECIPE_ID} min must be 0, got {ids.min()}"
    assert int(ids.max()) == n - 1, (
        f"{RECIPE_ID} max must be {n - 1}, got {ids.max()}"
    )
    assert len(set(ids.tolist())) == n, f"{RECIPE_ID} must be unique"

    # ingredient_count integer
    counts = df[INGREDIENT_COUNT].to_numpy()
    assert counts.dtype.kind in ("i", "u"), (
        f"{INGREDIENT_COUNT} must be integer dtype, got {counts.dtype}"
    )
    assert int(counts.min()) >= 3, (
        f"some recipes have <3 ingredients (min={counts.min()}); "
        "should have been filtered"
    )

    # Type checks on a sample row
    sample = df.iloc[0]
    assert isinstance(sample[TITLE], str), f"{TITLE} must be str"
    assert isinstance(sample[INSTRUCTIONS], str), f"{INSTRUCTIONS} must be str"
    assert isinstance(sample[NORMALIZED_INGREDIENTS], list), (
        f"{NORMALIZED_INGREDIENTS} must be list"
    )
    assert isinstance(sample[RAW_INGREDIENTS], list), (
        f"{RAW_INGREDIENTS} must be list"
    )
    assert all(isinstance(x, str) for x in sample[NORMALIZED_INGREDIENTS]), (
        f"{NORMALIZED_INGREDIENTS} elements must all be str"
    )
    assert all(isinstance(x, str) for x in sample[RAW_INGREDIENTS]), (
        f"{RAW_INGREDIENTS} elements must all be str"
    )

    # ingredient_count consistency (full-frame check, vectorised)
    list_lens = df[NORMALIZED_INGREDIENTS].map(len)
    assert (df[INGREDIENT_COUNT] == list_lens).all(), (
        f"{INGREDIENT_COUNT} does not match len({NORMALIZED_INGREDIENTS}) "
        "for some rows"
    )


if __name__ == "__main__":
    # CLI: python schemas.py path/to/recipes_clean.pkl
    import sys
    import pandas as pd

    if len(sys.argv) != 2:
        print("usage: python schemas.py <path-to-recipes_clean.pkl>")
        sys.exit(1)

    df = pd.read_pickle(sys.argv[1])
    validate(df)
    print(f"OK: {len(df):,} rows, schema v{SCHEMA_VERSION}")
