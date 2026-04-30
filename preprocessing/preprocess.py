"""
python preprocess.py --source ./path/to/full_dataset.csv

Outputs to ./data/:
  - recipes_clean.pkl      (canonical artifact)
  - recipes_clean.parquet  (inspection / portability copy)
  - ingredient_vocab.json  (ingredient -> document frequency)
  - preprocessing_report.md
  - schemas.py             (copied here for downstream import convenience)

Normalization uses RecipeNLG's NER column (already stripped of quantities/units)
as the starting point, then applies: lowercase --> singularize --> synonym map.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import logging
import random
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

# Local
from schemas import (
    ALL_COLUMNS,
    INGREDIENT_COUNT,
    INSTRUCTIONS,
    NORMALIZED_INGREDIENTS,
    RAW_INGREDIENTS,
    RECIPE_ID,
    SCHEMA_VERSION,
    TITLE,
    validate,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


# =============================================================================
# Normalization helpers (single-threaded, applied to NER tokens)
# =============================================================================

# Words that look plural but aren't, OR are mass-nouns we keep as-is.
# Also includes already-singular -us / -is words inflect would mangle.
UNCOUNTABLE = {
    "molasses", "couscous", "asparagus", "hummus", "swiss", "brussels",
    "octopus", "watercress", "anise", "licorice",
    "oats", "greens", "chives",  # plural-mass nouns to leave alone
    # "leaves" is intentionally absent: inflect → "leaf" is correct and
    # enables "coriander leaves" → "coriander leaf" → (synonym) "cilantro".
}

# Hand-curated synonym map. Keys are post-singularization forms.
# Each entry has a one-line rationale. Keep this list short — Word2Vec
# recovers most synonymy from co-occurrence on its own.
SYNONYMS = {
    # Onion family — three names for the same plant part.
    "scallion": "green onion",
    "spring onion": "green onion",

    # UK -> US naming
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "rocket": "arugula",
    # NOTE: we map "coriander leaves" -> "cilantro" but NOT "coriander" itself,
    # because in US recipes "coriander" alone usually means the seed/spice.
    "coriander leaf": "cilantro",
    "fresh coriander": "cilantro",

    # Sugar variants
    "confectioners sugar": "powdered sugar",
    "confectioner's sugar": "powdered sugar",
    "icing sugar": "powdered sugar",

    # Flour
    "plain flour": "all-purpose flour",
    "ap flour": "all-purpose flour",

    # Leavening
    "bicarbonate of soda": "baking soda",
    "bicarb": "baking soda",

    # Cream variants — all are >36% milkfat, used interchangeably
    "double cream": "heavy cream",
    "heavy whipping cream": "heavy cream",
    "whipping cream": "heavy cream",

    # Beans
    "garbanzo bean": "chickpea",
    "garbanzo": "chickpea",

    # Tomato products (judgment call — passata and puree are close enough)
    "tomato ketchup": "ketchup",
    "passata": "tomato sauce",
    "tomato puree": "tomato sauce",
}


# Lazy-init inflect engine (slow to construct, only need one)
_INFLECT = None
def _get_inflect():
    global _INFLECT
    if _INFLECT is None:
        import inflect
        _INFLECT = inflect.engine()
    return _INFLECT


def _singularize_last_word(name: str) -> str:
    """Singularize the last word of a multi-word ingredient name."""
    parts = name.split()
    if not parts:
        return name
    last = parts[-1]
    if last in UNCOUNTABLE or len(last) <= 3:
        return name
    sing = _get_inflect().singular_noun(last)
    if not sing:
        return name
    # Reject pathological singularizations (e.g., "molass" from "molasses").
    # Heuristic: a real plural -> singular shouldn't lop off more than 3 chars.
    if len(sing) < len(last) - 3:
        return name
    parts[-1] = sing
    return " ".join(parts)


_WS_RE = re.compile(r"\s+")

def post_normalize(name: str) -> str:
    """
    Normalize one NER token: lowercase → singularize → synonym map.
    Returns "" for tokens that should be dropped.
    """
    if not name:
        return ""
    name = unicodedata.normalize("NFKC", name).strip().lower()
    name = _WS_RE.sub(" ", name)
    name = _singularize_last_word(name)
    name = SYNONYMS.get(name, name)
    if len(name) <= 1:
        return ""
    if not any(c.isalpha() for c in name):
        return ""
    return name


# =============================================================================
# Phase 1: Load + initial cleaning
# =============================================================================

def _clean_text(s: str) -> str:
    """HTML unescape, NFKC normalize, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _coerce_list_field(val: Any) -> list[str]:
    """RecipeNLG list fields might be list, JSON string, or Python repr."""
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        # Try JSON first, fall back to ast.literal_eval (handles Python repr)
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
        # Last resort: treat as a single-item list (single string)
        return [val]
    return []


def load_recipenlg(source: str, max_recipes: int, seed: int) -> pd.DataFrame:
    """
    Load RecipeNLG. `source` may be a HuggingFace dataset name (e.g.,
    'mbien/recipe_nlg') or a path to a local CSV.
    """
    log.info("Loading from %s", source)
    if source.endswith(".csv") or Path(source).suffix == ".csv":
        df = pd.read_csv(source)
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            log.error("datasets not installed: %s", e)
            log.error("install it (`pip install datasets`) or pass --source <csv path>")
            raise SystemExit(1)
        ds = load_dataset(source, split="train", trust_remote_code=True)
        df = ds.to_pandas()

    log.info("Loaded %s rows", f"{len(df):,}")

    # Subsample BEFORE preprocessing
    if len(df) > max_recipes:
        df = df.sample(n=max_recipes, random_state=seed).reset_index(drop=True)
        log.info("Subsampled to %s rows (seed=%d)", f"{len(df):,}", seed)

    # Coerce column names. RecipeNLG variants:
    #   HF mbien/recipe_nlg: title, ingredients, directions, ner, link, source
    #   Original CSV:        title, ingredients, directions, ner, link, source
    # We accept either 'directions' or 'instructions'.
    rename = {}
    if "directions" in df.columns and "instructions" not in df.columns:
        rename["directions"] = "instructions"
    if rename:
        df = df.rename(columns=rename)

    required = {"title", "ingredients", "instructions", "NER"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Source columns {list(df.columns)} missing required {missing}"
        )

    return df


def phase_1_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows, clean text, coerce list fields."""
    log.info("[Phase 1] Initial cleaning")
    n0 = len(df)

    df = df.copy()
    df["ingredients"] = df["ingredients"].apply(_coerce_list_field)
    df["instructions"] = df["instructions"].apply(
        lambda v: " ".join(_coerce_list_field(v)) if not isinstance(v, str) else v
    )

    df[TITLE] = df["title"].apply(_clean_text)
    df[INSTRUCTIONS] = df["instructions"].apply(_clean_text)
    df[RAW_INGREDIENTS] = df["ingredients"].apply(
        lambda lst: [_clean_text(s) for s in lst if isinstance(s, str)]
    )
    # NER tokens are already stripped of quantities/units by RecipeNLG
    df["_ner"] = df["NER"].apply(_coerce_list_field).apply(
        lambda lst: [_clean_text(s) for s in lst if isinstance(s, str) and s.strip()]
    )

    mask = (
        (df[TITLE].str.len() > 0)
        & (df[INSTRUCTIONS].str.len() > 0)
        & (df[RAW_INGREDIENTS].str.len() > 0)
        & (df["_ner"].str.len() > 0)
    )
    df = df[mask].reset_index(drop=True)
    log.info("  dropped %s rows with missing fields (%s -> %s)",
             f"{n0 - len(df):,}", f"{n0:,}", f"{len(df):,}")
    return df[[TITLE, INSTRUCTIONS, RAW_INGREDIENTS, "_ner"]]


# =============================================================================
# Phase 2: Normalize NER tokens
# =============================================================================

def phase_2_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize NER tokens: lowercase → singularize → synonym map."""
    log.info("[Phase 2] Normalizing NER tokens")

    all_total = sum(len(lst) for lst in df["_ner"])
    unique_tokens = sorted({tok for lst in df["_ner"] for tok in lst})
    log.info("  %s total NER tokens, %s unique", f"{all_total:,}", f"{len(unique_tokens):,}")

    # Normalize unique tokens only, then fan back out (fast, single-threaded)
    norm_cache: dict[str, str] = {tok: post_normalize(tok) for tok in unique_tokens}

    def fan_out(ner_list: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for tok in ner_list:
            n = norm_cache.get(tok, "")
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    df = df.copy()
    df[NORMALIZED_INGREDIENTS] = df["_ner"].apply(fan_out)
    df = df.drop(columns=["_ner"])

    n_before = len(df)
    df = df[df[NORMALIZED_INGREDIENTS].str.len() > 0].reset_index(drop=True)
    if n_before != len(df):
        log.info("  dropped %s rows with no normalizable NER tokens",
                 f"{n_before - len(df):,}")

    return df


# =============================================================================
# Phase 3: Dedupe
# =============================================================================

def phase_3_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe on (frozenset(ingredients), len(instructions) // 100)."""
    log.info("[Phase 3] Deduplication")
    n0 = len(df)

    df = df.copy()
    df["_dedup_key"] = [
        (frozenset(row), len(instr) // 100)
        for row, instr in zip(df[NORMALIZED_INGREDIENTS], df[INSTRUCTIONS])
    ]
    df["_instr_len"] = df[INSTRUCTIONS].str.len()
    df = (
        df.sort_values("_instr_len", ascending=False)
        .drop_duplicates(subset="_dedup_key", keep="first")
        .drop(columns=["_dedup_key", "_instr_len"])
        .reset_index(drop=True)
    )

    log.info("  removed %s duplicates (%.1f%%): %s -> %s",
             f"{n0 - len(df):,}", 100 * (n0 - len(df)) / max(n0, 1),
             f"{n0:,}", f"{len(df):,}")
    return df


# =============================================================================
# Phase 4: Vocab filtering
# =============================================================================

def _doc_freq(df: pd.DataFrame) -> Counter:
    counter: Counter = Counter()
    for ingredients in df[NORMALIZED_INGREDIENTS]:
        # set() so each ingredient counts at most once per recipe
        counter.update(set(ingredients))
    return counter


def phase_4_filter_vocab(df: pd.DataFrame, min_df: int, min_ingredients: int) -> pd.DataFrame:
    log.info("[Phase 4] Vocab filtering (min_df=%d, min_ingredients=%d)",
             min_df, min_ingredients)
    n0 = len(df)

    # Count document frequency on dedup'd data
    counter = _doc_freq(df)
    keep = {ing for ing, c in counter.items() if c >= min_df}
    log.info("  vocab: %s ingredients total, %s after min_df filter",
             f"{len(counter):,}", f"{len(keep):,}")

    # Filter rare ingredients OUT OF the lists (recipes stay)
    df = df.copy()
    df[NORMALIZED_INGREDIENTS] = df[NORMALIZED_INGREDIENTS].apply(
        lambda lst: [ing for ing in lst if ing in keep]
    )

    # Drop recipes that fell below min_ingredients
    n_before_drop = len(df)
    df = df[df[NORMALIZED_INGREDIENTS].str.len() >= min_ingredients].reset_index(drop=True)
    log.info("  dropped %s recipes with <%d ingredients (%s -> %s)",
             f"{n_before_drop - len(df):,}", min_ingredients,
             f"{n_before_drop:,}", f"{len(df):,}")
    log.info("  Phase 4 net: %s -> %s", f"{n0:,}", f"{len(df):,}")

    return df


# =============================================================================
# Phase 5: Final assembly + save
# =============================================================================

def phase_5_finalize(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, Counter]:
    log.info("[Phase 5] Finalize + save")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Assign contiguous recipe_id LAST (after all filtering)
    df = df.reset_index(drop=True)
    df[RECIPE_ID] = df.index.astype("int64")
    df[INGREDIENT_COUNT] = df[NORMALIZED_INGREDIENTS].str.len().astype("int64")
    df = df[ALL_COLUMNS]

    # Validate before writing — fail loud if we screwed up
    validate(df)
    log.info("  schema validation: PASSED")

    # Recompute final document frequency for the JSON artifact
    final_vocab = _doc_freq(df)

    # Write outputs
    pkl_path = out_dir / "recipes_clean.pkl"
    pq_path = out_dir / "recipes_clean.parquet"
    vocab_path = out_dir / "ingredient_vocab.json"

    df.to_pickle(pkl_path)
    df.to_parquet(pq_path, compression="zstd")

    # Vocab sorted desc by frequency for readability
    vocab_sorted = dict(sorted(final_vocab.items(), key=lambda kv: -kv[1]))
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_sorted, f, ensure_ascii=False, indent=2)

    log.info("  wrote %s (%.1f MB)", pkl_path, pkl_path.stat().st_size / 1e6)
    log.info("  wrote %s (%.1f MB)", pq_path, pq_path.stat().st_size / 1e6)
    log.info("  wrote %s (%s ingredients)", vocab_path, f"{len(vocab_sorted):,}")

    return df, final_vocab

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description=""
    )
    p.add_argument("--source", default="./data/recipe_nlg/full_dataset.csv",
                   help="HuggingFace dataset name or local CSV path")
    p.add_argument("--max-recipes", type=int, default=50_000,
                   help="Subsample size (default 50000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="./data/processed")
    p.add_argument("--min-df", type=int, default=5,
                   help="Drop ingredients with document frequency < min_df")
    p.add_argument("--min-ingredients", type=int, default=3,
                   help="Drop recipes with fewer than this many ingredients")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    funnel: list[tuple[str, int]] = []

    t_start = time.perf_counter()

    # Load + subsample
    df = load_recipenlg(args.source, args.max_recipes, args.seed)
    funnel.append(("Loaded + subsampled", len(df)))

    # Phase 1
    df = phase_1_clean(df)
    funnel.append(("After Phase 1 (cleaning)", len(df)))

    # Phase 2
    df = phase_2_normalize(df)
    funnel.append(("After Phase 2 (NER normalize)", len(df)))

    # Phase 3
    df = phase_3_dedupe(df)
    funnel.append(("After Phase 3 (dedup)", len(df)))

    # Phase 4
    df = phase_4_filter_vocab(df, min_df=args.min_df,
                              min_ingredients=args.min_ingredients)
    funnel.append(("After Phase 4 (vocab filter)", len(df)))

    # Phase 5
    df_final, vocab = phase_5_finalize(df, out_dir)
    funnel.append(("Final", len(df_final)))

    # Copy schemas.py into out_dir for downstream convenience
    src_schemas = Path(__file__).parent / "schemas.py"
    if src_schemas.exists():
        shutil.copy(src_schemas, out_dir / "schemas.py")

    dt = time.perf_counter() - t_start
    log.info("DONE in %.1f min", dt / 60)
    log.info("Final: %s recipes, %s ingredients in vocab",
             f"{len(df_final):,}", f"{len(vocab):,}")
    log.info("Outputs in %s/", out_dir)


if __name__ == "__main__":
    main()