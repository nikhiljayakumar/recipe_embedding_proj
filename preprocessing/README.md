# Agent 1: Data & Preprocessing

Foundation pipeline for the recipe semantic embedding project.
Output is the canonical dataset that Agents 2, 3, 4 consume.

## Files in this folder

| File | What it is |
|---|---|
| `schemas.py` | Contract definition. **Downstream agents import from this.** |
| `requirements.txt` | Python deps |
| `smoke_test.py` | Phase 0 — verifies the normalization pipeline on sample NER tokens. **Run first.** |
| `preprocess.py` | Main pipeline. Phases 1–5, end-to-end. |
| `README.md` | This file |

## Setup (one-time)

```bash
cd preprocessing
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

### 1. Smoke test (<10 sec)

```bash
python smoke_test.py
```

This:
- Verifies `inflect` imports and initializes.
- Runs `post_normalize` on ~25 NER-style tokens and asserts expected mappings
  (singularization, UK→US naming, synonym collapses).
- Flags any token that normalizes to a non-alpha string.

**If smoke test fails on a normalization assertion**, you've changed the
`SYNONYMS` dict or `UNCOUNTABLE` set in `preprocess.py` and the test fixture
in `smoke_test.py` no longer matches. Update one or the other.

### 2. Main pipeline

Default (50k from HF):

```bash
python preprocess.py --source mbien/recipe_nlg
```

To run a smaller sample:

```bash
python preprocess.py --source mbien/recipe_nlg --max-recipes 30000
```

If `mbien/recipe_nlg` 404s on HF (license-gated mirrors come and go), download
the RecipeNLG `full_dataset.csv` from <https://recipenlg.cs.put.poznan.pl/dataset>
and point at it locally:

```bash
python preprocess.py --source ./full_dataset.csv
```

Useful flags:
- `--min-df 5` — minimum document frequency for an ingredient to survive
- `--min-ingredients 3` — minimum recipe size after vocab filter
- `--out-dir ./data` — where outputs land (default `./data/`)

## Output

Everything lands in `./data/`:

```
data/
├── recipes_clean.pkl          # canonical artifact (Agents 2/3 load this)
├── recipes_clean.parquet      # inspection / portability copy
├── ingredient_vocab.json      # ingredient -> doc frequency (sorted desc)
├── preprocessing_report.md    # funnel, top-50, 20 before/after samples, decisions
└── schemas.py                 # copy for downstream agents
```

**Pickle is canonical.** Parquet is a convenience for `duckdb` / `pandas` /
no-pickle environments. If you load the parquet and the `list[str]` columns
behave oddly, fall back to pickle.

## What downstream agents do

```python
import pandas as pd
from data.schemas import (
    NORMALIZED_INGREDIENTS, RECIPE_ID, TITLE, INSTRUCTIONS, validate
)

df = pd.read_pickle("data/recipes_clean.pkl")
validate(df)          # raises AssertionError if contract is violated

# Agent 2 — Word2Vec sentences
sentences = df[NORMALIZED_INGREDIENTS].tolist()  # list[list[str]]

# Agent 2 — SBERT input
texts = (df[TITLE] + " " +
         df[NORMALIZED_INGREDIENTS].apply(" ".join) + " " +
         df[INSTRUCTIONS].str.slice(0, 500)).tolist()

# Agent 3/4 — recipe lookup by id
recipe = df.loc[df[RECIPE_ID] == 12345].iloc[0]
```

## Compute notes (handoff to Agents 2/3)

This stage is light CPU work — finishes in a couple minutes on any modern
laptop now that we use the pre-computed NER column. **Do not spin up a GPU.**

- **Agent 2** should use the H100 for SBERT encoding. Word2Vec on 50k recipes
  is CPU work and finishes in minutes.
- **Agent 3** does not need a GPU. FAISS at 50k is CPU. UMAP at 50k runs in
  ~3–5 min on CPU.

## Sanity checks before declaring done

- `recipes_clean.pkl` loads, `validate(df)` passes
- Row count is roughly 35k–50k (some loss to dedup/filtering is expected)
- `ingredient_vocab.json` has ~2k–10k unique ingredients
- Top 20 in vocab are pantry items (salt, sugar, flour, butter, water, oil…)
- Eyeball 50 random rows in `preprocessing_report.md` and confirm normalized
  ingredients look reasonable

## Decisions worth knowing

These are documented in `preprocessing_report.md` too — surfacing here for
quick reference:

1. **NER column is the source of truth.** `normalized_ingredients` is built
   from RecipeNLG's pre-computed `ner` column (quantities/units already
   removed) — no heavy ingredient-parser step. We just lowercase, singularize
   the last word with `inflect`, and apply a small synonym dict.
2. **Multi-word ingredients stay as single tokens with spaces.** `"olive oil"`
   is ONE element of `normalized_ingredients`, not two. Don't split on whitespace.
3. **Cilantro vs coriander:** `"coriander leaves"` → `"cilantro"`, but bare
   `"coriander"` is left alone (US recipes use bare "coriander" for the spice).
4. **Synonym map is small (~25 entries).** Word2Vec recovers most synonymy
   from co-occurrence. We only collapse the egregious cases (UK/US naming,
   `confectioners sugar` / `icing sugar` / `powdered sugar`, etc.).
5. **Dedup key includes a coarse instruction-length bucket** so two recipes
   sharing an ingredient set but differing wildly in instruction length stay
   distinct.

## When something goes wrong

| Symptom | Likely fix |
|---|---|
| `ImportError: inflect` | `pip install inflect` |
| Smoke test asserts a mapping that doesn't match | You changed `SYNONYMS` / `UNCOUNTABLE` in `preprocess.py`; update the fixture in `smoke_test.py` to match |
| `load_dataset` raises auth/license error | Download CSV manually, pass `--source ./full_dataset.csv` |
| `ValueError: Source columns ... missing required {'ner'}` | Source isn't RecipeNLG (or is an old dump without the NER column) — get a fresh copy |
| `validate(df)` fails on output | Open an issue — this is a pipeline bug, not a downstream problem |
| Vocab size > 20k | NER column has noisy fragments leaking through; inspect top/bottom of `ingredient_vocab.json` |
| Vocab size < 1k | Over-filtered. Lower `--min-df`. |
