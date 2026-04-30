"""run_evals.py
outputs for blog post and video stuff
  - analogies.md            ~17 ingredient analogies (mix of hits and misses)
  - nearest_neighbors.md    Top-10 NN for 20 ingredients and 10 recipes
  - substitution_check.md   15 ingredients with annotated NN
"""

from __future__ import annotations

from pathlib import Path

from search import (
    find_similar_ingredients,
    find_similar_recipes_by_recipe_id,
    ingredient_analogy,
    is_known_ingredient,
)
from search import _df, _titles  # for picking probe recipes / displaying titles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "search" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


INGREDIENT_PROBES = [
    "basil", "oregano", "soy_sauce", "fish_sauce", "butter", "olive_oil",
    "lime", "lemon", "jalapeno", "cumin", "parmesan", "ginger", "vanilla",
    "tahini", "buttermilk", "shallot", "miso", "gochujang", "harissa",
    "sumac",
]

RECIPE_PROBES = [0, 100, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 40000]

ANALOGIES_TO_TRY: list[tuple[list[str], list[str], str]] = [
    # Cuisine-shift analogies. Per Agent 2's notebook, cuisine words
    # (italian, japanese, etc.) are usually NOT in the W2V vocab — they
    # show up in titles, not ingredient lists. Most of these will skip.
    (["parmesan", "japanese"],  ["italian"],  "parmesan - italian + japanese"),
    (["butter",   "frying"],    ["baking"],   "butter - baking + frying"),
    (["beef",     "indian"],    ["american"], "beef - american + indian"),
    (["tortilla", "indian"],    ["mexican"],  "tortilla - mexican + indian"),
    (["wine",     "japanese"],  ["french"],   "wine - french + japanese"),
    (["pasta",    "japanese"],  ["italian"],  "pasta - italian + japanese"),
    (["basil",    "thai"],      ["italian"],  "basil - italian + thai"),

    # Ingredient-only (more likely to land — all real ingredients).
    (["chocolate", "cinnamon"], [],           "chocolate + cinnamon"),
    (["lime",      "cilantro"], ["lemon"],    "lime + cilantro - lemon"),
    (["soy_sauce", "lime"],     [],           "soy sauce + lime"),
    (["heavy_cream", "lemon"],  [],           "heavy cream + lemon"),
    (["coffee",    "chocolate"],[],           "coffee + chocolate"),
    (["honey",     "lemon"],    [],           "honey + lemon"),
    (["bacon",     "egg"],      [],           "bacon + egg"),

    # Substitution-style (find replacement when removing one term).
    (["yogurt"],          ["milk"],          "yogurt - milk"),
    (["mozzarella", "cheddar"], ["parmesan"], "mozzarella + cheddar - parmesan"),
    (["fish_sauce"],      ["soy_sauce"],      "fish sauce - soy sauce"),
]

# Subset of ingredients to do a curated substitution check on.
SUBSTITUTION_CHECK = [
    "buttermilk", "shallot", "tahini", "fish_sauce", "sour_cream",
    "heavy_cream", "dijon_mustard", "sherry", "miso", "gochujang",
    "mascarpone", "ricotta", "sumac", "harissa", "gruyere",
]


# ---------------------------------------------------------------------------
# Markdown writers
def write_analogies() -> None:
    out = EVAL_DIR / "analogies.md"
    lines: list[str] = [
        "# Ingredient analogies",
        "",
        "Vector arithmetic on the Word2Vec ingredient space, computed via",
        "`gensim`'s `most_similar`. Returns the top-5 ingredients nearest",
        "to `sum(positive) - sum(negative)`.",
        "",
        "Each block shows the human-readable expression, then the result",
        "(or a `SKIPPED` line if any term is out-of-vocab — Agent 2 noted",
        "that cuisine words like 'italian' often don't appear in the",
        "ingredient vocab because they live in titles, not ingredient lists).",
        "",
        "---",
        "",
    ]

    for positive, negative, desc in ANALOGIES_TO_TRY:
        # Pre-check which terms resolve, for a clean SKIPPED message
        missing: list[str] = []
        for t in positive + negative:
            if is_known_ingredient(t) is None:
                missing.append(t)

        lines.append(f"## `{desc}`")
        if missing:
            lines.append("")
            lines.append(f"**SKIPPED** — out of W2V vocab: `{', '.join(missing)}`")
            lines.append("")
            continue

        results = ingredient_analogy(positive, negative, k=5)
        if not results:
            lines.append("")
            lines.append("**No results** (analogy returned empty — possibly all results match input terms).")
            lines.append("")
            continue

        lines.append("")
        lines.append("| rank | ingredient | score |")
        lines.append("|---:|:---|---:|")
        for rank, r in enumerate(results, 1):
            lines.append(f"| {rank} | `{r['name']}` | {r['score']:.3f} |")
        lines.append("")
        lines.append("> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out}")


def write_nearest_neighbors() -> None:
    out = EVAL_DIR / "nearest_neighbors.md"
    lines: list[str] = [
        "# Nearest neighbors",
        "",
        "## Ingredients (Word2Vec, top-10)",
        "",
    ]

    for ing in INGREDIENT_PROBES:
        canonical = is_known_ingredient(ing)
        if canonical is None:
            lines.append(f"### `{ing}`")
            lines.append("")
            lines.append("**Not in vocab.**")
            lines.append("")
            continue

        results = find_similar_ingredients(canonical, k=10)
        lines.append(f"### `{canonical}`")
        lines.append("")
        if not results:
            lines.append("(no neighbors returned)")
            lines.append("")
            continue
        lines.append("| rank | ingredient | score |")
        lines.append("|---:|:---|---:|")
        for rank, r in enumerate(results, 1):
            lines.append(f"| {rank} | `{r['name']}` | {r['score']:.3f} |")
        lines.append("")

    lines.append("## Recipes (SBERT, top-10, dedup'd)")
    lines.append("")

    for rid in RECIPE_PROBES:
        try:
            title = _titles[rid]
        except IndexError:
            continue
        results = find_similar_recipes_by_recipe_id(rid, k=10, mode="text", dedup=True)

        lines.append(f"### recipe_id={rid}: {title}")
        lines.append("")
        if not results:
            lines.append("(no neighbors returned)")
            lines.append("")
            continue
        lines.append("| rank | recipe_id | title | score |")
        lines.append("|---:|---:|:---|---:|")
        for rank, r in enumerate(results, 1):
            # escape pipes in titles
            safe_title = r["title"].replace("|", "\\|")
            lines.append(f"| {rank} | {r['id']} | {safe_title} | {r['score']:.3f} |")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out}")


def write_substitution_check() -> None:
    out = EVAL_DIR / "substitution_check.md"
    lines: list[str] = [
        "# Substitution check",
        "",
        "For each probe ingredient, the top-5 nearest neighbors in the",
        "Word2Vec space, with a placeholder annotation column to fill in",
        "after eyeballing.",
        "",
        "**Annotations to use:**",
        "- `SUB`  — plausible substitute (would work in a recipe in place of the probe)",
        "- `CO`   — co-occurrence, not substitute (often appears together, not interchangeable)",
        "- `NOISE` — neither — corpus artifact, low-frequency ingredient, etc.",
        "",
        "---",
        "",
    ]

    for ing in SUBSTITUTION_CHECK:
        canonical = is_known_ingredient(ing)
        lines.append(f"## `{ing}`")
        lines.append("")
        if canonical is None:
            lines.append("**Not in vocab.**")
            lines.append("")
            continue

        results = find_similar_ingredients(canonical, k=5)
        if not results:
            lines.append("(no neighbors returned)")
            lines.append("")
            continue
        lines.append("| rank | ingredient | score | annotation |")
        lines.append("|---:|:---|---:|:---|")
        for rank, r in enumerate(results, 1):
            lines.append(f"| {rank} | `{r['name']}` | {r['score']:.3f} | [review] |")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out}")


def main() -> None:
    print("Generating eval reports...\n")
    print("[1/3] analogies.md")
    write_analogies()
    print("\n[2/3] nearest_neighbors.md")
    write_nearest_neighbors()
    print("\n[3/3] substitution_check.md")
    write_substitution_check()
    print("\nDone. Curate the [review] markers in substitution_check.md")
    print("and the > Notes lines in analogies.md before sharing with Agent 4.")


if __name__ == "__main__":
    main()