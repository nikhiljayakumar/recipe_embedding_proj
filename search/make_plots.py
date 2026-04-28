"""make_plots.py — Generate the PNG plots for the blog post.

Produces, in search/visualization/:
  1. recipes_umap_unlabeled.png      Full scatter, light gray, no labels.
  2. recipes_umap_by_cuisine.png     Same scatter, colored by heuristic cuisine.
  3. ingredients_umap.png            Top-100 most common ingredients labeled.
  4. ingredient_neighborhoods.png    4-panel zoom on interesting clusters.

CUISINE LABELING IS A HEURISTIC: keyword matching against recipe TITLES (per
Agent 2's note that cuisine words live in titles, not in ingredient lists).
The plot caption flags this so blog readers don't think it's ground truth.

Optional: plot_analogy(positive, negative) renders one analogy with arrows
in the ingredient UMAP space. NOT called from main() because Agent 2's
sanity check showed cuisine words like "italian"/"japanese" are likely
out-of-vocab in the W2V space, killing the spec's headline analogies. Run
this manually after run_evals.py if a real analogy emerges:
    from make_plots import plot_analogy
    plot_analogy(positive=["yogurt"], negative=["milk"])

Usage: python make_plots.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = PROJECT_ROOT / "embeddings"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "recipes_clean.pkl"
VIS_DIR = PROJECT_ROOT / "search" / "visualization"
VIS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


# ---------------------------------------------------------------------------
# Cuisine heuristic (title-based per Agent 2's recommendation)
# ---------------------------------------------------------------------------
# Insertion-order preserved; ties on score broken in favor of earlier entries.

CUISINE_TITLE_KEYWORDS: dict[str, list[str]] = {
    "italian": [
        "italian", "pasta", "spaghetti", "lasagna", "pizza", "risotto",
        "carbonara", "bolognese", "marinara", "pesto", "ravioli",
        "fettuccine", "gnocchi", "tiramisu", "calzone",
    ],
    "mexican": [
        "mexican", "tex-mex", "taco", "burrito", "enchilada", "fajita",
        "quesadilla", "salsa", "guacamole", "tostada", "tamale",
        "chimichanga",
    ],
    "indian": [
        "indian", "curry", "tikka", "masala", "biryani", "naan", "samosa",
        "tandoori", "korma", "vindaloo", "paneer", "raita", "dal",
    ],
    "chinese": [
        "chinese", "lo mein", "chow mein", "kung pao", "general tso",
        "stir fry", "stir-fry", "fried rice", "egg roll", "wonton",
        "dumpling", "szechuan", "sichuan", "moo shu",
    ],
    "japanese": [
        "japanese", "sushi", "ramen", "teriyaki", "tempura", "miso",
        "udon", "yakitori", "sashimi", "donburi", "katsu",
    ],
    "thai": [
        "thai", "pad thai", "tom yum", "tom kha", "satay", "massaman",
        "panang", "larb",
    ],
    "french": [
        "french", "ratatouille", "coq au vin", "bouillabaisse", "crepe",
        "quiche", "beef bourguignon", "souffle", "tarte",
    ],
    "mediterranean": [
        "mediterranean", "greek", "lebanese", "turkish", "moroccan",
        "hummus", "falafel", "tabbouleh", "tzatziki", "souvlaki",
        "gyro", "kebab", "shawarma", "baklava", "pita",
    ],
    "american": [
        "american", "bbq", "barbecue", "burger", "hamburger",
        "cheeseburger", "meatloaf", "mac and cheese", "macaroni and cheese",
        "buffalo", "cornbread", "biscuits and gravy", "sloppy joe",
    ],
}


def label_cuisine(title: str) -> Optional[str]:
    """Score `title` against each cuisine's keywords, return the best.

    - Single-word keywords use word-boundary matching ("thai" doesn't match
      "thailand"... well actually thailand has 'thai' as a prefix. \\b handles
      it: \\bthai\\b matches "Pad Thai" but not "thailandic").
    - Multi-word / hyphenated keywords use substring matching.

    Returns the cuisine name if the best score is >= 1, else None.
    """
    title_lower = title.lower()
    best_cuisine, best_score = None, 0
    for cuisine, keywords in CUISINE_TITLE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if " " in kw or "-" in kw:
                if kw in title_lower:
                    score += 1
            else:
                if re.search(rf"\b{re.escape(kw)}\b", title_lower):
                    score += 1
        if score > best_score:
            best_cuisine, best_score = cuisine, score
    return best_cuisine


# ---------------------------------------------------------------------------
# Plot 1: recipes UMAP unlabeled
# ---------------------------------------------------------------------------

def plot_recipes_unlabeled(coords: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.3, color="#555555")
    ax.set_title(
        f"Recipe embedding space — UMAP of SBERT vectors (n={len(coords):,})",
        fontsize=14,
    )
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 2: recipes UMAP colored by cuisine
# ---------------------------------------------------------------------------

def plot_recipes_by_cuisine(
    coords: np.ndarray,
    titles: list[str],
    out_path: Path,
) -> None:
    cuisines = np.array([label_cuisine(t) for t in titles])

    fig, ax = plt.subplots(figsize=(14, 10))

    # Background: unlabeled recipes
    other_mask = cuisines == None  # noqa: E711 (need exact identity check on object array)
    n_other = int(other_mask.sum())
    ax.scatter(
        coords[other_mask, 0], coords[other_mask, 1],
        s=1.5, alpha=0.12, color="#cccccc",
        label=f"unlabeled (n={n_other:,})",
    )

    # Foreground: each labeled cuisine
    palette = sns.color_palette("husl", n_colors=len(CUISINE_TITLE_KEYWORDS))
    for color, cuisine in zip(palette, CUISINE_TITLE_KEYWORDS.keys()):
        mask = cuisines == cuisine
        n = int(mask.sum())
        if n == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=5, alpha=0.65, color=color,
            label=f"{cuisine} (n={n:,})",
        )

    ax.set_title(
        "Recipe embedding space colored by heuristic cuisine label\n"
        "(keyword match against recipe title — not ground truth)",
        fontsize=14,
    )
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=9, framealpha=0.92, markerscale=2.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 3: ingredients UMAP with top-N labeled
# ---------------------------------------------------------------------------

def plot_ingredients_umap(
    coords: np.ndarray,
    names: list[str],
    df: pd.DataFrame,
    out_path: Path,
    top_n: int = 100,
) -> None:
    # Most common ingredients = appear in the most recipes
    counter: Counter[str] = Counter()
    for ings in df["normalized_ingredients"]:
        counter.update(ings)
    top_set = set(name for name, _ in counter.most_common(top_n))
    top_idxs = [i for i, n in enumerate(names) if n in top_set]

    fig, ax = plt.subplots(figsize=(16, 12))

    # Background: all ingredients
    ax.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.22, color="#888888")

    # Highlight top-N
    ax.scatter(
        coords[top_idxs, 0], coords[top_idxs, 1],
        s=14, alpha=0.85, color="#cc3300", zorder=3,
    )

    # Labels (de-underscore for display)
    texts = []
    for i in top_idxs:
        label = names[i].replace("_", " ")
        texts.append(
            ax.text(coords[i, 0], coords[i, 1], label,
                    fontsize=8, color="black", zorder=4)
        )

    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5),
            expand_points=(1.2, 1.4),
        )
    except ImportError:
        print("  (adjustText not installed; labels may overlap — pip install adjustText)")

    ax.set_title(
        f"Ingredient embedding space — UMAP of Word2Vec vectors (n={len(coords):,})\n"
        f"top {top_n} most common ingredients labeled",
        fontsize=14,
    )
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 4: 4-panel ingredient neighborhoods
# ---------------------------------------------------------------------------

# Each panel anchored on a "core" ingredient. We try a few candidate names
# in case Agent 1's tokenization differs from what we expect.
NEIGHBORHOOD_PANELS: list[tuple[str, list[str]]] = [
    ("Mediterranean herbs",  ["basil", "oregano", "thyme", "parsley"]),
    ("Asian sauces",         ["soy_sauce", "soy sauce", "fish_sauce", "sesame_oil"]),
    ("Baking staples",       ["all_purpose_flour", "flour", "baking_powder", "vanilla"]),
    ("Mexican / SW spices",  ["cumin", "chili_powder", "cilantro", "paprika"]),
]


def plot_neighborhoods(
    coords: np.ndarray,
    names: list[str],
    name_to_idx: dict[str, int],
    out_path: Path,
    points_per_panel: int = 30,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    palette = sns.color_palette("Set2", 4)

    for ax, (title, candidates), color in zip(axes.flat, NEIGHBORHOOD_PANELS, palette):
        # First candidate that resolves to a real ingredient
        anchor = None
        for cand in candidates:
            for variant in (cand, cand.replace(" ", "_"), cand.replace("-", "_")):
                if variant in name_to_idx:
                    anchor = variant
                    break
            if anchor:
                break
        if anchor is None:
            ax.text(0.5, 0.5, f"No anchor in vocab\n(tried: {candidates})",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        anchor_idx = name_to_idx[anchor]
        anchor_pt = coords[anchor_idx]

        # K nearest in 2D space (good enough for a panel; UMAP keeps
        # similar things close)
        d2 = np.sum((coords - anchor_pt) ** 2, axis=1)
        near_idxs = np.argsort(d2)[:points_per_panel]

        # Background of full vocab
        ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.08, color="lightgray")
        # Neighborhood
        ax.scatter(
            coords[near_idxs, 0], coords[near_idxs, 1],
            s=22, alpha=0.85, color=color,
        )
        # Anchor distinct
        ax.scatter(
            [anchor_pt[0]], [anchor_pt[1]],
            s=80, color=color, edgecolor="black", linewidths=1.2, zorder=5,
        )

        # Label every neighborhood point
        for i in near_idxs:
            ax.annotate(
                names[i].replace("_", " "),
                (coords[i, 0], coords[i, 1]),
                fontsize=7, alpha=0.95,
                xytext=(3, 3), textcoords="offset points",
            )

        # Zoom with padding
        nx, ny = coords[near_idxs, 0], coords[near_idxs, 1]
        pad_x = max((nx.max() - nx.min()) * 0.18, 0.5)
        pad_y = max((ny.max() - ny.min()) * 0.18, 0.5)
        ax.set_xlim(nx.min() - pad_x, nx.max() + pad_x)
        ax.set_ylim(ny.min() - pad_y, ny.max() + pad_y)
        ax.set_title(f"{title}  (anchor: {anchor.replace('_', ' ')})", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Ingredient neighborhoods in UMAP space", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Optional Plot 5: analogy visualization (call manually if a good one exists)
# ---------------------------------------------------------------------------

def plot_analogy(
    positive: list[str],
    negative: list[str],
    out_path: Optional[Path] = None,
    top_neighbors: int = 8,
) -> None:
    """Plot an ingredient analogy in 2D UMAP space with arrows.

    All terms must be in the W2V vocab AND in the ingredient UMAP space
    (same thing in this project — both use the ingredient vocab).
    Arrows go from each negative term to each positive term to suggest
    the direction of analogy. Top-k nearest neighbors of the resolved
    point are highlighted.

    Call this manually after run_evals.py once you've found a working
    analogy. Example:
        plot_analogy(positive=["yogurt"], negative=["milk"])
    """
    # Lazy import to avoid loading SBERT etc. if main() is the only target
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from search import (  # type: ignore
        ingredient_analogy, _ingredient_name_to_id, _resolve_ingredient,
    )

    coords = np.load(VIS_DIR / "umap_ingredients.npy")
    with open(EMB_DIR / "ingredient_id_map_reverse.json", encoding="utf-8") as f:
        id_to_name = {int(k): v for k, v in json.load(f).items()}

    # Resolve and look up positions
    def pos(name: str) -> Optional[np.ndarray]:
        canon = _resolve_ingredient(name)
        if canon is None:
            return None
        return coords[_ingredient_name_to_id[canon]]

    pos_pts = [(t, pos(t)) for t in positive]
    neg_pts = [(t, pos(t)) for t in negative]
    if any(p is None for _, p in pos_pts + neg_pts):
        missing = [t for t, p in pos_pts + neg_pts if p is None]
        print(f"  skipped analogy plot — missing terms: {missing}")
        return

    # Find top neighbors of the analogy result for highlighting
    result = ingredient_analogy(positive, negative, k=top_neighbors)
    result_pts = [(r["name"], pos(r["name"])) for r in result]
    result_pts = [(n, p) for n, p in result_pts if p is not None]

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.1, color="lightgray")

    for name, p in pos_pts:
        ax.scatter(p[0], p[1], s=120, color="#1f77b4", edgecolor="black", linewidths=1)
        ax.annotate(f"+{name.replace('_', ' ')}", (p[0], p[1]),
                    xytext=(6, 6), textcoords="offset points", fontsize=11, fontweight="bold")
    for name, p in neg_pts:
        ax.scatter(p[0], p[1], s=120, color="#d62728", edgecolor="black", linewidths=1)
        ax.annotate(f"-{name.replace('_', ' ')}", (p[0], p[1]),
                    xytext=(6, 6), textcoords="offset points", fontsize=11, fontweight="bold")
    for name, p in result_pts:
        ax.scatter(p[0], p[1], s=70, color="#2ca02c", alpha=0.85, zorder=4)
        ax.annotate(name.replace("_", " "), (p[0], p[1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)

    # Arrows from each negative term to each positive term
    for _, np_ in neg_pts:
        for _, pp in pos_pts:
            ax.annotate(
                "",
                xy=(pp[0], pp[1]), xytext=(np_[0], np_[1]),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, alpha=0.6),
            )

    title_str = " + ".join(positive)
    if negative:
        title_str = f"{title_str} - {' - '.join(negative)}"
    ax.set_title(f"Ingredient analogy: {title_str}", fontsize=14)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_aspect("equal")
    plt.tight_layout()

    if out_path is None:
        safe = re.sub(r"[^a-z0-9]+", "_", title_str.lower()).strip("_")
        out_path = VIS_DIR / f"analogy_{safe}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading UMAP coordinates and metadata...")
    recipe_coords = np.load(VIS_DIR / "umap_recipes.npy")
    ing_coords = np.load(VIS_DIR / "umap_ingredients.npy")
    df = pd.read_pickle(DATA_PATH)
    titles = df["title"].tolist()

    with open(EMB_DIR / "ingredient_id_map.json", encoding="utf-8") as f:
        name_to_idx: dict[str, int] = json.load(f)
    with open(EMB_DIR / "ingredient_id_map_reverse.json", encoding="utf-8") as f:
        id_to_name = {int(k): v for k, v in json.load(f).items()}
    names = [id_to_name[i] for i in range(ing_coords.shape[0])]

    print("\nPlot 1: recipes UMAP unlabeled")
    plot_recipes_unlabeled(recipe_coords, VIS_DIR / "recipes_umap_unlabeled.png")

    print("\nPlot 2: recipes UMAP by heuristic cuisine")
    plot_recipes_by_cuisine(recipe_coords, titles, VIS_DIR / "recipes_umap_by_cuisine.png")

    print("\nPlot 3: ingredients UMAP with top-100 labeled")
    plot_ingredients_umap(ing_coords, names, df, VIS_DIR / "ingredients_umap.png")

    print("\nPlot 4: 4-panel ingredient neighborhoods")
    plot_neighborhoods(ing_coords, names, name_to_idx, VIS_DIR / "ingredient_neighborhoods.png")

    print("\nDone. See search/visualization/")
    print("(For the optional analogy plot: from make_plots import plot_analogy)")


if __name__ == "__main__":
    main()