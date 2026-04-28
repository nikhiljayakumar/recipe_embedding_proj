"""app.py — Recipe Embedding Explorer (Streamlit demo)

Run from the project root:
    streamlit run app.py

Expects the following layout:
    RECIPE_EMBEDDING_PROJ/
        app.py                          <- this file
        data/processed/
            recipes_clean.pkl
            ingredient_vocab.json
        search/
            search.py
            recipe_text_index.faiss
            recipe_ingredient_index.faiss
            ingredient_index.faiss
            visualization/
                umap_recipes.csv
                umap_ingredients.csv
        embeddings/
            recipe_vectors.npy
            recipe_vectors_from_ingredients.npy
            ingredient_vectors.npy
            ingredient_id_map.json
            ingredient_id_map_reverse.json
            recipe_id_map.json
            word2vec.model
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allow `import search` from the search/ subdirectory
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "search"))

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Recipe Embedding Explorer",
    page_icon="🍴",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Resource loaders (cached so models load once, not on every interaction)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading search indices and models…")
def load_search():
    """Import search.py and trigger its module-level resource loading."""
    import search as _search
    return _search


@st.cache_data(show_spinner="Loading recipe data…")
def load_recipes() -> pd.DataFrame:
    return pd.read_pickle(_ROOT / "data" / "processed" / "recipes_clean.pkl")


@st.cache_data(show_spinner="Loading UMAP coordinates…")
def load_umap_recipes() -> pd.DataFrame:
    return pd.read_csv(_ROOT / "search" / "visualization" / "umap_recipes.csv")


@st.cache_data(show_spinner="Loading ingredient UMAP coordinates…")
def load_umap_ingredients() -> pd.DataFrame:
    return pd.read_csv(_ROOT / "search" / "visualization" / "umap_ingredients.csv")


@st.cache_data(show_spinner="Loading ingredient vocabulary…")
def load_ingredient_vocab() -> list[str]:
    path = _ROOT / "data" / "processed" / "ingredient_vocab.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Handle both list ["basil", ...] and dict {"basil": 0, ...} formats
    if isinstance(data, dict):
        return sorted(data.keys())
    return sorted(data)


# ---------------------------------------------------------------------------
# Helper: build a recipe detail card
# ---------------------------------------------------------------------------

def _recipe_card(row: pd.Series, score: float, rank: int) -> None:
    """Render a single recipe result as a styled markdown card."""
    raw = row.get("raw_ingredients", [])
    # raw_ingredients is stored as a list; guard against stringified lists
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = [raw]

    preview = raw[:5]
    more = len(raw) - 5

    score_pct = f"{score:.2f}"

    with st.container():
        st.markdown(
            f"**{rank}. {row['title']}** &nbsp;&nbsp; "
            f"<span style='color:#888;font-size:0.85em;'>similarity: {score_pct}</span>",
            unsafe_allow_html=True,
        )
        bullet_items = "\n".join(f"- {ing}" for ing in preview)
        if more > 0:
            bullet_items += f"\n- *…and {more} more*"
        st.markdown(bullet_items)

        instructions = row.get("instructions", "")
        if instructions:
            with st.expander("Full details"):
                st.markdown("**All ingredients:**")
                for ing in raw:
                    st.markdown(f"- {ing}")
                st.markdown("**Instructions:**")
                st.markdown(instructions)

        st.divider()


# ---------------------------------------------------------------------------
# Page 1: Recipe Search
# ---------------------------------------------------------------------------

def page_recipe_search(search, df: pd.DataFrame) -> None:
    st.header("🔍 Recipe Search")
    st.caption(
        "Type anything — a cuisine, a mood, a dish — and we'll find the "
        "most semantically similar recipes in the dataset."
    )

    query = st.text_input(
        "Search for recipes",
        placeholder="e.g. spicy Thai noodles, cozy winter soup, quick weeknight pasta…",
    )
    col_k, col_btn = st.columns([1, 5])
    with col_k:
        k = st.selectbox("Results", [5, 10, 20], index=1)
    with col_btn:
        st.write("")  # vertical alignment spacer
        search_clicked = st.button("Search", type="primary")

    if not query and not search_clicked:
        st.info("Enter a query above and press **Search** to get started.")
        return

    if not query.strip():
        st.warning("Please enter a search query.")
        return

    with st.spinner("Searching…"):
        results = search.find_similar_recipes_by_text(query.strip(), k=k)

    if not results:
        st.error(
            "No results found. Try a different query — shorter phrases often "
            "work better than long sentences."
        )
        return

    st.success(f"Top {len(results)} results for **'{query}'**")

    # Build a recipe_id -> DataFrame row lookup
    id_to_row = df.set_index("recipe_id")

    for rank, hit in enumerate(results, start=1):
        rid = hit["id"]
        if rid not in id_to_row.index:
            continue
        _recipe_card(id_to_row.loc[rid], hit["score"], rank)


# ---------------------------------------------------------------------------
# Page 2: Ingredient Explorer
# ---------------------------------------------------------------------------

def page_ingredient_explorer(search, df: pd.DataFrame, vocab: list[str]) -> None:
    st.header("🧄 Ingredient Explorer")
    st.caption(
        "Find ingredients that behave similarly in recipes, or discover recipes "
        "that match a combination of ingredients you have on hand."
    )

    tab_similar, tab_combo = st.tabs(
        ["Similar Ingredients", "Find Recipes by Ingredients"]
    )

    # ---- Sub-tab A: similar ingredients ----
    with tab_similar:
        st.subheader("Ingredients similar to…")
        selected_ing = st.selectbox(
            "Choose an ingredient",
            options=vocab,
            index=None,
            placeholder="Type to search the vocabulary…",
        )
        k_ing = st.selectbox("Results", [5, 10, 20], index=1, key="k_ing")

        if selected_ing:
            with st.spinner("Finding similar ingredients…"):
                results = search.find_similar_ingredients(selected_ing, k=k_ing)

            if not results:
                st.warning(
                    f"No neighbours found for **{selected_ing}**. "
                    "This ingredient may have too few recipe appearances to "
                    "have learned a reliable embedding."
                )
            else:
                st.success(f"Top ingredients similar to **{selected_ing}**")
                rows = [
                    {"Ingredient": r["name"], "Similarity": f"{r['score']:.3f}"}
                    for r in results
                ]
                st.table(pd.DataFrame(rows))
        else:
            st.info("Select an ingredient above to see its nearest neighbours.")

    # ---- Sub-tab B: recipe finder by ingredients ----
    with tab_combo:
        st.subheader("What can I make with…")
        selected_combo = st.multiselect(
            "Choose one or more ingredients",
            options=vocab,
            placeholder="Type to search…",
        )
        k_combo = st.selectbox("Results", [5, 10, 20], index=1, key="k_combo")
        find_clicked = st.button("Find Recipes", type="primary")

        if find_clicked:
            if not selected_combo:
                st.warning("Please select at least one ingredient.")
            else:
                with st.spinner("Searching for matching recipes…"):
                    results = search.find_recipes_by_ingredients(
                        selected_combo, k=k_combo
                    )

                unknown = [
                    ing for ing in selected_combo
                    if search.is_known_ingredient(ing) is None
                ]
                if unknown:
                    st.warning(
                        f"These ingredients weren't in our vocabulary and were "
                        f"skipped: **{', '.join(unknown)}**"
                    )

                if not results:
                    st.error(
                        "No recipes found. None of the selected ingredients "
                        "had usable embeddings — try different ingredients."
                    )
                else:
                    id_to_row = df.set_index("recipe_id")
                    st.success(
                        f"Top {len(results)} recipes for: "
                        f"**{', '.join(selected_combo)}**"
                    )
                    for rank, hit in enumerate(results, start=1):
                        rid = hit["id"]
                        if rid not in id_to_row.index:
                            continue
                        _recipe_card(id_to_row.loc[rid], hit["score"], rank)
        else:
            st.info("Select ingredients above and press **Find Recipes**.")


# ---------------------------------------------------------------------------
# Page 3: Embedding Map
# ---------------------------------------------------------------------------

def page_embedding_map(
    umap_recipes: pd.DataFrame,
    umap_ingredients: pd.DataFrame,
) -> None:
    st.header("🗺️ Embedding Map")
    st.caption(
        "A 2-D projection of the recipe and ingredient embedding spaces. "
        "Recipes that appear nearby share similar ingredients and cooking styles. "
        "Hover over any point to see its name."
    )

    map_tab_r, map_tab_i = st.tabs(["Recipes", "Ingredients"])

    # ---- Recipe map ----
    with map_tab_r:
        n_total = len(umap_recipes)
        DISPLAY_LIMIT = 10_000
        display_df = (
            umap_recipes.sample(DISPLAY_LIMIT, random_state=42)
            if n_total > DISPLAY_LIMIT
            else umap_recipes
        )
        if n_total > DISPLAY_LIMIT:
            st.info(
                f"Showing a random sample of {DISPLAY_LIMIT:,} of the "
                f"{n_total:,} total recipes for browser performance."
            )

        fig = px.scatter(
            display_df,
            x="x",
            y="y",
            hover_data={"title": True, "x": False, "y": False},
            opacity=0.45,
            height=680,
            color_discrete_sequence=["#4C9BE8"],
            labels={"x": "", "y": ""},
            title="Recipe Embedding Space (UMAP projection)",
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            hoverlabel=dict(bgcolor="white", font_size=13),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Ingredient map ----
    with map_tab_i:
        n_total_i = len(umap_ingredients)
        DISPLAY_LIMIT_I = 5_000
        display_ing = (
            umap_ingredients.sample(DISPLAY_LIMIT_I, random_state=42)
            if n_total_i > DISPLAY_LIMIT_I
            else umap_ingredients
        )
        if n_total_i > DISPLAY_LIMIT_I:
            st.info(
                f"Showing a random sample of {DISPLAY_LIMIT_I:,} of the "
                f"{n_total_i:,} total ingredients."
            )

        fig2 = px.scatter(
            display_ing,
            x="x",
            y="y",
            hover_data={"name": True, "x": False, "y": False},
            opacity=0.5,
            height=680,
            color_discrete_sequence=["#E87B4C"],
            labels={"x": "", "y": ""},
            title="Ingredient Embedding Space (UMAP projection)",
        )
        fig2.update_traces(marker=dict(size=3))
        fig2.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            hoverlabel=dict(bgcolor="white", font_size=13),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🍴 Recipe Embedding Explorer")
    st.markdown(
        "Semantic search and exploration over a dataset of 50,000+ recipes — "
        "powered by SBERT recipe embeddings and Word2Vec ingredient embeddings."
    )
    st.divider()

    # Load shared data up front (cached after first run)
    with st.spinner("Loading models and data (first run only)…"):
        search = load_search()
        df = load_recipes()
        vocab = load_ingredient_vocab()

    tab_search, tab_ing, tab_map = st.tabs(
        ["🔍 Recipe Search", "🧄 Ingredient Explorer", "🗺️ Embedding Map"]
    )

    with tab_search:
        page_recipe_search(search, df)

    with tab_ing:
        page_ingredient_explorer(search, df, vocab)

    with tab_map:
        umap_recipes = load_umap_recipes()
        umap_ingredients = load_umap_ingredients()
        page_embedding_map(umap_recipes, umap_ingredients)


if __name__ == "__main__" or True:
    main()