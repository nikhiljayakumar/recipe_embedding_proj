"""
Phase 0 smoke test. Run this BEFORE preprocess.py.

Purpose:
  1. Verify inflect is installed and working.
  2. Test the NER-based normalization pipeline (lowercase, singularize, synonyms).
  3. Eyeball outputs on RecipeNLG-style NER tokens to catch obvious failures.

Usage:
  python smoke_test.py

If this prints "SMOKE TEST PASSED", you're cleared to run preprocess.py.
"""

from __future__ import annotations


# NER tokens as RecipeNLG produces them — no quantities, no units.
TEST_NER_TOKENS = [
    "chicken breasts",
    "garlic cloves",
    "tomatoes",
    "scallions",
    "heavy whipping cream",
    "plain flour",
    "eggs",
    "unsalted butter",
    "zucchini",
    "soy sauce",
    "baking soda",
    "olive oil",
    "onion",
    "fresh basil leaves",
    "crushed tomatoes",
    "ground beef",
    "courgette",
    "aubergine",
    "garbanzo beans",
    "confectioners sugar",
    "bay leaves",
    "coriander leaves",
    "spring onion",
]

# Subset where we assert an exact expected output.
EXPECTED = {
    "chicken breasts":      "chicken breast",
    "garlic cloves":        "garlic clove",
    "tomatoes":             "tomato",
    "scallions":            "green onion",
    "heavy whipping cream": "heavy cream",
    "plain flour":          "all-purpose flour",
    "eggs":                 "egg",
    "courgette":            "zucchini",
    "aubergine":            "eggplant",
    "garbanzo beans":       "chickpea",
    "confectioners sugar":  "powdered sugar",
    "coriander leaves":     "cilantro",
    "spring onion":         "green onion",
}


def main() -> None:
    print("=" * 70)
    print("Phase 0 smoke test: NER normalization pipeline")
    print("=" * 70)

    # Step 1: inflect
    try:
        import inflect
        inflect.engine()  # verify it initializes without error
    except ImportError as e:
        print(f"FAIL: cannot import inflect ({e})")
        print("  Fix: pip install inflect")
        raise SystemExit(1)
    print("[1/3] inflect import ok")

    # Step 2: post_normalize
    try:
        from preprocess import post_normalize
    except ImportError as e:
        print(f"FAIL: cannot import post_normalize from preprocess ({e})")
        raise SystemExit(1)
    print("[2/3] post_normalize import ok")

    # Step 3: run normalization and check expected mappings
    print()
    print("[3/3] eyeball normalization on NER-style tokens:")
    print("-" * 70)
    failures: list[tuple[str, str, str]] = []
    for raw in TEST_NER_TOKENS:
        result = post_normalize(raw)
        expected = EXPECTED.get(raw)
        tag = ""
        if expected is not None and result != expected:
            tag = f"  <-- EXPECTED {expected!r}"
            failures.append((raw, result, expected))
        print(f"  {raw:<40}  ->  {result}{tag}")
    print("-" * 70)

    # Sanity-check: no token should normalize to a pure number or fragment
    normalized_set = {post_normalize(t) for t in TEST_NER_TOKENS}
    suspicious = [t for t in normalized_set if t and not any(c.isalpha() for c in t)]
    if suspicious:
        print(f"\nWARN: suspicious tokens after normalization: {suspicious}")
        print("  If you see fragments like '1' or 'cup', the NER column may have noise.")

    if failures:
        print(f"\nFAIL: {len(failures)} normalization mismatch(es):")
        for raw, got, exp in failures:
            print(f"  {raw!r}: got {got!r}, expected {exp!r}")
        raise SystemExit(1)

    print()
    print("=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Ready to run:")
    print("  python preprocess.py --source mbien/recipe_nlg")
    print("  python preprocess.py --source ./full_dataset.csv")


if __name__ == "__main__":
    main()
