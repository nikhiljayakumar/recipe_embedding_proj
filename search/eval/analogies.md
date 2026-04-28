# Ingredient analogies

Vector arithmetic on the Word2Vec ingredient space, computed via
`gensim`'s `most_similar`. Returns the top-5 ingredients nearest
to `sum(positive) - sum(negative)`.

Each block shows the human-readable expression, then the result
(or a `SKIPPED` line if any term is out-of-vocab — Agent 2 noted
that cuisine words like 'italian' often don't appear in the
ingredient vocab because they live in titles, not ingredient lists).

---

## `parmesan - italian + japanese`

| rank | ingredient | score |
|---:|:---|---:|
| 1 | `tofu` | 0.661 |
| 2 | `doubanjiang` | 0.646 |
| 3 | `chicken soup stock granule` | 0.623 |
| 4 | `katakuriko` | 0.612 |
| 5 | `gochujang` | 0.599 |

> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_

## `butter - baking + frying`

**SKIPPED** — out of W2V vocab: `frying`

## `beef - american + indian`

**SKIPPED** — out of W2V vocab: `indian`

## `tortilla - mexican + indian`

**SKIPPED** — out of W2V vocab: `indian, mexican`

## `wine - french + japanese`

**SKIPPED** — out of W2V vocab: `french`

## `pasta - italian + japanese`

| rank | ingredient | score |
|---:|:---|---:|
| 1 | `doubanjiang` | 0.663 |
| 2 | `chicken soup stock granule` | 0.641 |
| 3 | `dashi` | 0.623 |
| 4 | `shoyu` | 0.617 |
| 5 | `wasabi` | 0.601 |

> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_

## `basil - italian + thai`

**SKIPPED** — out of W2V vocab: `thai`

## `chocolate + cinnamon`

| rank | ingredient | score |
|---:|:---|---:|
| 1 | `mashed banana` | 0.728 |
| 2 | `colored sprinkle` | 0.723 |
| 3 | `chocolate-covered` | 0.722 |
| 4 | `very ripe banana` | 0.720 |
| 5 | `regular sugar` | 0.714 |

> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_

## `lime + cilantro - lemon`

| rank | ingredient | score |
|---:|:---|---:|
| 1 | `fresh cilantro` | 0.613 |
| 2 | `cilantro leaf` | 0.600 |
| 3 | `habanero` | 0.522 |
| 4 | `plum sauce` | 0.518 |
| 5 | `queso fresco` | 0.518 |

> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_

## `soy sauce + lime`

**SKIPPED** — out of W2V vocab: `soy_sauce`

## `heavy cream + lemon`

**SKIPPED** — out of W2V vocab: `heavy_cream`

## `coffee + chocolate`

| rank | ingredient | score |
|---:|:---|---:|
| 1 | `dark chocolate` | 0.758 |
| 2 | `cocoa powder` | 0.750 |
| 3 | `espresso powder` | 0.749 |
| 4 | `chocolate curl` | 0.745 |
| 5 | `bittersweet chocolate` | 0.742 |

> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_

## `honey + lemon`

| rank | ingredient | score |
|---:|:---|---:|
| 1 | `tangerine` | 0.585 |
| 2 | `pink grapefruit` | 0.582 |
| 3 | `freshly squeezed orange juice` | 0.581 |
| 4 | `sweet apple` | 0.579 |
| 5 | `raspberry vinegar` | 0.575 |

> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_

## `bacon + egg`

| rank | ingredient | score |
|---:|:---|---:|
| 1 | `fresh white bread crumb` | 0.589 |
| 2 | `hot mashed potato` | 0.587 |
| 3 | `crisply cooked bacon` | 0.584 |
| 4 | `cream style corn` | 0.567 |
| 5 | `white sandwich bread` | 0.566 |

> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_

## `yogurt - milk`

| rank | ingredient | score |
|---:|:---|---:|
| 1 | `greek yogurt` | 0.430 |
| 2 | `tahini paste` | 0.385 |
| 3 | `bulgar wheat` | 0.380 |
| 4 | `low-fat yogurt` | 0.373 |
| 5 | `low-fat plain yogurt` | 0.371 |

> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_

## `mozzarella + cheddar - parmesan`

| rank | ingredient | score |
|---:|:---|---:|
| 1 | `cheddar cheese` | 0.609 |
| 2 | `shredded cheese` | 0.605 |
| 3 | `grated cheese` | 0.554 |
| 4 | `shredded sharp cheddar cheese` | 0.531 |
| 5 | `colby cheese` | 0.513 |

> _Notes: [add interpretation — successful substitution? noise? co-occurrence?]_

## `fish sauce - soy sauce`

**SKIPPED** — out of W2V vocab: `fish_sauce, soy_sauce`
