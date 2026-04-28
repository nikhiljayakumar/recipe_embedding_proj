# Substitution check

For each probe ingredient, the top-5 nearest neighbors in the
Word2Vec space, with a placeholder annotation column to fill in
after eyeballing.

**Annotations to use:**
- `SUB`  — plausible substitute (would work in a recipe in place of the probe)
- `CO`   — co-occurrence, not substitute (often appears together, not interchangeable)
- `NOISE` — neither — corpus artifact, low-frequency ingredient, etc.

---

## `buttermilk`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `sour milk` | 0.594 | [review] |
| 2 | `well-shaken buttermilk` | 0.539 | [review] |
| 3 | `sweet milk` | 0.518 | [review] |
| 4 | `kellogg` | 0.509 | [review] |
| 5 | `hot coffee` | 0.508 | [review] |

## `shallot`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `broccolini` | 0.637 | [review] |
| 2 | `baby new potato` | 0.632 | [review] |
| 3 | `taleggio cheese` | 0.626 | [review] |
| 4 | `button` | 0.624 | [review] |
| 5 | `green peppercorn` | 0.608 | [review] |

## `tahini`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `tahini paste` | 0.678 | [review] |
| 2 | `chickpea` | 0.644 | [review] |
| 3 | `pumpkin seed` | 0.639 | [review] |
| 4 | `bulgar wheat` | 0.627 | [review] |
| 5 | `salad ingredient` | 0.596 | [review] |

## `fish_sauce`

**Not in vocab.**

## `sour_cream`

**Not in vocab.**

## `heavy_cream`

**Not in vocab.**

## `dijon_mustard`

**Not in vocab.**

## `sherry`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `sherry wine` | 0.632 | [review] |
| 2 | `chicken liver` | 0.628 | [review] |
| 3 | `oyster mushroom` | 0.608 | [review] |
| 4 | `hot cooked rice` | 0.583 | [review] |
| 5 | `chabli` | 0.581 | [review] |

## `miso`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `dashi` | 0.926 | [review] |
| 2 | `doubanjiang` | 0.921 | [review] |
| 3 | `weipa` | 0.884 | [review] |
| 4 | `chicken soup stock granule` | 0.883 | [review] |
| 5 | `lotus root` | 0.864 | [review] |

## `gochujang`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `doubanjiang` | 0.860 | [review] |
| 2 | `ponzu sauce` | 0.860 | [review] |
| 3 | `stalks scallion` | 0.847 | [review] |
| 4 | `chili paste with garlic` | 0.836 | [review] |
| 5 | `slurry` | 0.831 | [review] |

## `mascarpone`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `whole blanched almond` | 0.731 | [review] |
| 2 | `base` | 0.719 | [review] |
| 3 | `bosc pear` | 0.718 | [review] |
| 4 | `sanding sugar` | 0.704 | [review] |
| 5 | `vanilla pod` | 0.693 | [review] |

## `ricotta`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `lasagna sheet` | 0.759 | [review] |
| 2 | `pecorino romano` | 0.707 | [review] |
| 3 | `milk ricotta cheese` | 0.645 | [review] |
| 4 | `parmigiano` | 0.639 | [review] |
| 5 | `mixed italian herb` | 0.628 | [review] |

## `sumac`

**Not in vocab.**

## `harissa`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `puy lentil` | 0.747 | [review] |
| 2 | `chile flake` | 0.742 | [review] |
| 3 | `frozen artichoke` | 0.714 | [review] |
| 4 | `tahini paste` | 0.712 | [review] |
| 5 | `tomato purée` | 0.711 | [review] |

## `gruyere`

| rank | ingredient | score | annotation |
|---:|:---|---:|:---|
| 1 | `gruyere cheese` | 0.802 | [review] |
| 2 | `freshly grated parmesan` | 0.733 | [review] |
| 3 | `chanterelle mushroom` | 0.733 | [review] |
| 4 | `emmenthaler cheese` | 0.728 | [review] |
| 5 | `taleggio cheese` | 0.723 | [review] |
