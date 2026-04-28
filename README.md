# Recipe Embedding Explorer

If you were to search for a recipe on Google, it would simply try to keyword match and return to websites or other recipes with the same words as the one you searched up. However, recipes are often more similar than we think, and across cuisines similar recipes might not have the same words but are effectively the same thing. 

This project allows you to search for recipes and get similar recipes back, and search for ingredients and get similar ingredients back. 

## Step 0: Environment Setup
Create a virtual environment and get the requirements. I used `python=3.11.5` in anaconda.
```bash
# Create virtual environment
python -m venv .venv
# Linux
source .venv/bin/activate
# Windows
.venv\Scripts\Activate
# Install requirements
pip install -r requirements.txt
```

## Step 1: Preprocessing the `RecipeNLG` Dataset
This step does a lot of filtering to the dataset and also restructures into a cleaner dataset. The schema for the processed dataset is found in `preprocessing/schemas.py`. \
IMPORTANT: For testing purposes `preprocess.py` subsamples the entire 2 million entry dataset to just 50k samples. We can change it back or just keep it how it is (but training longer).

First you need to download the RecipeNLG dataset from [here](https://recipenlg.cs.put.poznan.pl/) and put `full_dataset.csv` inside `data/recipe_nlg/`
Then to run it:
```bash
python preprocessing/preprocess.py
```

## Step 2: Generating the recipe embeddings and vector embeddings
```bash
python embeddings/train_word2vec.py
python embeddings/encode_sbert.py
python embeddings/pool_ingredients.py
```

## Step 3. Build UMAP and Plot Graphs
```bash
python search/build_indices.py
python search/run_umap.py
```
## Step 4. Streamlit Demo App
```bash
streamlit run app.py
```


