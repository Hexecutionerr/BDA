# Data

The raw dataset is NOT committed (too large).

Download the **ratings-only** file for a category (we used Electronics) from the
Amazon Review Data page by Ni, Li & McAuley (UCSD):
https://nijianmo.github.io/amazon/index.html

Place the CSV in this folder, e.g. `data/Electronics.csv`.
Expected columns (no header row): item, user, rating, timestamp.
If the column order differs on the download page, edit the schema in `src/als_recommender.py`.
