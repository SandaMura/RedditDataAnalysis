import pandas as pd
import os

DATA_PATH = "data/reddit_links.tsv"

# Only download if file doesn't exist
if not os.path.exists(DATA_PATH):
    print("Downloading data...")
    url = "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv"
    df = pd.read_csv(url, sep="\t", comment="#")
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, sep="\t", index=False)
else:
    print("Loading data from local file...")
    df = pd.read_csv(DATA_PATH, sep="\t")
