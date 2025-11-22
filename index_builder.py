import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="", help="path to CSV with title,description (optional)")
parser.add_argument("--out_prefix", type=str, default="models/faiss", help="output prefix")
parser.add_argument("--max_rows", type=int, default=0, help="0 => all rows")
args = parser.parse_args()

# load data
if args.csv:
    df = pd.read_csv(args.csv).dropna(subset=["description"])
else:
    # fallback to toy data
    data = [
        {"id": 0, "title": "Wireless Bluetooth Headphones", "description": "Over-ear bluetooth headphones with noise cancelling and 30h battery."},
        {"id": 1, "title": "USB-C Fast Charger", "description": "30W USB-C wall charger for fast charging phones and tablets."},
        {"id": 2, "title": "Stainless Steel Water Bottle", "description": "Insulated 1L water bottle keeps drinks cold for 24 hours."},
        {"id": 3, "title": "Gaming Mouse", "description": "Ergonomic RGB gaming mouse with adjustable DPI and extra buttons."},
        {"id": 4, "title": "Organic Green Tea", "description": "Loose leaf green tea, cultivated without pesticides."}
    ]
    df = pd.DataFrame(data)

if args.max_rows and args.max_rows > 0:
    df = df.head(args.max_rows)

df = df.reset_index(drop=True)
texts = (df["title"].fillna("") + ". " + df["description"].fillna("")).tolist()

# model and embeddings
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Encoding {len(texts)} texts...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True).astype("float32")

# normalize for cosine similarity
faiss.normalize_L2(embeddings)

# build index
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)
print(f"Added {index.ntotal} vectors of dim {d}")

# ensure output dir exists
import os
out_dir = os.path.dirname(args.out_prefix)
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir)

# save index and metadata and embeddings
index_path = args.out_prefix + ".index"
meta_path = args.out_prefix + "_meta.pkl"
emb_path = args.out_prefix + "_emb.npy"

faiss.write_index(index, index_path)
df.to_pickle(meta_path)
np.save(emb_path, embeddings)

print("Saved index to", index_path)
print("Saved metadata to", meta_path)
print("Saved embeddings to", emb_path)
# gonna use this dataset https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset