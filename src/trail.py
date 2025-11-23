import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import time

# 1) tiny dataset
data = [
    {"id": 0, "title": "Wireless Bluetooth Headphones", "description": "Over-ear bluetooth headphones with noise cancelling and 30h battery."},
    {"id": 1, "title": "USB-C Fast Charger", "description": "30W USB-C wall charger for fast charging phones and tablets."},
    {"id": 2, "title": "Stainless Steel Water Bottle", "description": "Insulated 1L water bottle keeps drinks cold for 24 hours."},
    {"id": 3, "title": "Gaming Mouse", "description": "Ergonomic RGB gaming mouse with adjustable DPI and extra buttons."},
    {"id": 4, "title": "Organic Green Tea", "description": "Loose leaf green tea, cultivated without pesticides."}
]
df = pd.DataFrame(data)
texts = (df["title"] + ". " + df["description"]).tolist()

# 2) load model and encode
print("Loading model and encoding texts...")
model = SentenceTransformer("all-MiniLM-L6-v2")   # fast & small
embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

# 3) normalize for cosine similarity (inner product on normalized vectors)
faiss.normalize_L2(embeddings)

# 4) build FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)   # inner product (works with normalized vectors -> cosine)
index.add(embeddings)
print(f"Index built with {index.ntotal} vectors, dim={d}")

# 5) helper search function
def semantic_search(query, k=3):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        row = df.iloc[idx]
        results.append({"id": int(row["id"]), "title": row["title"], "desc": row["description"], "score": float(score)})
    return results

# 6) try some queries
queries = [
    "noise cancelling headphones",
    "fast phone charger",
    "healthy tea leaves",
    "mouse for gaming",
    "insulated bottle for travel"
]

for q in queries:
    print("\nQUERY:", q)
    for r in semantic_search(q, k=3):
        print(f"  - {r['title']} (score {r['score']:.3f})")
