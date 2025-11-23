import argparse
import time
import numpy as np
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def load_all(index_path, meta_path, emb_path, model_name):
    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        meta_obj = pickle.load(f)
    df = meta_obj['df']
    emb = np.load(emb_path)
    model = SentenceTransformer(model_name)
    return index, df, emb, model

def precision_recall_at_k(df, index, model, k=5, samples=200):
    np.random.seed(42)
    indices = np.random.choice(len(df), size=min(samples, len(df)), replace=False)
    precisions = []
    recalls = []
    latencies = []

    # precompute category counts
    cat_counts = df['category'].value_counts().to_dict()

    for i in indices:
        query_title = df.iloc[i]['title']
        true_cat = df.iloc[i]['category']
        start = time.time()
        q_emb = model.encode([query_title], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, k)
        elapsed = (time.time() - start) * 1000.0
        latencies.append(elapsed)

        retrieved_idx = [int(x) for x in I[0] if int(x) >= 0]
        retrieved_cats = df.iloc[retrieved_idx]['category'].tolist()

        # precision@k = fraction of retrieved items with same category
        prec = sum(1 for c in retrieved_cats if c == true_cat) / max(1, len(retrieved_cats))
        precisions.append(prec)

        # recall@k = retrieved_same / (total items in same category excluding query)
        total_relevant = max(0, cat_counts.get(true_cat, 0) - 1)
        if total_relevant == 0:
            recall = 0.0
        else:
            retrieved_relevant = sum(1 for c in retrieved_cats if c == true_cat)
            recall = retrieved_relevant / total_relevant
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls), np.mean(latencies)

def plot_embeddings(emb, df, out_png='embeddings_viz.png', n_sample=2000):
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    n = min(n_sample, emb.shape[0])
    sample_idx = np.random.choice(len(df), size=n, replace=False)
    emb_s = emb[sample_idx]
    labels = df.iloc[sample_idx]['category'].astype(str).tolist()
    try:
        print("Running PCA -> TSNE embedding reduction (this may take a while)...")
        pca = PCA(n_components=50)
        emb_p = pca.fit_transform(emb_s)
        tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(emb_p)
    except Exception as e:
        print("t-SNE failed or too slow, using PCA->2 directly. Error:", e)
        pca2 = PCA(n_components=2)
        emb_2d = pca2.fit_transform(emb_s)

    # color by category (top 10 categories, rest as 'Other')
    top_cats = df['category'].value_counts().nlargest(10).index.tolist()
    colors = []
    mapped = []
    for lab in labels:
        mapped.append(lab if lab in top_cats else 'Other')
    uniq = sorted(list(set(mapped)))
    cmap = plt.get_cmap('tab20')
    color_map = {u: cmap(i % 20) for i, u in enumerate(uniq)}

    plt.figure(figsize=(10, 8))
    for u in uniq:
        sel = [i for i, m in enumerate(mapped) if m == u]
        if not sel:
            continue
        pts = emb_2d[sel]
        plt.scatter(pts[:,0], pts[:,1], s=8, label=u, alpha=0.7)
    plt.legend(markerscale=2, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Embedding visualization (sampled)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved embedding visualization to:", out_png)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='models/faiss.index')
    parser.add_argument('--meta', type=str, default='models/faiss_meta.pkl')
    parser.add_argument('--emb', type=str, default='models/faiss_emb.npy')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--viz_out', type=str, default='models/embeddings_viz.png')
    args = parser.parse_args()

    print("Loading index/meta/embeddings...")
    index, df, emb, model = load_all(args.index, args.meta, args.emb, args.model)
    print("Index size:", index.ntotal, "Data rows:", len(df), "Emb shape:", emb.shape)

    print("Evaluating (precision/recall/latency)...")
    p, r, lat = precision_recall_at_k(df, index, model, k=args.k, samples=args.samples)
    print(f"Precision@{args.k}: {p:.4f}")
    print(f"Recall@{args.k}: {r:.4f}")
    print(f"Avg latency (ms): {lat:.2f}")

    print("Creating embedding visualization...")
    plot_embeddings(emb, df, out_png=args.viz_out, n_sample=2000)

if __name__ == "__main__":
    main()
