import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import time
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="Semantic Product Search", layout="wide")

# Default paths (change if different)
DEFAULT_INDEX = 'models/faiss.index'
DEFAULT_META = 'models/faiss_meta.pkl'
DEFAULT_EMB = 'models/faiss_emb.npy'
DEFAULT_MODEL = 'all-MiniLM-L6-v2'

@st.cache_resource(show_spinner=False)
def load_resources(index_path=DEFAULT_INDEX, meta_path=DEFAULT_META, model_name=DEFAULT_MODEL):
    model = SentenceTransformer(model_name)
    # load faiss index
    index = faiss.read_index(index_path)
    # load meta
    with open(meta_path, 'rb') as f:
        meta_obj = pickle.load(f)
    df = meta_obj['df']
    tokenized = meta_obj['tokenized']
    bm25 = BM25Okapi(tokenized)
    return model, index, df, bm25

st.sidebar.title("Index settings")
index_path = st.sidebar.text_input("FAISS index path", value=DEFAULT_INDEX)
meta_path = st.sidebar.text_input("Metadata path", value=DEFAULT_META)
model_name = st.sidebar.text_input("SBERT model", value=DEFAULT_MODEL)
k = st.sidebar.slider("Top K", 1, 20, 5)
normalize = st.sidebar.checkbox("Normalize query (cosine)", value=True)
hybrid_alpha = st.sidebar.slider("Semantic weight (alpha)", 0.0, 1.0, 0.6)  # alpha * semantic + (1-alpha) * bm25

model, index, meta_df, bm25 = load_resources(index_path=index_path, meta_path=meta_path, model_name=model_name)

st.title("Semantic Product Search — Demo")

# Category filter
all_cats = sorted(meta_df['category'].fillna('Unknown').unique().tolist())
selected_category = st.sidebar.selectbox("Filter by category", ["All"] + all_cats)

query = st.text_input("Enter your search query", "")
if st.button("Search") and query.strip():
    q = query.strip()
    start = time.time()
    # semantic
    q_emb = model.encode([q], convert_to_numpy=True).astype('float32')
    if normalize:
        faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k*5)  # retrieve more to allow filtering + hybrid rerank
    sem_scores = D[0]  # inner-product scores (cosine if normalized)
    sem_idxs = I[0]

    # BM25 scores on full corpus
    bm25_scores = np.array(bm25.get_scores(q.split()), dtype=float)  # length = corpus size

    # Build candidate list (dedupe)
    candidates = []
    seen = set()
    for score, idx in zip(sem_scores, sem_idxs):
        if idx < 0 or idx in seen:
            continue
        # category filter at candidate stage
        row = meta_df.iloc[idx]
        if selected_category != "All" and row['category'] != selected_category:
            continue
        seen.add(idx)
        candidates.append((idx, float(score)))

    # If BM25 brings other candidates (helpful when semantic misses)
    if len(candidates) < k:
        # bring top BM25 (filtered by category)
        bm25_order = np.argsort(bm25_scores)[::-1]
        for idx in bm25_order:
            if idx in seen: continue
            row = meta_df.iloc[idx]
            if selected_category != "All" and row['category'] != selected_category:
                continue
            seen.add(int(idx))
            candidates.append((int(idx), 0.0))
            if len(candidates) >= k*2:
                break

    # Compute combined scores for candidates
    combined = []
    for idx, sem_score in candidates:
        bm = bm25_scores[idx]
        # normalize bm25 to 0..1 by dividing by max (safe guard)
        bm_norm = bm / (bm25_scores.max() + 1e-12)
        combined_score = hybrid_alpha * sem_score + (1.0 - hybrid_alpha) * bm_norm
        combined.append((idx, combined_score, sem_score, bm_norm))

    # rank by combined score
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)[:k]
    elapsed = (time.time() - start) * 1000.0

    st.write(f"Search time: {elapsed:.1f} ms — index size: {index.ntotal} — candidates considered: {len(candidates)}")
    for idx, comb, sem, bm in combined_sorted:
        row = meta_df.iloc[idx]
        st.markdown(f"**{row.get('title','(no title)')}** — score: `{comb:.4f}`  (sem: {sem:.4f}, bm25: {bm:.4f})")
        desc = row.get('description', '')
        st.write(desc[:400] + ("..." if len(desc) > 400 else ""))
        # optional: show product_link or image if present
        if 'product_link' in row.index:
            st.markdown(f"[View Product]({row['product_link']})")
        st.markdown("---")
