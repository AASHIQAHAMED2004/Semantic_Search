# app.py
import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

@st.cache_resource
def load_resources(index_path="models/faiss.index", meta_path="models/faiss_meta.pkl"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(index_path)
    meta = pd.read_pickle(meta_path)
    return model, index, meta

st.set_page_config(page_title="Semantic Search Demo", layout="centered")
st.title("Semantic Product Search — Demo")

with st.sidebar:
    st.markdown("### Index settings")
    index_path = st.text_input("FAISS index path", value="models/faiss.index")
    meta_path = st.text_input("Metadata path", value="models/faiss_meta.pkl")
    k = st.slider("Top K", 1, 20, 5)
    normalize = st.checkbox("Normalize query (cosine)", value=True)

model, index, meta = load_resources(index_path=index_path, meta_path=meta_path)

query = st.text_input("Enter your search query")
if st.button("Search") and query:
    start = time.time()
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    if normalize:
        faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    elapsed = time.time() - start

    st.write(f"Search time: {elapsed*1000:.1f} ms — found {index.ntotal} indexed items")
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        row = meta.iloc[idx]
        st.markdown(f"**{row.get('title', 'title missing')}** — score: `{score:.4f}`")
        desc = row.get('description', '')
        st.write(desc[:400] + ("..." if len(desc) > 400 else ""))
        st.markdown("---")
