
# Semantic Product Search Engine (SBERT + FAISS + BM25 + Streamlit)

A production-style semantic search engine built using:

- **Sentence-BERT embeddings**
- **FAISS vector search (Flat / HNSW / IVF)**
- **Hybrid BM25 + Semantic ranking**
- **Category-aware metadata filtering**
- **Interactive Streamlit UI**
- **Evaluation metrics (Precision@5, Recall@5, Latency)**
- **Embedding visualization (PCA / t-SNE)**


---

## Features

### Semantic Search Using SBERT
Uses the model:
```
all-MiniLM-L6-v2
```
Fast, lightweight, strong contextual understanding.

---

### Vector Search with FAISS  
Supports 3 index types:

| Index | Mode | Pros |
|-------|------|------|
| `flat` | exact search | perfect accuracy |
| `hnsw` | ANN graph | fast, scalable |
| `ivf` | inverted file | ideal for millions of items |

---

### Hybrid Ranking (BM25 + Semantic)
Real search engines combine lexical & semantic signals.

We compute a hybrid score:
```
hybrid_score = α * semantic + (1 - α) * bm25
```
User controls α in UI.

---

### Category Filtering  
Metadata-aware search improves relevance:
- Better precision  
- Domain-specific control  
- Mirrors Amazon/Rakuten-style filters  

---

### Analysis & Evaluation
Included in `evaluate.py`:

- Precision@5  
- Recall@5  
- Avg. Query Latency  
- Embedding visualization saved as  
  ```
  models/embeddings_viz.png
  ```

---

### Streamlit Web App  
Features:
- Live semantic search  
- Category filter  
- Hybrid score tuning slider  
- Millisecond latency display  
- Clean UI  

Run via:
```
streamlit run src/app.py
```

---

## Project Structure

```
semantic-search-engine/
├── README.md
├── requirements.txt
│
├── data/
│   ├── amazon_clean.csv
│   └── raw/amazon.csv
│
├── models/
│   ├── faiss.index
│   ├── faiss_meta.pkl
│   └── faiss_emb.npy
│
└── src/
    ├── clean_dataset.py
    ├── index_builder.py
    ├── app.py
    ├── trail.py
    └── evaluate.py

│
└── .github/workflows/
    └── build-index.yml
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Step 1 — Clean Dataset

```bash
python src/clean_dataset.py
```

Produces:

```
data/amazon_clean.csv
```

---

## Step 2 — Build FAISS Index

Example using HNSW and 10,000 rows:

```bash
python src/index_builder.py --csv data/amazon_clean.csv --out_prefix models/faiss --index_type hnsw --max_rows 10000
```

This outputs:

- `faiss.index`  
- `faiss_meta.pkl`  
- `faiss_emb.npy`

---

## Step 3 — Run Streamlit UI

```bash
streamlit run src/app.py
```

Open:
```
http://localhost:8501
```

---

## Step 4 — Evaluation

```bash
python src/evaluate.py --index models/faiss.index --meta models/faiss_meta.pkl --emb models/faiss_emb.npy
```

Outputs:
- Precision@5  
- Recall@5  
- Latency  
- Embedding visualization PNG  

---

## Architecture

```
Raw Dataset
    ↓ clean_dataset.py
Clean CSV (title, description, category)
    ↓ index_builder.py
SBERT Embeddings → FAISS Index → BM25 Corpus
    ↓ app.py
Streamlit UI (Semantic + Hybrid Search)
```

---

## Results & Observations
- Semantic search retrieves conceptually similar products even without keyword overlap  
- Hybrid BM25 + embeddings improves short or keyword-heavy queries  
- HNSW achieves latency under ~15ms on CPU  
- Categories enhance filtering and relevance  
- Embedding space forms visible clusters (Accessories, Electronics, etc.)

---

<!-- ## Future Enhancements
- Multimodal search with CLIP (text + images)  
- FastAPI backend  
- Deployment on HuggingFace Spaces  
- Cross-encoder re-ranking  
- Query auto-suggestions  
- Metadata-aware reranking (price, rating, etc.)  

--- -->

## Acknowledgements
- Kaggle Amazon Dataset  (https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
- Sentence Transformers (UKPLab)  
- Facebook FAISS  
- Streamlit  
- scikit-learn  

---

