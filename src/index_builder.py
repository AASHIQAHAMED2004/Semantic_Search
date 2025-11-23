import os
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import pickle
from rank_bm25 import BM25Okapi

def build_index(embeddings, index_type='hnsw', hnsw_m=32, ivf_nlist=256):
    d = embeddings.shape[1]
    if index_type == 'hnsw':
        print("Building HNSW index...")
        index = faiss.IndexHNSWFlat(d, hnsw_m)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16
        index.add(embeddings)
    elif index_type == 'ivf':
        print("Building IVF index...")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        # train then add
        index.train(embeddings)
        index.add(embeddings)
    else:
        print("Building flat exact index (IndexFlatIP)...")
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
    return index

def normalize_embeddings(emb):
    faiss.normalize_L2(emb)
    return emb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='clean CSV path (title,description,category)')
    parser.add_argument('--out_prefix', type=str, default='models/faiss', help='output prefix for index/meta/emb')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='sentence-transformers model')
    parser.add_argument('--max_rows', type=int, default=0, help='max rows to index (0 => all)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--index_type', type=str, default='hnsw', choices=['flat','hnsw','ivf'])
    parser.add_argument('--hnsw_m', type=int, default=32)
    parser.add_argument('--ivf_nlist', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    print("Loading CSV:", args.csv)
    df = pd.read_csv(args.csv)
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)
    df = df.reset_index(drop=True)

    texts = (df['title'].fillna('') + '. ' + df['description'].fillna('')).tolist()

    print(f"Encoding {len(texts)} texts with model {args.model_name} ...")
    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=args.batch_size, convert_to_numpy=True).astype('float32')

    # normalize for cosine (inner product on normalized vectors)
    embeddings = normalize_embeddings(embeddings)

    index = build_index(embeddings, index_type=args.index_type, hnsw_m=args.hnsw_m, ivf_nlist=args.ivf_nlist)

    index_path = args.out_prefix + '.index'
    meta_path = args.out_prefix + '_meta.pkl'
    emb_path = args.out_prefix + '_emb.npy'

    print("Saving index to", index_path)
    faiss.write_index(index, index_path)

    print("Saving metadata to", meta_path)
    # We also save the DataFrame plus tokenized descriptions for BM25
    tokenized = [txt.split() for txt in df['description'].astype(str).tolist()]
    meta = {
        'df': df,                     # pandas DataFrame
        'tokenized': tokenized
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print("Saving embeddings to", emb_path)
    np.save(emb_path, embeddings)

    print("Done. Saved files:")
    print("  -", index_path)
    print("  -", meta_path)
    print("  -", emb_path)

if __name__ == "__main__":
    main()
# gonna use this dataset https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset