import os
import pandas as pd

# Path to the raw CSV 
RAW_CSV = 'data/raw/amazon.csv'   
OUT_DIR = os.path.join('data')
OUT_CSV = os.path.join(OUT_DIR, 'amazon_clean.csv')

def main(raw_csv=RAW_CSV, out_csv=OUT_CSV):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print("Loading:", raw_csv)
    df = pd.read_csv(raw_csv)

    # normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # pick best title column
    title_candidates = ['product_name', 'product name', 'title', 'name']
    title_col = next((c for c in title_candidates if c in df.columns), None)
    if title_col is None:
        raise ValueError(f"No title column found. Candidates: {title_candidates}")

    # description candidates
    desc_candidates = ['about_product', 'description', 'product_description', 'review', 'product description']
    desc_cols = [c for c in desc_candidates if c in df.columns]
    if not desc_cols:
        raise ValueError(f"No description columns found. Candidates: {desc_candidates}")

    # optional category
    category_col = 'category' if 'category' in df.columns else None

    print("Title column:", title_col)
    print("Description columns:", desc_cols)
    print("Category column:", category_col if category_col else "None (filled with 'Unknown')")

    # build description by concatenating relevant fields
    df['description'] = df[desc_cols].fillna('').agg('. '.join, axis=1)
    df['description'] = df['description'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # keep minimal columns
    keep = [title_col, 'description']
    if category_col:
        keep.append(category_col)
    df_min = df[keep].copy()

    # rename title -> title
    df_min = df_min.rename(columns={title_col: 'title'})

    # handle category
    if category_col is None:
        df_min['category'] = 'Unknown'
    else:
        df_min = df_min.rename(columns={category_col: 'category'})
        df_min['category'] = df_min['category'].fillna('Unknown').astype(str)

    # drop short / empty descriptions
    df_min['description'] = df_min['description'].astype(str)
    df_min = df_min[df_min['description'].str.len() > 20]

    # dedupe
    df_min = df_min.drop_duplicates(subset=['title', 'description'])
    df_min = df_min.reset_index(drop=True)

    # save
    df_min.to_csv(out_csv, index=False)
    print(f"Saved cleaned dataset to: {out_csv}")
    print("Rows:", len(df_min))
    print(df_min.head(5).to_string())

if __name__ == "__main__":
    main()
