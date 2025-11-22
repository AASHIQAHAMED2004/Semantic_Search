import pandas as pd

# Load dataset
df = pd.read_csv('amazon.csv')

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

# Pick the best available text columns
title_col = 'product_name'
desc_cols = ['about_product', 'description', 'review']

# Ensure missing columns are skipped
desc_cols = [c for c in desc_cols if c in df.columns]

# Create final "description" field
df['description'] = df[desc_cols].fillna('').agg('. '.join, axis=1)

# Keep only title + description
df_clean = df[[title_col, 'description']].copy()

# Remove empty text
df_clean['description'] = df_clean['description'].str.strip()
df_clean = df_clean[df_clean['description'].str.len() > 20]

# Remove duplicates
df_clean = df_clean.drop_duplicates(subset=[title_col, 'description'])

# Rename to final column names
df_clean = df_clean.rename(columns={
    title_col: 'title'
})

# Save
out_path = 'amazon_clean.csv'
df_clean.to_csv(out_path, index=False)

print("Created:", out_path)
print("Rows:", len(df_clean))
print(df_clean.head())
