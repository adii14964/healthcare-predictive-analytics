"""
Feature engineering: embeddings for clinical notes plus aggregation and join with structured features.

Usage:
    python -m src.feature_engineering --raw_dir data/raw --out_dir data/processed --model all-MiniLM-L6-v2
"""
import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

def compute_embeddings(notes_series, model_name="all-MiniLM-L6-v2", cache_path=None):
    """Try to compute sentence-transformer embeddings. If library missing, fall back to TF-IDF dense vectors (simple)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb = model.encode(notes_series.tolist(), show_progress_bar=True)
        return np.array(emb)
    except Exception as e:
        print("Warning: sentence-transformers not available or failed to load. Falling back to simple TF-IDF vectors. Error:", e)
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer(max_features=256, ngram_range=(1,2))
        X = tf.fit_transform(notes_series.fillna("").tolist())
        return X.toarray()

def aggregate_embeddings(encounter_ids, embeddings, notes_df):
    """Aggregate per-encounter embeddings (here one note per encounter). Keep simple: map by index."""
    # assume notes_df.index aligns with embeddings
    emb_df = pd.DataFrame(embeddings, index=notes_df['encounter_id'].values)
    emb_df.index.name = 'encounter_id'
    return emb_df.reset_index()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    notes_path = os.path.join(args.raw_dir, "clinical_notes.csv")
    patients_path = os.path.join(args.raw_dir, "patients.csv")
    encounters_path = os.path.join(args.raw_dir, "encounters.csv")
    if not (os.path.exists(notes_path) and os.path.exists(patients_path) and os.path.exists(encounters_path)):
        raise FileNotFoundError("Raw CSV files not found. Run data_prep first.")

    notes = pd.read_csv(notes_path)
    patients = pd.read_csv(patients_path)
    encounters = pd.read_csv(encounters_path)

    # compute embeddings
    texts = notes['note_text'].astype(str)
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    print("Computing embeddings with model:", model_name)
    embeddings = compute_embeddings(texts, model_name=model_name)

    emb_df = aggregate_embeddings(encounters['encounter_id'].values, embeddings, notes)

    # Minimal structured features: patient-level join
    patients_small = patients[['patient_id','age','gender','chronic_conditions_count','social_risk_score']]
    merged = encounters.merge(patients_small, on='patient_id', how='left')
    merged = merged.merge(emb_df, on='encounter_id', how='left')

    out_path = os.path.join(args.out_dir, "features.parquet")
    merged.to_parquet(out_path, index=False)
    print("Wrote features to", out_path)

if __name__ == "__main__":
    main()