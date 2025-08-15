from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import subprocess, os, joblib, pandas as pd, numpy as np
from typing import Optional
from datetime import datetime

app = FastAPI(title="Healthcare Predictive Analytics Platform", version="0.5.0")

DATA_RAW = "data/raw"
DATA_PROC = "data/processed"

class PredictRequest(BaseModel):
    encounter_id: str

def _maybe_add_risk_scores(df):
    """If model exists, compute readmission risk for all rows and add a 'risk_score' column (0-1).
    Caches might be desirable but this is a simple implementation for demo purposes.
    """
    model_path = os.path.join(DATA_PROC, "model.joblib")
    if not os.path.exists(model_path):
        # no model, leave risk_score as NaN
        df['risk_score'] = float('nan')
        return df
    try:
        clf = joblib.load(model_path)
        # prepare X by dropping non-feature cols
        X = df.drop(columns=['encounter_id','patient_id','admit_date','discharge_date','readmitted_30d'], errors='ignore')
        # predict_proba for all rows
        proba = clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else clf.predict(X)
        df['risk_score'] = proba
    except Exception as e:
        df['risk_score'] = float('nan')
    return df

@app.get("/health")
def health():
    return {"status":"ok","version":"0.5.0"}

@app.post("/generate-data")
def generate_data(n_patients: int = 2000):
    cmd = ["python","-m","src.data_prep","--n_patients", str(n_patients), "--out_dir", DATA_RAW]
    try:
        subprocess.run(cmd, check=True)
        return {"status":"ok","message":"Data generated"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/featurize")
def featurize():
    cmd = ["python","-m","src.feature_engineering","--raw_dir", DATA_RAW, "--out_dir", DATA_PROC]
    try:
        subprocess.run(cmd, check=True)
        return {"status":"ok","message":"Features generated"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train():
    cmd = ["python","-m","src.model_train","--processed", os.path.join(DATA_PROC,"features.parquet"), "--out_dir", DATA_PROC]
    try:
        subprocess.run(cmd, check=True)
        return {"status":"ok","message":"Model trained and saved"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/encounters")
def list_encounters(
    page: int = Query(1, ge=1),
    page_size: int = Query(15, ge=1, le=200),
    search: Optional[str] = None,
    sort_by: Optional[str] = Query(None, description="One of: date, risk_score, patient_id"),
    sort_order: Optional[str] = Query("desc", description="asc or desc"),
    date_from: Optional[str] = Query(None, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="YYYY-MM-DD"),
    risk_min: Optional[float] = Query(None, ge=0.0, le=1.0),
    risk_max: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    features_path = os.path.join(DATA_PROC, "features.parquet")
    if not os.path.exists(features_path):
        raise HTTPException(status_code=404, detail="Features not found. Run /featurize first.")
    df = pd.read_parquet(features_path)

    # parse dates for filtering
    if date_from:
        try:
            d0 = pd.to_datetime(date_from).date()
            df = df[pd.to_datetime(df['admit_date']).dt.date >= d0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"date_from parse error: {e}")
    if date_to:
        try:
            d1 = pd.to_datetime(date_to).date()
            df = df[pd.to_datetime(df['admit_date']).dt.date <= d1]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"date_to parse error: {e}")

    # search by encounter_id or patient_id (case-insensitive)
    if search:
        s = str(search).lower()
        df = df[df['encounter_id'].str.lower().str.contains(s) | df['patient_id'].str.lower().str.contains(s)]

    # Add risk scores if needed for filtering/sorting
    if (risk_min is not None) or (risk_max is not None) or (sort_by == 'risk_score'):
        df = _maybe_add_risk_scores(df)

    # apply risk filters
    if risk_min is not None:
        df = df[df['risk_score'] >= float(risk_min)]
    if risk_max is not None:
        df = df[df['risk_score'] <= float(risk_max)]

    # apply sorting (single-column)
    if sort_by:
        if sort_by == 'date':
            # sort by admit_date
            df['admit_date_parsed'] = pd.to_datetime(df['admit_date'], errors='coerce')
            df = df.sort_values(by='admit_date_parsed', ascending=(sort_order=='asc'))
        elif sort_by == 'patient_id':
            df = df.sort_values(by='patient_id', ascending=(sort_order=='asc'))
        elif sort_by == 'risk_score':
            if 'risk_score' not in df.columns:
                df = _maybe_add_risk_scores(df)
            df = df.sort_values(by='risk_score', ascending=(sort_order=='asc'))
        else:
            raise HTTPException(status_code=400, detail="Unsupported sort_by value. Use one of: date, risk_score, patient_id")

    total = len(df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    end = start + page_size
    df_page = df[['encounter_id','patient_id','admit_date','discharge_date','readmitted_30d']].iloc[start:end]
    # include risk_score if present
    if 'risk_score' in df.columns:
        df_page = df_page.copy()
        df_full = df.reset_index(drop=True)
        # map risk_score values for the page
        risk_map = df.set_index('encounter_id')['risk_score'].to_dict()
        df_page['risk_score'] = df_page['encounter_id'].map(risk_map)
    return {
        "page": page,
        "page_size": page_size,
        "total": int(total),
        "total_pages": total_pages,
        "items": df_page.to_dict(orient='records')
    }

@app.get("/encounters/{encounter_id}")
def encounter_details(encounter_id: str):
    features_path = os.path.join(DATA_PROC, "features.parquet")
    if not os.path.exists(features_path):
        raise HTTPException(status_code=404, detail="Features not found. Run /featurize first.")
    df = pd.read_parquet(features_path)
    row = df[df['encounter_id']==encounter_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Encounter not found")
    out = row.to_dict(orient='records')[0]
    clinical_note = None
    try:
        notes = pd.read_csv(os.path.join(DATA_RAW, "clinical_notes.csv"))
        note_row = notes[notes['encounter_id']==encounter_id]
        if not note_row.empty:
            clinical_note = note_row.iloc[0]['note_text']
    except Exception:
        clinical_note = None
    return {"encounter": out, "clinical_note": clinical_note}

@app.post("/predict")
def predict(req: PredictRequest):
    features_path = os.path.join(DATA_PROC, "features.parquet")
    model_path = os.path.join(DATA_PROC, "model.joblib")
    if not os.path.exists(features_path) or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Features or model not available. Run /featurize and /train first.")
    df = pd.read_parquet(features_path)
    row = df[df['encounter_id']==req.encounter_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Encounter not found")
    X = row.drop(columns=['encounter_id','patient_id','admit_date','discharge_date','readmitted_30d'], errors='ignore')
    clf = joblib.load(model_path)
    proba = clf.predict_proba(X)[:,1][0] if hasattr(clf, "predict_proba") else float(clf.predict(X)[0])
    return {"encounter_id": req.encounter_id, "readmission_risk": float(proba)}

@app.get("/patients/{encounter_id}/explain")
def explain(encounter_id: str):
    features_path = os.path.join(DATA_PROC, "features.parquet")
    model_path = os.path.join(DATA_PROC, "model.joblib")
    if not os.path.exists(features_path) or not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Features or model not available. Run /featurize and /train first.")
    df = pd.read_parquet(features_path)
    row = df[df['encounter_id']==encounter_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Encounter not found")
    X = row.drop(columns=['encounter_id','patient_id','admit_date','discharge_date','readmitted_30d'], errors='ignore')
    clf = joblib.load(model_path)
    try:
        import shap
        explainer = shap.Explainer(clf.named_steps['model'], masker=shap.maskers.Data(X))
        shap_vals = explainer(X)
        out = {
            "encounter_id": encounter_id,
            "shap_mean_abs": dict(zip(X.columns.tolist(), np.abs(shap_vals.values[0]).tolist())),
            "expected_value": float(shap_vals.base_values[0]) if hasattr(shap_vals, 'base_values') else None
        }
        return out
    except Exception as e:
        return {"warning": "SHAP not available on server", "error": str(e)}