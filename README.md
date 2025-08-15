# Healthcare Predictive Analytics Platform (In Progress)

**Status:** In Progress • **Timeline:** Jan 2026 – Present  
**Tech:** Python, Scikit-learn, Hugging Face Transformers, FastAPI, AWS, Pandas

A GenAI-augmented predictive analytics platform that forecasts **patient 30‑day readmission risk** from EHR‑like **structured data** and **unstructured clinical notes**.  
It demonstrates an **end‑to‑end pipeline** — data generation, preprocessing, LLM‑assisted feature engineering, ML modeling, explainability, and a FastAPI dashboard.

> Targeting **15–20% improvement** in predictive accuracy over baseline statistical models via LLM‑assisted features from clinical notes.

---

## ✨ Key Features
- **Synthetic EHR generator**: produces de‑identified structured (demographics, diagnoses, labs, utilization) + unstructured clinical notes.
- **GenAI-assisted feature engineering**: uses Hugging Face text embeddings to extract latent risk signals from notes (medication adherence, social factors, care gaps).
- **Predictive modeling**: gradient boosting / XGBoost‑style classifiers with robust validation.
- **Explainability**: SHAP values for global and per‑patient insights.
- **FastAPI dashboard**: patient risk profiles, top drivers, and preventive recommendations.
- **AWS-ready**: containerized with Docker and deploy guide for ECS/Fargate or EC2.

---

## 🧱 Repository Structure
```
Healthcare-Predictive-Analytics/
├── data/
│   ├── raw/          # synthetic source data (generated locally; not committed)
│   ├── processed/    # train/test features (not committed)
│   └── external/     # public reference lookups, code lists, etc.
├── notebooks/        # EDA and experiments
├── src/
│   ├── data_prep.py
│   ├── feature_engineering.py
│   ├── model_train.py
│   ├── risk_scoring.py
│   └── app.py        # FastAPI app entrypoint
├── deployment/
│   ├── Dockerfile
│   └── aws_instructions.md
├── .github/workflows/
│   └── ci.yml        # lint & unit tests (placeholder)
├── requirements.txt
├── .gitignore
├── LICENSE (MIT)
└── README.md
```

> **Note:** `data/raw` and `data/processed` are git‑ignored to avoid committing sensitive or bulky files.

---

## 🚀 Quickstart (Local)

1) **Clone & set up**  
```bash
git clone https://github.com/adii14964/Healthcare-Predictive-Analytics.git
cd Healthcare-Predictive-Analytics
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Generate synthetic EHR data** *(coming in next commit)*  
```bash
python -m src.data_prep --generate
```
This will write CSV/Parquet files into `data/raw/` (patients, encounters, labs, diagnoses, and clinical_notes).

3) **Feature engineering + train model** *(coming in next commit)*  
```bash
python -m src.feature_engineering --embed-notes
python -m src.model_train --train
```

4) **Launch FastAPI dashboard** *(coming in next commit)*  
```bash
uvicorn src.app:app --reload --port 8000
```
Open http://localhost:8000 to view interactive docs and the dashboard.

---

## 🧠 GenAI-Assisted Features
- Sentence-level embeddings of clinical notes using **Hugging Face** (e.g., `sentence-transformers/all-MiniLM-L6-v2` as a performant baseline).
- Aggregations over time windows (last discharge, last 30/90 days).
- Derived semantic indicators (frailty cues, social risk, care plan adherence) that complement structured features.

---

## 📈 Modeling & Evaluation
- Train/valid/test split by patient to avoid leakage between visits.
- Baseline: logistic regression / simple rules.
- Target: **15–20% AUC/PR improvement** with text-derived features.
- Explainability: **SHAP** for global and per‑patient drivers.

---

## 📊 FastAPI Dashboard (Overview)
- **/docs**: interactive OpenAPI.
- **/predict**: score a patient encounter.
- **/patients/{id}/explain**: SHAP reasons for risk.
- UI renders: risk gauge, top drivers, and recommended preventive actions (follow-up scheduling, medication reconciliation, education).

---

## ☁️ Deploy on AWS (Preview)
- Build container with `deployment/Dockerfile`.
- Run locally with `docker build -t hpap . && docker run -p 8000:8000 hpap`.
- See `deployment/aws_instructions.md` (to be completed) for ECS/Fargate steps.

---

## 🛣️ Roadmap
- [ ] Synthetic EHR generator (structured + notes)
- [ ] HF embeddings + caching pipeline
- [ ] Model training + evaluation notebook
- [ ] SHAP explainability service
- [ ] FastAPI dashboard UI
- [ ] Dockerize & CI workflow
- [ ] AWS deployment guide
- [ ] Example screenshots & demo dataset

---

## 📇 Credit
Built by **Aditya Singh**. MIT Licensed.

## Streamlit Dashboard (Demo)

A lightweight Streamlit dashboard is included for demoing model outputs and SHAP explainability.
Run it locally (after installing Streamlit):
```bash
pip install streamlit
streamlit run src/streamlit_app.py --server.port 8501
```
The dashboard uses static demo images in `assets/` but can be wired to call the FastAPI endpoints for live predictions and explanations.

## 🖥️ Streamlit Dashboard (Demo)

A Streamlit dashboard is included for quick visualization and local demos.

Run after generating data, featurizing, and training the model:

```bash
pip install streamlit
streamlit run src/ui_streamlit.py
```

If processed features or model are missing, the dashboard will display demo screenshots located in `docs/screenshots/`.

Screenshots included:
- `docs/screenshots/risk_dashboard.png`
- `docs/screenshots/shap_bar.png`