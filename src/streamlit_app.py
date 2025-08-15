"""Streamlit dashboard wired to FastAPI backend with paginated, searchable, sortable, filterable encounters list.
Run backend first:
    uvicorn src.app:app --reload --port 8000
Then run streamlit:
    streamlit run src/streamlit_app.py --server.port 8501
"""
import streamlit as st
import requests, os, pandas as pd, numpy as np, plotly.graph_objects as go, plotly.express as px, json, time

API_BASE = st.sidebar.text_input("FastAPI base URL", value="http://localhost:8000")

st.set_page_config(page_title='Healthcare Predictive Analytics (Live)', layout='wide')
st.title("Healthcare Predictive Analytics — Live Dashboard")

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Health check / ping API"):
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5).json()
        st.sidebar.success(f"API: {r.get('status')} v{r.get('version')}")
    except Exception as e:
        st.sidebar.error(f"API unreachable: {e}")

if st.sidebar.button("Generate demo data (2000 patients)"):
    try:
        r = requests.post(f"{API_BASE}/generate-data", params={"n_patients":2000}, timeout=120)
        st.sidebar.success(r.json().get("message","ok"))
    except Exception as e:
        st.sidebar.error(f"Failed: {e}")

if st.sidebar.button("Featurize (compute embeddings)"):
    try:
        r = requests.post(f"{API_BASE}/featurize", timeout=600)
        st.sidebar.success(r.json().get("message","ok"))
    except Exception as e:
        st.sidebar.error(f"Failed: {e}")

if st.sidebar.button("Train model"):
    try:
        r = requests.post(f"{API_BASE}/train", timeout=600)
        st.sidebar.success(r.json().get("message","ok"))
    except Exception as e:
        st.sidebar.error(f"Failed: {e}")

# Sorting & filters
st.sidebar.subheader("Sorting & Filters")
sort_by = st.sidebar.selectbox("Sort by", options=["", "date", "risk_score", "patient_id"], index=0)
sort_order = st.sidebar.selectbox("Sort order", options=["desc","asc"], index=0)
date_range = st.sidebar.date_input("Admit date range", value=(None, None))
risk_range = st.sidebar.slider("Risk score range", min_value=0.0, max_value=1.0, value=(0.0,1.0), step=0.01)

# Pagination & search state
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'search' not in st.session_state:
    st.session_state.search = ""

def fetch_page(page, page_size, search, sort_by, sort_order, date_from, date_to, risk_min, risk_max):
    try:
        params = {"page": page, "page_size": page_size}
        if search:
            params["search"] = search
        if sort_by:
            params["sort_by"] = sort_by
            params["sort_order"] = sort_order
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        # risk filters only if we have a model; still pass them - server handles absence
        params["risk_min"] = risk_min
        params["risk_max"] = risk_max
        r = requests.get(f"{API_BASE}/encounters", params=params, timeout=30)
        if r.status_code==200:
            return r.json()
        else:
            st.error(f"Server returned {r.status_code}: {r.text}")
            return None
    except Exception as e:
        st.error(f"Failed to fetch encounters: {e}")
        return None

st.sidebar.subheader("Search encounters")
search_input = st.sidebar.text_input("patient_id or encounter_id", value=st.session_state.search)
if st.sidebar.button("Search") or (search_input != st.session_state.search):
    st.session_state.search = search_input
    st.session_state.page = 1

page_size = st.sidebar.selectbox("Page size", [10,15,25,50], index=1)

# parse date range
date_from = None
date_to = None
try:
    if isinstance(date_range, tuple) and date_range[0] and date_range[1]:
        date_from = date_range[0].isoformat()
        date_to = date_range[1].isoformat()
except Exception:
    date_from = None
    date_to = None

# Fetch current page
page_data = fetch_page(st.session_state.page, page_size, st.session_state.search, sort_by, sort_order, date_from, date_to, risk_range[0], risk_range[1])

if page_data:
    items = page_data.get("items", [])
    total = page_data.get("total", 0)
    total_pages = page_data.get("total_pages", 1)
    st.write(f"Showing page {page_data.get('page')} / {total_pages} — total encounters: {total}")
    # Display a radio list for selection (replaces dropdown)
    options = [f\"{it['encounter_id']} | {it['patient_id']} | admit:{it['admit_date']} | risk:{round(it.get('risk_score', 0),3) if it.get('risk_score') is not None else 'n/a'}\" for it in items]
    if options:
        selected = st.radio("Select an encounter from this page", options, index=0)
        encounter_id = selected.split("|")[0].strip()
    else:
        st.info("No encounters on this page. Try different search or filters.")
        encounter_id = None

    # Pagination buttons
    colp1, colp2, colp3 = st.columns([1,1,8])
    with colp1:
        if st.button("Previous") and st.session_state.page>1:
            st.session_state.page -= 1
            st.experimental_rerun()
    with colp2:
        if st.button("Next") and st.session_state.page < total_pages:
            st.session_state.page += 1
            st.experimental_rerun()

else:
    st.info("No encounter data available. Run featurize first.")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Predicted Readmission Risk")
    if 'encounter_id' not in st.session_state:
        st.session_state.encounter_id = None
    if encounter_id:
        st.session_state.encounter_id = encounter_id
    if st.session_state.encounter_id:
        if st.button("Get Prediction"):
            try:
                r = requests.post(f"{API_BASE}/predict", json={"encounter_id": st.session_state.encounter_id}, timeout=20)
                if r.status_code==200:
                    proba = r.json().get("readmission_risk", 0.0)
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = proba*100,
                        number = {'suffix': "%", 'valueformat': '.1f'},
                        delta = {'reference': 20, 'suffix': "%"},
                        gauge = {
                            'axis': {'range': [0,100]},
                            'bar': {'color': "darkblue"},
                            'steps' : [
                                {'range': [0,50], 'color':'lightgreen'},
                                {'range': [50,80], 'color':'orange'},
                                {'range': [80,100], 'color':'red'}
                            ]
                        },
                        title = {'text': "30-day Readmission Risk"}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Prediction failed: {r.text}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Show patient details / clinical note
        if st.button("Load Encounter Details"):
            try:
                r = requests.get(f"{API_BASE}/encounters/{st.session_state.encounter_id}", timeout=10)
                if r.status_code==200:
                    data = r.json()
                    enc = data.get("encounter", {})
                    note = data.get("clinical_note", "")
                    df = pd.DataFrame([{
                        "encounter_id": enc.get("encounter_id"),
                        "patient_id": enc.get("patient_id"),
                        "admit_date": enc.get("admit_date"),
                        "discharge_date": enc.get("discharge_date"),
                        "readmitted_30d": enc.get("readmitted_30d")
                    }])
                    st.table(df.set_index("encounter_id"))
                    if note:
                        st.markdown("**Clinical Note**")
                        st.write(note)
                else:
                    st.error(f"Failed to load: {r.text}")
            except Exception as e:
                st.error(f"Error loading encounter: {e}")

        st.subheader("Top Drivers (SHAP)")
        if st.button("Get SHAP Explanation"):
            try:
                r = requests.get(f"{API_BASE}/patients/{st.session_state.encounter_id}/explain", timeout=30)
                if r.status_code==200:
                    data = r.json()
                    if "shap_mean_abs" in data:
                        shap = data["shap_mean_abs"]
                        items = sorted(shap.items(), key=lambda x: -abs(x[1]))[:10]
                        names = [i[0] for i in items]
                        vals = [i[1] for i in items]
                        df_shap = pd.DataFrame({"feature": names, "impact": vals})
                        fig = px.bar(df_shap, x='impact', y='feature', orientation='h', title="Top SHAP feature impacts")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(data.get("warning", "No SHAP available"))
                else:
                    st.error(f"Explain failed: {r.text}")
            except Exception as e:
                st.error(f"Explain error: {e}")

with col2:
    st.subheader("Quick Actions")
    st.markdown("- Schedule follow-up within 7 days\n- Medication reconciliation\n- Home health referral\n- Social work consult")

    st.subheader("Search by Patient ID (quick)")
    pid = st.text_input("Enter patient_id (e.g., P100000)")
    if st.button("Search Patient Encounters"):
        if not pid:
            st.info("Enter a patient_id")
        else:
            try:
                r = requests.get(f"{API_BASE}/encounters", params={"search": pid, "page":1, "page_size":25}, timeout=10)
                if r.status_code==200:
                    matches = r.json().get("items", [])
                    if not matches:
                        st.info("No encounters found for this patient (ensure features created)")
                    else:
                        st.write(pd.DataFrame(matches))
                else:
                    st.error("Search failed")
            except Exception as e:
                st.error(f"Search failed: {e}")

st.markdown("---")
st.caption("This dashboard calls the FastAPI backend endpoints for live predictions and SHAP explainability. Ensure `uvicorn src.app:app --reload --port 8000` is running first.")