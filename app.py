import os
import io
import requests
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Page Config (Must be first) ---
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("Sentiment Analysis Dashboard")
st.markdown(
    "Analyze text sentiment using a Hugging Face Inference API model. "
    "**Note:** If you see 'Model Loading' errors, wait 30 seconds and try again."
)

# --- 2. Load Optional Libraries ---
@st.cache_resource
def load_keyword_extractor():
    try:
        with st.spinner("Loading AI models..."):
            from keybert import KeyBERT
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download('punkt_tab', quiet=True) 
            return True
    except Exception:
        return False

_KEYBERT_AVAILABLE = load_keyword_extractor()

# --- 3. Configuration (Sidebar) ---
st.sidebar.header("Configuration")

# FIXED: Default to a stable model that definitely exists
# We use the full "Organization/ModelName" format to avoid 404/410 errors
DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

HF_MODEL = st.sidebar.text_input(
    "Hugging Face model",
    value=DEFAULT_MODEL,
    help="Model ID to call via the HF Inference API."
)

# Token Logic
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN and hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]

# Sidebar Token Override
sidebar_token = st.sidebar.text_input(
    "Hugging Face token",
    value=HF_TOKEN if HF_TOKEN else "",
    type="password",
    help="Paste your hf_... token here if not using Secrets"
)
if sidebar_token:
    HF_TOKEN = sidebar_token.strip()

if not HF_TOKEN:
    st.warning("⚠️ No Hugging Face token found. API calls may fail. Please add it in the sidebar.")

# --- 4. API Logic ---
API_URL_BASE = "https://api-inference.huggingface.co/models"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def call_hf_inference(text: str, model: str = HF_MODEL):
    if not HF_TOKEN:
        return {"error": "Missing Token"}
    
    url = f"{API_URL_BASE}/{model}"
    try:
        resp = requests.post(url, headers=HEADERS, json={"inputs": text}, timeout=30)
        
        if resp.status_code == 401:
            return {"error": "401 Unauthorized (Check Token)"}
        if resp.status_code == 404:
            return {"error": f"404 Model Not Found: {model}"}
        if resp.status_code == 410:
             return {"error": "410 Model Deleted/Gone"}
        if resp.status_code == 503:
            return {"error": "503 Loading... (Try again in 30s)"}
            
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Connection Error: {e}"}

def parse_hf_response(resp: Any) -> Dict[str, Any]:
    out = {"label": None, "score": None, "raw": resp}
    
    # Handle explicit errors
    if isinstance(resp, dict) and "error" in resp:
        out["label"] = f"⚠️ {resp['error']}"
        return out

    try:
        # Flatten list if needed (API returns list of lists for some models)
        data = resp
        if isinstance(data, list) and data:
            if isinstance(data[0], list):
                data = data[0]
            
            # Find best score
            if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                best = max(data, key=lambda x: x.get("score", 0))
                out["label"] = best.get("label")
                out["score"] = best.get("score")
                return out
                
        # Handle single dict response
        if isinstance(data, dict) and "label" in data:
            out["label"] = data.get("label")
            out["score"] = data.get("score")
            return out
            
    except Exception:
        pass

    out["label"] = "PARSING ERROR"
    return out

# --- 5. UI & Processing ---
results = []

# Helper to process text and save result
def analyze_and_record(t: str, source: str = "input"):
    parsed = parse_hf_response(call_hf_inference(t))
    row = {
        "source": source,
        "text": t,
        "label": parsed["label"],
        "score": parsed["score"],
        "raw": str(parsed["raw"])
    }
    
    # Keyword extraction (only if library loaded and no error)
    if _KEYBERT_AVAILABLE and "⚠️" not in str(row["label"]):
        try:
            from keybert import KeyBERT 
            kw_model = KeyBERT()
            kws = kw_model.extract_keywords(t, top_n=5)
            row["keywords"] = ", ".join([kw for kw, _ in kws])
        except:
            row["keywords"] = ""
            
    results.append(row)

# Input Section
with st.expander("Single text input", expanded=True):
    text_in = st.text_area("Enter text to analyze", height=100)
    if st.button("Analyze text"):
        if text_in.strip():
            with st.spinner("Analyzing..."):
                analyze_and_record(text_in.strip(), "single")
        else:
            st.info("Enter some text first.")

# Batch Section
with st.expander("Batch / File upload"):
    uploaded_file = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
    if uploaded_file and st.button("Analyze uploaded file"):
        content = uploaded_file.read()
        
        # Parse file type
        lines = []
        if uploaded_file.name.endswith(".txt"):
            lines = content.decode("utf-8", errors="ignore").splitlines()
        else:
            try:
                df_temp = pd.read_csv(io.BytesIO(content))
                # Find first text column
                text_cols = [c for c in df_temp.columns if df_temp[c].dtype == "object"]
                if text_cols:
                    lines = df_temp[text_cols[0]].dropna().astype(str).tolist()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

        # Process lines (limit to 50 for safety)
        if lines:
            progress = st.progress(0)
            limit = 50
            subset = [l for l in lines if l.strip()][:limit]
            
            for i, line in enumerate(subset):
                analyze_and_record(line, f"file row {i+1}")
                progress.progress((i + 1) / len(subset))
            
            if len(lines) > limit:
                st.warning(f"Stopped after {limit} rows to preserve API limits.")

# --- 6. Results Display ---
if results:
    df_res = pd.DataFrame(results)
    st.subheader("Results")
    st.dataframe(df_res)

    # Charts
    valid_df = df_res[df_res["score"].notna()]
    if not valid_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Sentiment Distribution")
            counts = valid_df["label"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            st.pyplot(fig)
        with col2:
            st.caption("Confidence Scores")
            fig2, ax2 = plt.subplots()
            ax2.hist(valid_df["score"], bins=10)
            st.pyplot(fig2)
