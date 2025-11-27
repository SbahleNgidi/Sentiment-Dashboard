import os
import io
import requests
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Page Config ---
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("Sentiment Analysis Dashboard")
st.markdown("Analyze text sentiment using the **DistilBERT** model.")

# --- 2. Load Keyword Extractor ---
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

# --- 3. Configuration ---
# Hardcoded to the stable model to fix the "410 Gone" error
HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# --- 4. Token Logic ---
HF_TOKEN = os.getenv("HF_TOKEN")

# Try Secrets (Cloud)
if not HF_TOKEN and hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]

# Sidebar Token Input
# We allow the user to paste the token here if it wasn't found in the environment
if not HF_TOKEN:
    HF_TOKEN = st.sidebar.text_input("Hugging Face token", type="password", help="Paste your hf_... token here")

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- 5. API Call Function ---
def call_hf_inference(text: str):
    # Check token inside the function so we don't crash the whole app on load
    if not HF_TOKEN:
        return {"error": "Missing Token. Check Sidebar."}

    try:
        resp = requests.post(API_URL, headers=HEADERS, json={"inputs": text}, timeout=30)
        if resp.status_code == 401:
            return {"error": "Unauthorized (Check Token)"}
        if resp.status_code == 503:
            return {"error": "Model Loading... (Wait 30s)"}
        if resp.status_code != 200:
            return {"error": f"Error {resp.status_code}: {resp.text}"}
        return resp.json()
    except Exception as e:
        return {"error": f"Connection Error: {e}"}

def parse_response(resp):
    # Error handling
    if isinstance(resp, dict) and "error" in resp:
        return {"label": f"⚠️ {resp['error']}", "score": 0.0}
    
    # Successful parsing
    try:
        # Flatten list if needed
        data = resp[0] if isinstance(resp, list) else resp
        # If it's a list of dicts (standard for this model), get the highest score
        if isinstance(data, list):
            best = max(data, key=lambda x: x['score'])
            return {"label": best['label'], "score": best['score']}
    except:
        pass
    return {"label": "Parse Error", "score": 0.0}

# --- 6. UI & Logic ---
# UI is now OUTSIDE any if-blocks so it always renders
with st.expander("Single Text Input", expanded=True):
    text = st.text_area("Enter text", height=100)
    if st.button("Analyze Text"):
        if not HF_TOKEN:
            st.error("Please paste your Hugging Face Token in the sidebar first.")
        else:
            with st.spinner("Analyzing..."):
                res = parse_response(call_hf_inference(text))
                st.metric(label="Sentiment", value=res['label'], delta=f"{res['score']:.2f}")

# Batch Upload
with st.expander("Batch Upload (CSV/TXT)"):
    uploaded_file = st.file_uploader("Upload file", type=["txt", "csv"])
    if uploaded_file and st.button("Analyze File"):
        if not HF_TOKEN:
             st.error("Please paste your Hugging Face Token in the sidebar first.")
        else:
            content = uploaded_file.read()
            if uploaded_file.name.endswith(".txt"):
                lines = content.decode("utf-8", errors="ignore").splitlines()
            else:
                 # Requires import io
                 lines = pd.read_csv(io.BytesIO(content)).iloc[:,0].astype(str).tolist()
            
            results = []
            # Progress bar for feedback
            progress_bar = st.progress(0)
            
            # Limit to first 50 lines for speed/demo purposes
            limit = 50
            process_lines = [l for l in lines if l.strip()][:limit]
            
            for i, line in enumerate(process_lines): 
                res = parse_response(call_hf_inference(line.strip()))
                results.append({"text": line.strip(), "label": res['label'], "score": res['score']})
                progress_bar.progress((i + 1) / len(process_lines))
                
            st.dataframe(pd.DataFrame(results))
