import os
import requests
from typing import Any, Dict, List

import streamlit as stimport os
import io  # Added missing import
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
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Page Config (Must be the very first Streamlit command) ---
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# --- 2. Render Title Immediately ---
st.title("Sentiment Analysis Dashboard")
st.markdown(
    "Analyze text sentiment using a Hugging Face Inference API model. "
    "**Note:** If you see 'Model Loading' errors, wait 30 seconds and try again."
)

# --- 3. Load Optional Libraries ---
@st.cache_resource
def load_keyword_extractor():
    try:
        with st.spinner("Loading keyword extraction models..."):
            from keybert import KeyBERT
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download('punkt_tab', quiet=True) 
            return True
    except Exception as e:
        return False

_KEYBERT_AVAILABLE = load_keyword_extractor()

# -------- Configuration --------
# UPDATED: The previous CardiffNLP models were returning 410 (Gone).
# Switched to 'distilbert-base-uncased-finetuned-sst-2-english' which is highly stable.
# Added key="model_id_v2" to force the sidebar to update to this new default.
HF_MODEL = st.sidebar.text_input(
    "Hugging Face model",
    value="distilbert-base-uncased-finetuned-sst-2-english",
    key="model_id_v2",
    help="Model ID to call via the HF Inference API."
)

# --- TOKEN LOGIC (Updated for Cloud) ---
# 1. Try Environment Variable (Local/System)
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Try Streamlit Secrets (Cloud Deployment)
if not HF_TOKEN and hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HF_TOKEN"]

# 3. Try Sidebar (Manual Entry)
_sidebar_token = st.sidebar.text_input(
    "Hugging Face token (paste if not in secrets)", 
    value="", 
    type="password"
)
if not HF_TOKEN and _sidebar_token:
    HF_TOKEN = _sidebar_token.strip()

if not HF_TOKEN:
    st.warning("⚠️ No Hugging Face token found. The API call will likely fail (401 Unauthorized). Please add it to your Secrets or the Sidebar.")

API_URL_BASE = "https://api-inference.huggingface.co/models"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# We do NOT cache this function anymore so that if you fix the token, 
# you don't get the old "Error" result from the cache.
def call_hf_inference(text: str, model: str = HF_MODEL, headers: Dict[str, str] = HEADERS) -> Any:
    """Call Hugging Face Inference API. Returns JSON or dict with specific error info."""
    if not headers:
        return {"error": "Missing Token. Please set HF_TOKEN."}
    
    url = f"{API_URL_BASE}/{model}"
    payload = {"inputs": text}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        
        # Handle specific HTTP errors
        if resp.status_code == 401:
            return {"error": "401 Unauthorized. Check your HF_TOKEN."}
        if resp.status_code == 404 or resp.status_code == 410:
             return {"error": f"{resp.status_code} Model not found. Check model ID."}
        if resp.status_code == 503:
            return {"error": "503 Model Loading. Please wait 30s and retry."}
        
        resp.raise_for_status() # Raise error for other 4xx/5xx
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}

def parse_hf_response(resp: Any) -> Dict[str, Any]:
    out = {"label": None, "score": None, "raw": resp}
    
    # Check for explicit error keys in the JSON
    if isinstance(resp, dict) and "error" in resp:
        # Pass the actual error message to the label so the user sees it
        out["label"] = f"⚠️ {resp['error']}"
        return out

    try:
        # Standard list-of-dicts response
        if isinstance(resp, list) and resp:
            # Flatten if nested (some models return [[{...}]])
            candidate = None
            if isinstance(resp[0], list) and resp[0]:
                inner = resp[0]
                if all(isinstance(i, dict) for i in inner):
                    candidate = max(inner, key=lambda x: x.get("score", 0))
            elif all(isinstance(i, dict) for i in resp):
                candidate = max(resp, key=lambda x: x.get("score", 0))
            elif isinstance(resp[0], dict):
                candidate = resp[0]

            if candidate and "label" in candidate:
                out["label"] = candidate["label"]
                out["score"] = float(candidate.get("score", 0.0))
                return out

        # Single dict response
        if isinstance(resp, dict) and "label" in resp:
            out["label"] = resp["label"]
            out["score"] = float(resp.get("score", 0.0))
            return out

    except Exception:
        pass

    out["label"] = "PARSING ERROR (Check 'raw' column)"
    return out

# -------- UI: Inputs --------
with st.expander("Single text input", expanded=True):
    text = st.text_area("Enter text to analyze", height=100)
    analyze_single = st.button("Analyze text")

with st.expander("Batch / File upload"):
    uploaded_file = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
    analyze_batch = st.button("Analyze uploaded file")

# -------- Processing --------
results: List[Dict[str, Any]] = []
kw_extract = False
if _KEYBERT_AVAILABLE:
    kw_extract = st.sidebar.checkbox("Enable Keyword Extraction", value=False)

def analyze_and_record(t: str, source: str = "input"):
    parsed = parse_hf_response(call_hf_inference(t))
    row = {
        "text": t,
        "label": parsed.get("label"),
        "score": parsed.get("score"),
        "raw": str(parsed.get("raw")),
        "source": source,
    }
    # Keyword extraction (only if success)
    if kw_extract and _KEYBERT_AVAILABLE and "⚠️" not in str(row["label"]):
        try:
            from keybert import KeyBERT 
            kw_model = KeyBERT()
            kws = kw_model.extract_keywords(t, top_n=5)
            row["keywords"] = ", ".join([kw for kw, _ in kws])
        except Exception:
            row["keywords"] = ""
    results.append(row)

if analyze_single and text:
    with st.spinner("Analyzing..."):
        analyze_and_record(text.strip(), source="single")

if analyze_batch and uploaded_file:
    content = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        lines = content.decode("utf-8", errors="ignore").splitlines()
        with st.spinner(f"Analyzing {len(lines)} lines..."):
            for i, line in enumerate(lines):
                if line.strip():
                    analyze_and_record(line.strip(), source=f"line {i+1}")
    else:
        try:
            df_in = pd.read_csv(io.BytesIO(content))
            text_cols = [c for c in df_in.columns if df_in[c].dtype == "object"]
            if text_cols:
                col = st.selectbox("Choose text column", text_cols)
                if st.button("Run CSV Analysis"):
                    for idx, t in enumerate(df_in[col].dropna().astype(str)):
                        analyze_and_record(t, source=f"row {idx+1}")
        except Exception as e:
            st.error(f"CSV Error: {e}")

# -------- Results --------
if results:
    df = pd.DataFrame(results)
    st.subheader("Results")
    st.dataframe(df)

    # Only show charts if we have valid scores (no errors)
    valid_df = df[df["score"].notna()]
    
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
    else:
        st.warning("No valid results to plot. Check the 'label' column for error messages.")
