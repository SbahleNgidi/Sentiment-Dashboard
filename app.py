import os
import io
import requests
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Page Config (Must be the very first Streamlit command) ---
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# --- 2. Render Title Immediately (Prevents blank screen) ---
st.title("Sentiment Analysis Dashboard")
st.markdown(
    "Analyze text sentiment using a Hugging Face Inference API model. "
    "Set `HF_TOKEN` as an environment variable for deployments, or paste it into the sidebar (local testing)."
)

# --- 3. Load Optional Libraries (Lazy Loading) ---
# We use @st.cache_resource so this only runs once and doesn't freeze the app on every reload
@st.cache_resource
def load_keyword_extractor():
    try:
        # Show a status on the UI while this runs
        with st.spinner("Loading keyword extraction models (this may take a minute)..."):
            from keybert import KeyBERT
            import nltk
            # Download necessary NLTK data
            nltk.download("punkt", quiet=True)
            nltk.download('punkt_tab', quiet=True) 
            return True
    except Exception as e:
        print(f"Keyword extraction disabled: {e}")
        return False

# Check availability without blocking the whole UI immediately
_KEYBERT_AVAILABLE = load_keyword_extractor()

# -------- Configuration --------
HF_MODEL = st.sidebar.text_input(
    "Hugging Face model (inference API)",
    value="cardiffnlp/twitter-roberta-base-sentiment",
    help="Model ID to call via the HF Inference API (e.g. cardiffnlp/twitter-roberta-base-sentiment)",
)

# Prefer env var HF_TOKEN; fall back to secure sidebar input for local testing
_env_token = os.getenv("HF_TOKEN")
_sidebar_token = st.sidebar.text_input(
    "Hugging Face token (paste here if HF_TOKEN not set)", value="", type="password"
)
HF_TOKEN = _env_token or (_sidebar_token.strip() if _sidebar_token else None)

if not HF_TOKEN:
    st.sidebar.warning(
        "No Hugging Face token found. Set HF_TOKEN environment variable or paste it in the sidebar to call the API."
    )

API_URL_BASE = "https://api-inference.huggingface.co/models"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# Cache API calls to avoid repeating identical requests
@st.cache_data(show_spinner=False)
def call_hf_inference(text: str, model: str = HF_MODEL, headers: Dict[str, str] = HEADERS) -> Any:
    """Call Hugging Face Inference API. Returns JSON or dict with 'error'."""
    if not headers:
        return {"error": "Missing Hugging Face token. Set HF_TOKEN env var or paste it in the sidebar."}
    url = f"{API_URL_BASE}/{model}"
    payload = {"inputs": text}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Request failed: {e}"}

def parse_hf_response(resp: Any) -> Dict[str, Any]:
    """
    Robustly parse common HF Inference API shapes:
      - list of dicts: [{'label':..., 'score':...}, ...]
      - nested lists: [[{'label':..., 'score':...}], ...]
      - single dict: {'label':..., 'score':...}
      - error dict: {'error': '...'}
    Returns dict: {'label': str|None, 'score': float|None, 'raw': resp}
    """
    out = {"label": None, "score": None, "raw": resp}
    try:
        if isinstance(resp, dict) and "error" in resp:
            out["label"] = "ERROR"
            return out

        # list-of-dicts or nested list
        if isinstance(resp, list) and resp:
            # flatten one level if nested lists present
            candidate = None
            # If first element is a list of dicts
            if isinstance(resp[0], list) and resp[0]:
                inner = resp[0]
                if all(isinstance(i, dict) for i in inner):
                    candidate = max(inner, key=lambda x: x.get("score", 0))
            # If resp itself is a list of dicts
            elif all(isinstance(i, dict) for i in resp):
                candidate = max(resp, key=lambda x: x.get("score", 0))
            # If first element is dict (and not all are dicts), try first
            elif isinstance(resp[0], dict):
                candidate = resp[0]

            if candidate and isinstance(candidate, dict) and "label" in candidate and "score" in candidate:
                out["label"] = candidate["label"]
                out["score"] = float(candidate["score"])
                return out

        # single dict with label/score
        if isinstance(resp, dict) and "label" in resp and "score" in resp:
            out["label"] = resp["label"]
            out["score"] = float(resp["score"])
            return out

    except Exception:
        pass

    # Fallback
    out["label"] = "UNKNOWN"
    return out

# -------- UI: Inputs --------
with st.expander("Single text input", expanded=True):
    text = st.text_area("Enter text to analyze", height=150, placeholder="Type or paste text here...")
    analyze_single = st.button("Analyze text")

with st.expander("Batch / File upload"):
    uploaded_file = st.file_uploader("Upload a text (.txt) or CSV (.csv) file for batch processing", type=["txt", "csv"])
    analyze_batch = st.button("Analyze uploaded file")

# Optional keyword extraction
kw_extract = False
if _KEYBERT_AVAILABLE:
    kw_extract = st.sidebar.checkbox("Enable keyword extraction (KeyBERT)", value=False)
else:
    st.sidebar.caption("Keyword extraction disabled (Library not loaded)")

# -------- Processing --------
results: List[Dict[str, Any]] = []

def analyze_and_record(t: str, source: str = "input"):
    parsed = parse_hf_response(call_hf_inference(t))
    row = {
        "text": t,
        "label": parsed.get("label"),
        "score": parsed.get("score"),
        "raw": str(parsed.get("raw")),
        "source": source,
    }
    if kw_extract and _KEYBERT_AVAILABLE:
        try:
            # Local import to prevent errors if library missing
            from keybert import KeyBERT 
            kw_model = KeyBERT()
            kws = kw_model.extract_keywords(t, top_n=5)
            row["keywords"] = ", ".join([kw for kw, _ in kws])
        except Exception:
            row["keywords"] = ""
    results.append(row)

if analyze_single:
    if text and text.strip():
        with st.spinner("Calling Hugging Face inference..."):
            analyze_and_record(text.strip(), source="single")
    else:
        st.info("Please enter text in the text area above before analyzing.")

if analyze_batch and uploaded_file:
    content = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        try:
            lines = content.decode("utf-8", errors="ignore").splitlines()
        except Exception:
            lines = []
        with st.spinner("Analyzing lines from TXT file..."):
            for i, line in enumerate(lines):
                if line.strip():
                    analyze_and_record(line.strip(), source=f"file:{uploaded_file.name}:line{i+1}")
    else:
        # CSV handling
        try:
            df_in = pd.read_csv(io.BytesIO(content))
            text_cols = [c for c in df_in.columns if df_in[c].dtype == "object"]
            if not text_cols:
                st.error("No text column detected in CSV. Ensure you have at least one column with textual data.")
            else:
                col = st.selectbox("Choose text column to analyze from CSV", text_cols)
                if st.button("Start CSV analysis"):
                    with st.spinner("Analyzing rows from CSV..."):
                        for idx, t in enumerate(df_in[col].dropna().astype(str)):
                            analyze_and_record(t, source=f"file:{uploaded_file.name}:row{idx+1}")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# -------- Results & Visuals --------
if results:
    df = pd.DataFrame(results)
    st.subheader("Results")
    st.write(f"Total texts analyzed: {len(df)}")
    display_cols = ["source", "text", "label", "score"]
    if "keywords" in df.columns:
        display_cols.append("keywords")
    st.dataframe(df[display_cols])

    # Sentiment distribution
    st.subheader("Sentiment distribution")
    counts = df["label"].fillna("UNKNOWN").value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=counts.index.tolist(), autopct="%1.1f%%")
    ax.set_aspect("equal")
    st.pyplot(fig)

    # Score histogram
    if df["score"].notna().any():
        st.subheader("Confidence score distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df["score"].dropna(), bins=10)
        ax2.set_xlabel("Confidence score")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    # Download results
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", csv_bytes, file_name="sentiment_results.csv", mime="text/csv")

# -------- Evaluation helper --------
st.markdown("---")
st.subheader("Quick evaluation (manual labels)")
st.markdown("Upload a CSV with `text` and `manual_label` columns to compute quick predictions (no detailed metrics unless scikit-learn is installed).")
eval_file = st.file_uploader("Upload CSV for evaluation (text + manual_label)", type=["csv"], key="eval")
if eval_file:
    try:
        eval_df = pd.read_csv(eval_file)
        if "text" not in eval_df.columns or "manual_label" not in eval_df.columns:
            st.error("CSV must contain `text` and `manual_label` columns.")
        else:
            pred_rows = []
            with st.spinner("Running predictions for evaluation..."):
                for idx, t in enumerate(eval_df["text"].astype(str)):
                    parsed = parse_hf_response(call_hf_inference(t))
                    pred_rows.append(parsed.get("label"))
            eval_df["predicted_label"] = pred_rows

            try:
                from sklearn.metrics import classification_report, confusion_matrix

                report = classification_report(eval_df["manual_label"], eval_df["predicted_label"])
                st.text("Classification report:\n" + report)
                labels = list(eval_df["manual_label"].unique())
                cm = confusion_matrix(eval_df["manual_label"], eval_df["predicted_label"], labels=labels)
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                st.write("Confusion matrix:")
                st.dataframe(cm_df)
            except Exception:
                st.write("Install scikit-learn to get a detailed classification report and confusion matrix.")
    except Exception as e:
        st.error(f"Failed to read evaluation CSV: {e}")

st.markdown("---")
st.caption("Built with Streamlit & Hugging Face Inference API — set HF_TOKEN as env var or provide in sidebar for local testing.")
