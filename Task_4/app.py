"""
Streamlit Sentiment Analysis Application.
Connects to the FastAPI backend for predictions.
"""

import streamlit as st
import requests
import time

# ── Configuration ────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

# ── Page Setup ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #6b7280;
        font-size: 1.05rem;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .sentiment-label {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0 0.2rem;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #64748b;
    }

    /* Sentiment colours */
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    .neutral  { color: #f59e0b; }

    /* Probability bars */
    .prob-row {
        display: flex;
        align-items: center;
        margin: 0.4rem 0;
        gap: 0.6rem;
    }
    .prob-label {
        width: 80px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: capitalize;
    }
    .prob-bar-bg {
        flex: 1;
        height: 22px;
        background: #e2e8f0;
        border-radius: 11px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 11px;
        transition: width 0.6s ease;
    }
    .prob-value {
        width: 50px;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 500;
        color: #475569;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────
SENTIMENT_EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}
SENTIMENT_COLOR = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#f59e0b"}


def check_api_status() -> str:
    """Return the detailed status of the FastAPI backend ('warming_up', 'ready', or 'offline')."""
    try:
        r = requests.get(f"{API_URL}/status", timeout=3)
        if r.status_code == 200:
            return r.json().get("status", "ready")
        return "offline"
    except requests.ConnectionError:
        return "offline"
    except Exception:
        return "offline"


def predict_sentiment(text: str) -> dict | None:
    """Call the /predict endpoint and return the JSON response."""
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def render_probability_bars(probabilities: dict):
    """Render horizontal probability bars for each class."""
    for label, prob in sorted(probabilities.items()):
        color = SENTIMENT_COLOR.get(label, "#94a3b8")
        pct = prob * 100
        st.markdown(f"""
        <div class="prob-row">
            <span class="prob-label">{label}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: {pct}%; background: {color};"></div>
            </div>
            <span class="prob-value">{pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)


# ── Main UI ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎭 Sentiment Analyzer</h1>
    <p>Powered by a Random Forest model trained on Iran-War social media data</p>
</div>
""", unsafe_allow_html=True)

# API status indicator
api_status = check_api_status()
api_ok = (api_status == "ready")

if api_status == "warming_up":
    st.info("⏳ The API is currently warming up and loading the NLP dictionary. This typically takes ~15 seconds. Please wait...")
    with st.spinner("Waiting for NLP Caching to finish..."):
        while api_status == "warming_up":
            time.sleep(1.5)
            api_status = check_api_status()
        st.rerun()  # Refresh the page once it finishes loading!

elif api_status == "ready":
    st.success("✅ API is online and ready", icon="🟢")
else:
    st.error(
        "❌ Cannot reach the FastAPI backend.  \n"
        "Start it with: `uvicorn api.api:app --reload` from the Task_4 folder."
    )

st.divider()

# ── Session state init ───────────────────────────────────────
if "example_text" not in st.session_state:
    st.session_state["example_text"] = ""

# ── Text input ───────────────────────────────────────────────
text_input = st.text_area(
    "Enter text to analyze",
    value=st.session_state["example_text"],
    height=140,
    placeholder="Type or paste a sentence here...",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍  Analyze Sentiment", use_container_width=True, type="primary")

# ── Example buttons ──────────────────────────────────────────
st.markdown("**Quick examples:**")
ex_cols = st.columns(3)
examples = [
    ("😊 Positive", "The recent peace talks have concluded successfully, bringing a tremendous amount of hope for stability in the region."),
    ("😠 Negative", "War is terrible, thousands of innocent civilians are suffering and dying."),
    ("😐 Neutral",  "The embassy released a brief statement outlining the timeline of last night's events.."),
]
for col, (label, ex_text) in zip(ex_cols, examples):
    if col.button(label, use_container_width=True):
        st.session_state["example_text"] = ex_text
        st.rerun()

st.divider()

# ── Prediction ───────────────────────────────────────────────
if analyze_btn and text_input.strip():
    if not api_ok:
        st.error("API is offline. Start the FastAPI server first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_sentiment(text_input.strip())

        if result:
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]
            emoji = SENTIMENT_EMOJI.get(sentiment, "❓")
            css_class = sentiment

            # Result card
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size: 3rem;">{emoji}</div>
                <div class="sentiment-label {css_class}">{sentiment.upper()}</div>
                <div class="confidence-text">Confidence: {confidence * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("#### Class Probabilities")
            render_probability_bars(probabilities)

elif analyze_btn and not text_input.strip():
    st.warning("Please enter some text first.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Social Data Analytics Project — Task 4: Model Deployment<br>
    FastAPI + Streamlit + Random Forest (GloVe Style B)
</div>
""", unsafe_allow_html=True)
