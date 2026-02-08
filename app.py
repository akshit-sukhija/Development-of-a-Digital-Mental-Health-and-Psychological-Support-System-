# app.py
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional, Dict, Any

# =====================================================
# HealthNav AI
# Early Health & Mental Wellness Guidance System
# Non-diagnostic | Assistive | Ethical AI
# =====================================================

# -------------------------
# Rule-based sentiment analyzer (fallback)
# -------------------------
POS_WORDS = {
    "good", "great", "happy", "joy", "love", "awesome", "fantastic", "relieved",
    "satisfied", "hopeful", "calm", "better", "improved", "positive", "grateful",
    "amazing", "pleased", "excited"
}
NEG_WORDS = {
    "sad", "depressed", "angry", "upset", "anxious", "worried", "hopeless",
    "terrible", "awful", "stress", "stressed", "pain", "hurt", "bad", "negative",
    "lonely", "panic", "hate", "annoyed", "frustrated"
}
NEGATIONS = {"not", "no", "never", "n't", "hardly", "rarely"}
INTENSIFIERS = {"very", "extremely", "really", "so", "too", "super"}


def tokenize(text: str) -> list:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def rule_sentiment(text: str) -> Tuple[str, float]:
    tokens = tokenize(text)
    score = 0.0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        weight = 1.0
        if t in INTENSIFIERS and i + 1 < len(tokens):
            weight = 1.8
            i += 1
            t = tokens[i]
        neg_window = tokens[max(0, i - 3):i]
        is_negated = any(w in NEGATIONS for w in neg_window)
        if t in POS_WORDS:
            score += weight * (-1.0 if is_negated else 1.0)
        elif t in NEG_WORDS:
            score += weight * (1.0 if is_negated else -1.0)
        i += 1

    normalized = score / max(1.0, len(tokens))
    if normalized > 0.05:
        label = "positive"
    elif normalized < -0.05:
        label = "negative"
    else:
        label = "neutral"

    confidence = float(min(1.0, abs(normalized) * 5.0))
    return label, round(confidence, 2)


# -------------------------
# Optional ML model loader
# -------------------------
@st.cache_resource
def load_ml_model() -> Tuple[Optional[Any], Optional[Any]]:
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None


ml_model, ml_vectorizer = load_ml_model()


def predict_with_model(text: str) -> Tuple[str, Optional[float]]:
    if ml_model is None or ml_vectorizer is None:
        raise RuntimeError("Model not available")

    X = ml_vectorizer.transform([text])
    raw = ml_model.predict(X)[0]

    label = "neutral"
    if str(raw).lower().startswith("pos"):
        label = "positive"
    elif str(raw).lower().startswith("neg"):
        label = "negative"

    confidence = None
    if hasattr(ml_model, "predict_proba"):
        probs = ml_model.predict_proba(X)[0]
        confidence = float(np.max(probs))

    return label, confidence


# -------------------------
# Guidance-level mapping (NON-CLINICAL)
# -------------------------
def map_to_support_level(text: str, sentiment: str) -> Tuple[str, str]:
    t = text.lower()

    strong_distress = [
        "suicide", "kill myself", "end my life",
        "no reason to live", "self harm", "self-harm"
    ]

    moderate_distress = [
        "depressed", "anxious", "panic", "overwhelmed",
        "burnout", "lonely", "stressed"
    ]

    for k in strong_distress:
        if k in t:
            return (
                "IMMEDIATE SUPPORT RECOMMENDED",
                "Strong emotional distress signals detected. Please consider reaching out to a trusted person or a professional support service."
            )

    for k in moderate_distress:
        if k in t:
            return (
                "SUPPORT SUGGESTED",
                "Signs of emotional strain detected. Gentle self-care and professional guidance may be helpful."
            )

    if sentiment == "negative":
        return (
            "SUPPORT SUGGESTED",
            "Mild distress detected. Monitoring your well-being and using coping strategies is recommended."
        )

    return (
        "SELF-CARE GUIDANCE",
        "No critical distress signals detected. Continue healthy routines and self-care."
    )


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="HealthNav AI", layout="wide")

st.markdown("<h1 style='text-align:center'>🧭 HealthNav AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>Early Health & Mental Wellness Guidance System (Non-diagnostic)</p>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .center { max-width:900px; margin:auto; }
    .card { background:#ffffff; padding:18px; border-radius:10px; box-shadow:0 3px 10px rgba(0,0,0,0.06); }
    .badge { padding:8px 14px; border-radius:8px; font-weight:600; display:inline-block; }
    .b1 { background:#5cb85c; color:white; }
    .b2 { background:#f0ad4e; color:black; }
    .b3 { background:#d9534f; color:white; }
    </style>
    """,
    unsafe_allow_html=True
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["User Dashboard", "Well-Being Check"])
    st.markdown("---")
    method = st.radio("Analysis Method", ["Rule-based", "ML model (if available)"])

# -------------------------
# User Dashboard
# -------------------------
if page == "User Dashboard":
    st.markdown("<div class='center card'>", unsafe_allow_html=True)
    st.subheader("Early Guidance Check")
    st.write("Share a short message about how you are feeling or any concern. This is **not a diagnosis**.")

    user_input = st.text_area("", height=120)
    if st.button("Get Guidance"):
        if user_input.strip():
            try:
                if method.startswith("ML") and ml_model:
                    sentiment, conf = predict_with_model(user_input)
                else:
                    sentiment, conf = rule_sentiment(user_input)
            except Exception:
                sentiment, conf = rule_sentiment(user_input)

            level, message = map_to_support_level(user_input, sentiment)

            if "IMMEDIATE" in level:
                st.markdown("<div class='badge b3'>IMMEDIATE SUPPORT RECOMMENDED</div>", unsafe_allow_html=True)
                st.warning(message)
            elif "SUPPORT" in level:
                st.markdown("<div class='badge b2'>SUPPORT SUGGESTED</div>", unsafe_allow_html=True)
                st.info(message)
            else:
                st.markdown("<div class='badge b1'>SELF-CARE GUIDANCE</div>", unsafe_allow_html=True)
                st.success(message)

            if any(k in user_input.lower() for k in ["pain", "fever", "headache", "stomach", "injury"]):
                st.info(
                    "🩺 If this relates to a physical health concern, HealthNav AI encourages seeking appropriate medical guidance if symptoms persist or worsen."
                )

            st.session_state.history.append((user_input, sentiment))
        else:
            st.warning("Please enter some text.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Well-Being Check
# -------------------------
elif page == "Well-Being Check":
    st.markdown("<div class='center card'>", unsafe_allow_html=True)
    st.subheader("🌱 Quick Well-Being Check")
    st.write("A short self-reflection exercise (not a medical assessment).")

    questions = [
        ("How is your mood today?", [("Happy", 4), ("Okay", 2), ("Low", 1)]),
        ("How stressed do you feel?", [("Relaxed", 4), ("Somewhat stressed", 2), ("Very stressed", 1)]),
        ("How is your sleep?", [("Good", 4), ("Average", 2), ("Poor", 1)]),
    ]

    score = 0
    for i, (q, opts) in enumerate(questions):
        st.write(f"**{i+1}. {q}**")
        choice = st.radio("", [o[0] for o in opts], key=i)
        score += dict(opts)[choice]

    if st.button("View Guidance"):
        if score >= 10:
            st.success("You seem to be doing well. Keep maintaining healthy habits.")
        elif score >= 6:
            st.info("You may benefit from small self-care breaks and relaxation.")
        else:
            st.warning("You may be feeling strained. Reaching out for support could help.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:13px;color:gray;'>HealthNav AI — Early guidance, not diagnosis | AI HealthX 2026</p>",
    unsafe_allow_html=True
)
