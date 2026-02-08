import streamlit as st
import joblib
import numpy as np
import re
import json
import random
from typing import Tuple, Optional

# -------------------------------------------------
# HealthNav AI
# Early Health & Mental Wellness Guidance Platform
# Assistive and non-diagnostic
# -------------------------------------------------

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

SYMPTOM_RULES = safe_load_json("symptom_rules.json")
SUPPORT_PHRASES = safe_load_json("mental_support_phrases.json")

# ---------------- Sentiment (rule-based fallback) ----------------

POS_WORDS = {
    "good","great","happy","joy","love","relieved","hopeful","calm",
    "better","positive","grateful","excited","satisfied"
}
NEG_WORDS = {
    "sad","depressed","angry","upset","anxious","worried","hopeless",
    "stress","stressed","pain","hurt","lonely","panic","frustrated"
}
INTENSIFIERS = {"very","extremely","really","too","so"}

def tokenize(text):
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
            weight = 1.6
            i += 1
            t = tokens[i]
        if t in POS_WORDS:
            score += weight
        elif t in NEG_WORDS:
            score -= weight
        i += 1

    norm = score / max(len(tokens), 1)
    if norm > 0.05:
        return "positive", round(min(1.0, norm * 5), 2)
    if norm < -0.05:
        return "negative", round(min(1.0, abs(norm) * 5), 2)
    return "neutral", 0.5

# ---------------- Optional ML model ----------------

@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None

ml_model, ml_vectorizer = load_ml_model()

def predict_with_model(text):
    X = ml_vectorizer.transform([text])
    raw = str(ml_model.predict(X)[0]).lower()
    label = "positive" if "pos" in raw else "negative" if "neg" in raw else "neutral"
    conf = None
    if hasattr(ml_model, "predict_proba"):
        conf = float(np.max(ml_model.predict_proba(X)[0]))
    return label, conf

# ---------------- Guidance mapping ----------------

def map_to_support_level(text: str, sentiment: str):
    t = text.lower()

    if any(k in t for k in [
        "suicide","kill myself","end my life","self harm","self-harm"
    ]):
        return (
            "PRIORITY SUPPORT ADVISED",
            "Strong emotional distress signals detected. Please consider reaching out to a trusted person or professional support service."
        )

    if any(k in t for k in [
        "depressed","anxious","panic","overwhelmed","burnout","lonely","stressed"
    ]):
        return (
            "SUPPORT MAY BE HELPFUL",
            "Signs of emotional strain detected. Gentle self-care and external support may help."
        )

    if sentiment == "negative":
        return (
            "SUPPORT MAY BE HELPFUL",
            "Mild distress signals detected. Monitoring your well-being and using coping strategies is recommended."
        )

    return (
        "SELF-CARE SUGGESTED",
        "No critical distress signals detected. Continue healthy routines and self-care."
    )

# ---------------- UI ----------------

st.set_page_config(page_title="HealthNav AI", layout="wide")

st.markdown("""
<div style="text-align:center;padding:10px 0 25px 0;">
  <h1 style="font-size:42px;">🧭 HealthNav AI</h1>
  <p style="font-size:17px;color:#9aa0a6;">
    Early Health & Mental Wellness Guidance Platform
  </p>
  <p style="font-size:13px;color:#6b7280;">
    Assistive • Non-diagnostic • Designed for early support
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.center { max-width:900px; margin:auto; }
.card {
  background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
  padding: 22px;
  border-radius: 14px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}
.badge {
  padding:8px 14px;
  border-radius:8px;
  font-weight:600;
  display:inline-block;
}
.low { background:#5cb85c; color:white; }
.mid { background:#f0ad4e; color:black; }
.high { background:#f08a5d; color:white; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["User Dashboard", "Well-Being Check"])
    st.markdown("---")
    method = st.radio("Analysis Method", ["Rule-based", "ML model (if available)"], index=0)

# ---------------- User Dashboard ----------------

if page == "User Dashboard":
    st.markdown("<div class='center card'>", unsafe_allow_html=True)
    st.subheader("🩺 Early Guidance Check")
    st.write("Share how you are feeling or any concern. This is **not a diagnosis**.")

    with st.expander("ℹ️ How this works"):
        st.markdown("""
        - You share a short concern or feeling  
        - The system identifies early emotional or health signals  
        - Guidance is provided for self-care or support escalation  

        This system does not replace medical or mental health professionals.
        """)

    user_input = st.text_area("", height=120)

    if st.button("View Guidance Summary"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            try:
                if method.startswith("ML") and ml_model:
                    sentiment, _ = predict_with_model(user_input)
                else:
                    sentiment, _ = rule_sentiment(user_input)
            except Exception:
                sentiment, _ = rule_sentiment(user_input)

            level, message = map_to_support_level(user_input, sentiment)

            if "PRIORITY" in level:
                st.markdown("<div class='badge high'>🟠 PRIORITY SUPPORT ADVISED</div>", unsafe_allow_html=True)
                st.warning(message)
            elif "SUPPORT" in level:
                st.markdown("<div class='badge mid'>🟡 SUPPORT MAY BE HELPFUL</div>", unsafe_allow_html=True)
                st.info(message)
            else:
                st.markdown("<div class='badge low'>🟢 SELF-CARE SUGGESTED</div>", unsafe_allow_html=True)
                st.success(message)

            if SUPPORT_PHRASES:
                key = "low_support" if "SELF" in level else "medium_support" if "SUPPORT" in level else "high_support"
                if key in SUPPORT_PHRASES:
                    st.write(random.choice(SUPPORT_PHRASES[key]))

            for symptom, rule in SYMPTOM_RULES.items():
                if symptom in user_input.lower():
                    st.info(rule.get("guidance"))
                    st.info(rule.get("escalation"))

            st.caption("Guidance provided is supportive and non-diagnostic.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Well-Being Check ----------------

elif page == "Well-Being Check":
    st.markdown("<div class='center card'>", unsafe_allow_html=True)
    st.subheader("🌱 Quick Well-Being Check")
    st.write("A short self-reflection exercise (not a medical assessment).")

    questions = [
        ("How is your mood today?", [("Happy",4),("Okay",2),("Low",1)]),
        ("How stressed do you feel?", [("Relaxed",4),("Somewhat stressed",2),("Very stressed",1)]),
        ("How is your sleep?", [("Good",4),("Average",2),("Poor",1)])
    ]

    score = 0
    for i,(q,opts) in enumerate(questions):
        st.write(f"**{i+1}. {q}**")
        choice = st.radio("", [o[0] for o in opts], key=i)
        score += dict(opts)[choice]

    if st.button("View Guidance"):
        if score >= 10:
            st.success("You appear to be doing well. Maintain healthy routines.")
        elif score >= 6:
            st.info("You may benefit from small self-care breaks and relaxation.")
        else:
            st.warning("You may be feeling strained. Reaching out for support could help.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ----------------

st.markdown("""
<hr>
<p style="text-align:center;font-size:12px;color:#6b7280;">
HealthNav AI is an assistive wellness guidance tool.<br>
It does not provide medical diagnosis or treatment.<br><br>
AI HealthX 2026 • Ethical AI for Healthcare
</p>
""", unsafe_allow_html=True)
