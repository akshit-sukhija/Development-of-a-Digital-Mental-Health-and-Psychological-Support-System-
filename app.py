# app.py
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional, Dict, Any

# -------------------------
# Rule-based analyzer (fallback)
# -------------------------
POS_WORDS = {
    "good", "great", "happy", "joy", "love", "awesome", "fantastic", "relieved",
    "satisfied", "hopeful", "calm", "better", "improved", "positive", "grateful",
    "amazing", "pleased", "excited"
}
NEG_WORDS = {
    "sad", "depressed", "angry", "upset", "anxious", "worried", "hopeless",
    "terrible", "awful", "stress", "stressed", "pain", "hurt", "bad", "negative",
    "lonely", "suicidal", "panic", "hate", "annoyed", "frustrated"
}
NEGATIONS = {"not", "no", "never", "n't", "hardly", "rarely"}
INTENSIFIERS = {"very", "extremely", "really", "so", "too", "super"}


def tokenize(text: str) -> list:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def rule_sentiment(text: str) -> Tuple[str, float]:
    """
    Simple lexicon-based sentiment: returns (label, confidence)
    """
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
    denom = max(1.0, len(tokens))
    normalized = score / denom
    if normalized > 0.05:
        label = "positive"
    elif normalized < -0.05:
        label = "negative"
    else:
        label = "neutral"
    confidence = float(min(1.0, abs(normalized) * 5.0))
    return label, round(confidence, 2)


# -------------------------
# ML model loader
# -------------------------
@st.cache_resource
def load_ml_model() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Attempts to load model and vectorizer from models/*.pkl
    Returns (model, vectorizer) or (None, None) if not present or fails.
    """
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None


ml_model, ml_vectorizer = load_ml_model()


# -------------------------
# Helpers to robustly map ML outputs to canonical labels
# -------------------------
def guess_label_from_string(raw: Any) -> Optional[str]:
    try:
        s = str(raw).lower()
    except Exception:
        return None
    if any(k in s for k in ["pos", "positive", "+", "good", "happy"]):
        return "positive"
    if any(k in s for k in ["neg", "negative", "-", "sad", "bad", "angry"]):
        return "negative"
    if "neu" in s or "neutral" in s:
        return "neutral"
    return None


def infer_mapping_using_probes(model: Any, vectorizer: Any) -> Dict[Any, str]:
    """
    Probe the model with three clear sentences to map raw outputs to canonical labels.
    Returns {raw_value: canonical_label}
    If anything fails, returns empty dict.
    """
    mapping: Dict[Any, str] = {}
    if model is None or vectorizer is None:
        return mapping
    probes = {
        "positive": "I am very happy and I love this. Absolutely fantastic and great!",
        "negative": "I am very sad and I hate this. Absolutely terrible and awful.",
        "neutral":  "It was okay, average day, nothing special."
    }
    try:
        for canonical, text in probes.items():
            X = vectorizer.transform([text])
            raw = model.predict(X)[0]
            mapping[raw] = canonical
    except Exception:
        return {}
    return mapping


def predict_with_model(text: str) -> Tuple[str, Optional[float]]:
    """
    Predict canonical label + confidence using ML model.
    If model mapping is not determinable, raises RuntimeError to trigger fallback.
    """
    if ml_model is None or ml_vectorizer is None:
        raise RuntimeError("Model not loaded")

    X = ml_vectorizer.transform([text])

    # raw prediction
    raw_pred = ml_model.predict(X)[0]

    # mapping from probes
    mapping = infer_mapping_using_probes(ml_model, ml_vectorizer)
    label = mapping.get(raw_pred)

    # try string guess on raw_pred (covers textual labels like 'POS','NEG')
    if label is None:
        label = guess_label_from_string(raw_pred)

    probs = None
    classes = getattr(ml_model, "classes_", None)
    try:
        if hasattr(ml_model, "predict_proba"):
            probs = ml_model.predict_proba(X)[0]
    except Exception:
        probs = None

    # If label still unknown, attempt to use classes/probs heuristics
    if label is None and classes is not None:
        try:
            if probs is not None:
                best_idx = int(np.argmax(probs))
                candidate = classes[best_idx]
                label = guess_label_from_string(candidate)
            else:
                ordered = list(sorted(classes))
                raw_selected = raw_pred
                # numeric classes heuristic
                if len(ordered) == 3:
                    if raw_selected == ordered[0]:
                        label = "negative"
                    elif raw_selected == ordered[1]:
                        label = "neutral"
                    else:
                        label = "positive"
                elif len(ordered) == 2:
                    label = "positive" if raw_selected == ordered[1] else "negative"
        except Exception:
            label = None

    # final fallback -> if still None, use rule-based
    if label is None:
        raise RuntimeError("Unable to map model output to canonical label")

    # confidence
    conf = None
    if probs is not None:
        conf = float(np.max(probs))
        # for binary models, apply neutral band if confidence low
        if len(probs) == 2 and conf < 0.60:
            label = "neutral"

    return label, (None if conf is None else round(conf, 2))


# -------------------------
# New: Risk mapping + triage report (minimal rule-based clinical layer)
# -------------------------
def map_sentiment_to_risk(text: str, sentiment_label: str) -> Tuple[str, str]:
    """
    Map sentiment + keyword signals into clinical triage categories.
    Returns (risk_level, recommendation)
    """
    text_lower = (text or "").lower()

    # high-risk explicit indicators
    high_risk_keywords = [
        "suicide", "kill myself", "end my life", "end it all", "hopeless",
        "self harm", "self-harm", "no reason to live", "worthless", "want to die"
    ]

    medium_risk_keywords = [
        "depressed", "depression", "anxious", "anxiety", "overwhelmed",
        "stressed", "lonely", "burnout", "panic attack", "panic"
    ]

    for kw in high_risk_keywords:
        if kw in text_lower:
            return "HIGH", "Immediate professional intervention advised. If imminent harm is mentioned, contact emergency services or crisis line."

    for kw in medium_risk_keywords:
        if kw in text_lower:
            return "MEDIUM", "Signs of significant distress. Recommend scheduling a counseling / mental health consultation soon."

    # fallback rules using sentiment
    if sentiment_label == "negative":
        return "MEDIUM", "Emotional distress detected. Consider supportive resources and follow up."

    return "LOW", "No critical distress detected. Encourage self-care and monitoring."

def build_triage_report(text: str, risk: str, recommendation: str) -> str:
    """
    Construct a structured triage report for display.
    """
    indicators = []
    if risk == "HIGH":
        indicators = ["Suicidal ideation or explicit harm language"]
    elif risk == "MEDIUM":
        indicators = ["Depressive/anxiety language", "Stress/overwhelm signals"]
    else:
        indicators = ["No acute distress signals"]

    indicators_md = "\n".join(f"- {i}" for i in indicators)

    report = f"""
**Mental Risk Level:** **{risk}**

**Detected Indicators:**
{indicators_md}

**Triage Recommendation:**
{recommendation}

**Disclaimer:** This tool is for early screening and support. It does NOT replace clinical diagnosis. Always consult a licensed professional for assessment.
"""
    return report

# -------------------------
# Streamlit UI + logic
# -------------------------
st.set_page_config(page_title="Digital Mental Health", layout="wide")
st.markdown("<h1 style='text-align:center'>💬 Digital Mental Health Support</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>Mental Health Risk Assessment — quick well-being check and early triage. Optional ML model: place files in <code>models/</code>.</p>",
    unsafe_allow_html=True,
)

# Minimal styling for clean presentation
st.markdown(
    """
<style>
.center-block { max-width: 900px; margin:auto; }
.card { background:white; padding:18px; border-radius:10px; box-shadow:0 3px 10px rgba(0,0,0,0.06); }
.small { color:#666; font-size:14px; }
.risk-badge { padding:8px 12px; border-radius:8px; color:white; display:inline-block; font-weight:600; }
.risk-high { background:#d9534f; }
.risk-medium { background:#f0ad4e; color:#1f1f1f; }
.risk-low { background:#5cb85c; }
</style>
""",
    unsafe_allow_html=True,
)

# initialize session state
if "history" not in st.session_state:
    st.session_state.history = []  # tuples: (text, label, confidence, method)
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []  # tuples: (score, message)
if "reports" not in st.session_state:
    st.session_state.reports = []  # list of triage reports (text, risk, recommendation)

# Sidebar: navigation + options
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Student Dashboard", "Well-Being Check", "Admin Dashboard"])
    st.markdown("---")
    st.title("Options")
    method_choice = st.radio("Prediction method:", ["Rule-based (no dataset)", "ML model (if available)"])
    if method_choice == "ML model (if available)":
        if ml_model is None:
            st.warning("ML model not found — using rule-based fallback.")
        else:
            st.success("ML model loaded — using ML predictions where possible.")

# ---------- Student Dashboard ----------
if page == "Student Dashboard":
    st.markdown("<div class='center-block card'>", unsafe_allow_html=True)
    st.subheader("Mental Health Risk Assessment")
    st.write("<div class='small'>Enter a short message or feedback and press Assess to get a quick triage summary (not a diagnosis).</div>", unsafe_allow_html=True)

    user_input = st.text_area("", height=110)
    col1, col2, col3 = st.columns([1.2, 1.2, 1.2])
    with col1:
        analyze_btn = st.button("Assess Mental Risk", use_container_width=True)
    with col2:
        clear_recent = st.button("Clear Recent Analyses", use_container_width=True)
    with col3:
        st.write("")

    if clear_recent:
        st.session_state.history = []
        st.session_state.reports = []
        st.success("Cleared recent analyses and reports.")

    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter text to analyze.")
        else:
            # prefer ML if chosen and available
            if method_choice == "ML model (if available)" and ml_model is not None and ml_vectorizer is not None:
                try:
                    label, confidence = predict_with_model(user_input)
                    used_method = "ml_model"
                except Exception:
                    # fallback quietly to rule-based
                    label, confidence = rule_sentiment(user_input)
                    used_method = "rule_based"
            else:
                label, confidence = rule_sentiment(user_input)
                used_method = "rule_based"

            # store basic sentiment history (keeps your existing tables/charts working)
            st.session_state.history.append((user_input, label, confidence, used_method))

            # --- new clinical triage layer ---
            risk, recommendation = map_sentiment_to_risk(user_input, label)
            report_md = build_triage_report(user_input, risk, recommendation)
            st.session_state.reports.append((user_input, risk, recommendation, report_md))

            # show colored risk badge + structured triage report
            if risk == "HIGH":
                st.markdown(f"<div class='risk-badge risk-high'>HIGH RISK</div>", unsafe_allow_html=True)
                st.error(report_md, icon="⚠️")
            elif risk == "MEDIUM":
                st.markdown(f"<div class='risk-badge risk-medium'>MEDIUM RISK</div>", unsafe_allow_html=True)
                st.warning(report_md)
            else:
                st.markdown(f"<div class='risk-badge risk-low'>LOW RISK</div>", unsafe_allow_html=True)
                st.success(report_md)

            # show helpful tips (existing behavior)
            if label == "positive":
                st.markdown("- Keep journaling positive moments\n- Share your happiness with someone")
            elif label == "negative":
                st.markdown("- Try 2 minutes deep breathing\n- Take a short walk or speak with a friend or counselor")
            else:
                st.markdown("- Try a small uplifting activity: listen to music, stretch, hydrate")

    st.markdown("---")
    st.subheader("Recent Analyses")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history[::-1], columns=["Comment", "Sentiment", "Confidence", "Method"]).head(10)
        st.table(df_hist)
    else:
        st.info("No analyses yet — add one above to populate the list.")

    st.markdown("---")
    st.subheader("Sentiment Distribution")
    if st.session_state.history:
        preds = [lbl for _, lbl, _, _ in st.session_state.history]
        counts = {
            "positive": preds.count("positive"),
            "negative": preds.count("negative"),
            "neutral": preds.count("neutral"),
        }
        keys = list(counts.keys())
        vals = [counts[k] for k in keys]
        total = sum(vals)
        if total == 0:
            st.info("No sentiment data yet.")
        else:
            # create donut pie + legend to avoid overlapping labels
            fig, ax = plt.subplots(figsize=(6, 4), dpi=90)
            wedges, texts = ax.pie(vals, startangle=90, wedgeprops=dict(width=0.45), labels=None)
            ax.set(aspect="equal")
            # center text total
            ax.text(0, 0, f"{total}\nresponses", ha="center", va="center", fontsize=12)
            # legend on the right
            ax.legend(wedges, [f"{k} ({counts[k]})" for k in keys], title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.3, 1))
            st.pyplot(fig, bbox_inches="tight")
    else:
        st.write("No data yet to show distribution.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Well-Being Check ----------
elif page == "Well-Being Check":
    st.markdown("<div class='center-block card'>", unsafe_allow_html=True)
    st.subheader("🌱 Quick Well-being Check")
    st.write("<div class='small'>5 short questions — quick self-check (not a diagnosis).</div>", unsafe_allow_html=True)

    QUESTIONS = [
        {"q": "How is your mood today?", "options": {"😊 Happy/positive": 4, "😐 Neutral/okay": 2, "😞 Sad/low": 1}},
        {"q": "How much energy do you have right now?", "options": {"⚡ Full of energy": 4, "🙂 Somewhat energetic": 3, "😴 Tired/exhausted": 1}},
        {"q": "How stressed do you feel?", "options": {"😌 Calm/relaxed": 4, "😕 A bit stressed": 2, "😫 Very stressed": 1}},
        {"q": "How was your sleep recently?", "options": {"🌙 Slept well": 4, "😶 Average sleep": 2, "🥱 Poor sleep": 1}},
        {"q": "How connected do you feel socially?", "options": {"👥 Very connected": 4, "🙂 Somewhat connected": 2, "😔 Lonely/isolated": 1}},
    ]

    responses = []
    for i, q in enumerate(QUESTIONS):
        st.write(f"**{i+1}. {q['q']}**")
        choice = st.radio("", list(q["options"].keys()), key=f"wb_q{i}")
        responses.append(q["options"][choice])

    if st.button("Submit Well-being Check", use_container_width=True):
        total = sum(responses)
        if total >= 17:
            msg = "🌟 You seem happy and doing well! Keep up your positive habits."
            st.success(msg)
        elif 12 <= total < 17:
            msg = "🙂 You seem okay, but could use some self-care. Try short breaks and relaxation."
            st.info(msg)
        else:
            msg = "💡 You may be stressed or feeling low. Consider relaxation techniques or reaching out for support."
            st.warning(msg)
        st.session_state.quiz_results.append((total, msg))
        st.markdown("<div class='small'>Note: This is not a clinical diagnosis.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Admin Dashboard ----------
elif page == "Admin Dashboard":
    st.markdown("<div class='center-block card'>", unsafe_allow_html=True)
    st.subheader("Admin Dashboard")

    if not st.session_state.history and not st.session_state.quiz_results:
        st.info("No feedback or quiz results yet.")
    else:
        if st.session_state.history:
            st.write("Student Comments")
            df = pd.DataFrame(st.session_state.history, columns=["Comment", "Sentiment", "Confidence", "Method"])
            st.table(df)
            st.write("Sentiment summary")
            st.bar_chart(df["Sentiment"].value_counts())

            st.write("Flagged negative comments")
            negatives = df[df["Sentiment"] == "negative"]["Comment"].tolist()
            if negatives:
                for n in negatives:
                    st.write("- " + n)
            else:
                st.write("No negative comments detected.")

        if st.session_state.quiz_results:
            st.write("Well-being Check Results")
            qdf = pd.DataFrame(st.session_state.quiz_results, columns=["Score", "Result"])
            st.table(qdf)
            st.bar_chart(qdf["Score"])
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
