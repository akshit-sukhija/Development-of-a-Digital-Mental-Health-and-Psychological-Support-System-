# app.py
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional, Dict, Any

# -------------------------
# RULE-BASED SENTIMENT (fallback)
# -------------------------
POS_WORDS = {
    "good","great","happy","joy","love","awesome","fantastic","relieved",
    "satisfied","hopeful","calm","better","improved","positive","grateful",
    "amazing","pleased","excited"
}
NEG_WORDS = {
    "sad","depressed","angry","upset","anxious","worried","hopeless",
    "terrible","awful","stress","stressed","pain","hurt","bad","negative",
    "lonely","suicidal","panic","hate","annoyed","frustrated"
}
NEGATIONS = {"not","no","never","n't","hardly","rarely"}
INTENSIFIERS = {"very","extremely","really","so","too","super"}

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
# ML MODEL LOADER & PREDICTION HELPERS
# -------------------------
@st.cache_resource
def load_ml_model() -> Tuple[Optional[Any], Optional[Any]]:
    """Try to load model and vectorizer; return (model, vectorizer) or (None, None)."""
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None

ml_model, ml_vectorizer = load_ml_model()

def guess_label_from_string(raw: Any) -> Optional[str]:
    """Try to interpret textual labels like 'POS','positive','neg'."""
    try:
        s = str(raw).lower()
    except Exception:
        return None
    if any(k in s for k in ["pos", "positive", "+", "good", "happy"]):
        return "positive"
    if any(k in s for k in ["neg", "negative", "-", "sad", "bad", "angry"]):
        return "negative"
    if "neu" in s or "neutral" in s or "0" == s:
        return "neutral"
    return None

@st.cache_resource
def infer_mapping_using_probes(model, vectorizer) -> Dict[Any, str]:
    """
    Probe the model with three clearly positive/negative/neutral sentences
    and map raw model outputs to canonical labels.
    Returns mapping raw_value -> 'positive'|'negative'|'neutral'.
    """
    mapping = {}
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
        # keep mapping empty if anything fails
        return {}
    return mapping

def predict_ml_text(text: str) -> Tuple[str, Optional[float]]:
    """
    Predict with the ML model and return canonical label + confidence.
    If problems occur, raise Exception to let caller fallback to rule-based.
    """
    if ml_model is None or ml_vectorizer is None:
        raise RuntimeError("ML model not loaded")
    # transform
    X = ml_vectorizer.transform([text])
    # predict raw
    raw_pred = ml_model.predict(X)[0]
    # try mapping from probes
    mapping = infer_mapping_using_probes(ml_model, ml_vectorizer)
    label = mapping.get(raw_pred)
    # try guessing from raw representation
    if label is None:
        label = guess_label_from_string(raw_pred)
    # handle classes_ and probabilities if available
    probs = None
    classes = getattr(ml_model, "classes_", None)
    try:
        if hasattr(ml_model, "predict_proba"):
            probs = ml_model.predict_proba(X)[0]
    except Exception:
        probs = None
    # If still unknown, use classes and probs heuristics
    if label is None and classes is not None:
        # if textual classes, try string guess on the class with highest prob
        if probs is not None:
            best_idx = int(np.argmax(probs))
            candidate = classes[best_idx]
            label = guess_label_from_string(candidate)
        else:
            # no probs: try to map classes to ordered numeric heuristic
            # example: classes = [0,1] or [-1,0,1] or [0,2]
            try:
                ordered = list(sorted(classes))
                # if 3 classes, map min->negative, mid->neutral, max->positive
                if len(ordered) == 3:
                    raw_selected = ml_model.predict(X)[0]
                    if raw_selected == ordered[0]:
                        label = "negative"
                    elif raw_selected == ordered[1]:
                        label = "neutral"
                    else:
                        label = "positive"
                elif len(ordered) == 2:
                    # binary: assume lower -> negative, higher -> positive
                    raw_selected = ml_model.predict(X)[0]
                    label = "positive" if raw_selected == ordered[1] else "negative"
                else:
                    # fallback
                    label = None
            except Exception:
                label = None

    # if still unknown, raise to let fallback happen
    if label is None:
        raise RuntimeError("Unable to map model output to canonical label")

    # confidence calculation
    conf = None
    if probs is not None:
        conf = float(np.max(probs))
        # For binary models, if confidence low => neutral
        if len(probs) == 2 and conf < 0.60:
            label = "neutral"
    elif hasattr(ml_model, "score"):
        # best-effort: try predict_proba failed but we can set None
        conf = None

    return label, (None if conf is None else round(conf, 2))

# -------------------------
# STREAMLIT UI + LOGIC
# -------------------------
st.set_page_config(page_title="Digital Mental Health", layout="wide")
st.markdown("<h1 style='text-align:center'>💬 Digital Mental Health Support</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Sentiment analyzer + well-being self-check. ML model optional (place files in <code>models/</code>).</p>", unsafe_allow_html=True)

# simple styling for a clean look
st.markdown("""
<style>
.block { background: white; padding:18px; border-radius:10px; box-shadow:0 3px 10px rgba(0,0,0,0.06); }
.small { color: #666; font-size:14px; }
.center { max-width:880px; margin:auto; }
</style>
""", unsafe_allow_html=True)

# initialize session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of tuples (text,label,confidence,method)
if "quiz" not in st.session_state:
    st.session_state.quiz = []     # list of tuples (score,message)

# Sidebar navigation (clean)
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Student Dashboard", "Well-Being Check", "Admin Dashboard"])
    st.markdown("---")
    st.title("Sentiment Options")
    method_choice = st.radio("Prediction method:", ["Rule-based (no dataset)", "ML model (if available)"])
    if method_choice == "ML model (if available)":
        if ml_model is None:
            st.warning("ML model not found. Falling back to rule-based analyzer.")
        else:
            st.success("ML model loaded — predictions will use the model.")

# -------------------------
# STUDENT DASHBOARD
# -------------------------
if page == "Student Dashboard":
    st.markdown("<div class='center block'>", unsafe_allow_html=True)
    st.subheader("Sentiment Analyzer")
    user_input = st.text_area("Type your thought or feedback here:", height=120)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        analyze = st.button("Analyze", use_container_width=True)
    with col2:
        clear_recent = st.button("Clear Recent Analyses", use_container_width=True)
    with col3:
        st.write("")  # placeholder for layout

    if clear_recent:
        st.session_state.history = []
        st.success("Recent analyses cleared.")

    if analyze:
        if not user_input.strip():
            st.warning("Please enter a comment to analyze.")
        else:
            # try ML if chosen
            if method_choice == "ML model (if available)" and ml_model is not None and ml_vectorizer is not None:
                try:
                    label, confidence = predict_ml_text(user_input)
                    method = "ml_model"
                except Exception:
                    # fall back silently to rule-based
                    label, confidence = rule_sentiment(user_input)
                    method = "rule_based"
            else:
                label, confidence = rule_sentiment(user_input)
                method = "rule_based"

            st.session_state.history.append((user_input, label, confidence, method))

            # display result and short tips
            if label == "positive":
                st.success(f"😊 Positive — confidence: {confidence}")
                st.markdown("- Keep journaling positives\n- Share with a friend")
            elif label == "negative":
                st.error(f"😞 Negative — confidence: {confidence}")
                st.markdown("- Try 2 minutes deep breathing\n- Take a short walk or talk to someone")
            else:
                st.info(f"😐 Neutral — confidence: {confidence}")
                st.markdown("- Try a small activity: stretch, hydrate, listen to music")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Recent Analyses")
    if st.session_state.history:
        # show last 10 in a neat table
        df_hist = pd.DataFrame(st.session_state.history[::-1], columns=["Comment","Sentiment","Confidence","Method"]).head(10)
        st.table(df_hist)
    else:
        st.info("No analyses yet. Enter a comment and press Analyze.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# WELL-BEING CHECK (MCQ)
# -------------------------
elif page == "Well-Being Check":
    st.markdown("<div class='center block'>", unsafe_allow_html=True)
    st.subheader("🌱 Quick Well-being Check")
    st.markdown("<p class='small'>5 short questions — takes under 1 minute. This is a self-check, not a diagnosis.</p>", unsafe_allow_html=True)

    QUESTIONS = [
        {"q":"How is your mood today?","options":{"😊 Happy/positive":4,"😐 Neutral/okay":2,"😞 Sad/low":1}},
        {"q":"How much energy do you have right now?","options":{"⚡ Full of energy":4,"🙂 Somewhat energetic":3,"😴 Tired/exhausted":1}},
        {"q":"How stressed do you feel?","options":{"😌 Calm/relaxed":4,"😕 A bit stressed":2,"😫 Very stressed":1}},
        {"q":"How was your sleep recently?","options":{"🌙 Slept well":4,"😶 Average sleep":2,"🥱 Poor sleep":1}},
        {"q":"How connected do you feel socially?","options":{"👥 Very connected":4,"🙂 Somewhat connected":2,"😔 Lonely/isolated":1}}
    ]

    responses = []
    for i,q in enumerate(QUESTIONS):
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
            msg = "💡 You may be stressed or feeling low. Consider relaxation techniques or seek support."
            st.warning(msg)
        st.session_state.quiz.append((total, msg))
        st.markdown("<p class='small'>Note: This is not a clinical diagnosis.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# ADMIN DASHBOARD
# -------------------------
elif page == "Admin Dashboard":
    st.markdown("<div class='center block'>", unsafe_allow_html=True)
    st.subheader("Admin Dashboard")
    if not st.session_state.history and not st.session_state.quiz:
        st.info("No student feedback or quiz results yet.")
    else:
        if st.session_state.history:
            st.write("Student Comments")
            df = pd.DataFrame(st.session_state.history, columns=["Comment","Sentiment","Confidence","Method"])
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
        if st.session_state.quiz:
            st.write("Well-being Check Results")
            qdf = pd.DataFrame(st.session_state.quiz, columns=["Score","Result"])
            st.table(qdf)
            st.bar_chart(qdf["Score"])
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Clean & simple UI — ready for SIH demo. ML model optional: drop 'sentiment_model.pkl' and 'vectorizer.pkl' into a 'models/' folder.")

