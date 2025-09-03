import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

# -------------------------
# Rule-based analyzer
# -------------------------
POS_WORDS = {"good","great","happy","joy","love","awesome","fantastic","relieved","satisfied","hopeful","calm","better","improved","positive","grateful"}
NEG_WORDS = {"sad","depressed","angry","upset","anxious","worried","hopeless","terrible","awful","stress","stressed","pain","hurt","bad","negative","lonely","suicidal","angst","panic"}
NEGATIONS = {"not","no","never","n't","hardly","rarely"}
INTENSIFIERS = {"very","extremely","really","so","too","super"}

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()

def rule_sentiment(text: str):
    tokens = tokenize(text)
    score, i = 0.0, 0
    while i < len(tokens):
        t, weight = tokens[i], 1.0
        if t in INTENSIFIERS and i + 1 < len(tokens):
            weight, i, t = 1.8, i + 1, tokens[i + 1]
        neg_window = tokens[max(0, i - 3):i]
        is_negated = any(w in NEGATIONS for w in neg_window)
        if t in POS_WORDS:
            score += weight * (-1.0 if is_negated else 1.0)
        elif t in NEG_WORDS:
            score += weight * (1.0 if is_negated else -1.0)
        i += 1
    normalized = score / max(1.0, len(tokens))
    if normalized > 0.05: label = "positive"
    elif normalized < -0.05: label = "negative"
    else: label = "neutral"
    return label, round(min(1.0, abs(normalized) * 5.0), 2)

# -------------------------
# ML model loader
# -------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None

ml_model, ml_vectorizer = load_model()

# -------------------------
# Robust ML prediction helpers
# -------------------------
def _string_guess(label_raw: str) -> str | None:
    s = str(label_raw).strip().lower()
    if any(k in s for k in ["pos", "positive", "+"]):   return "positive"
    if any(k in s for k in ["neg", "negative", "-"]):   return "negative"
    if "neu" in s or "neutral" in s:                    return "neutral"
    return None

@st.cache_resource
def infer_label_mapping(model, vectorizer):
    """
    Use probe sentences to infer how the model encodes labels.
    Returns dict: {raw_label_from_model: 'positive'/'negative'/'neutral'}
    Works for numeric or custom string labels; supports binary and 3-class.
    """
    try:
        probes = {
            "positive": "I am very happy and I love this! absolutely fantastic",
            "negative": "I am very sad and hate this. absolutely terrible",
            "neutral":  "It is okay, nothing special, just average"
        }
        mapping = {}
        for canon, text in probes.items():
            Xp = vectorizer.transform([text])
            raw = model.predict(Xp)[0]
            mapping[raw] = canon
        # If binary, probes for neutral may collide with pos/neg; that's fine.
        return mapping
    except Exception:
        return {}

def predict_with_model(text: str):
    """
    Unified ML prediction:
    - Uses mapping inferred from probes, or guesses from label strings.
    - For binary models, applies a neutral band when confidence is low.
    Returns (label, confidence)
    """
    if ml_model is None or ml_vectorizer is None:
        raise RuntimeError("Model not loaded")

    X = ml_vectorizer.transform([text])
    raw = ml_model.predict(X)[0]

    # Try mapping inferred from probes
    mapping = infer_label_mapping(ml_model, ml_vectorizer)
    label = mapping.get(raw)

    # If still unknown, try string guess (handles 'POS','NEG','NEU')
    if label is None:
        label = _string_guess(raw)

    # If still unknown and numeric classes, try using class index + probabilities
    try:
        classes = getattr(ml_model, "classes_", None)
        probs = ml_model.predict_proba(X)[0] if hasattr(ml_model, "predict_proba") else None
    except Exception:
        classes, probs = None, None

    if label is None and classes is not None:
        # If classes are already textual, map them by string guess
        if isinstance(classes[0], str):
            idx = int(np.argmax(probs)) if probs is not None else 0
            label = _string_guess(classes[idx])
        else:
            # Numeric classes: use probe mapping if available; otherwise heuristic:
            # assume larger index is "more positive".
            idx = int(np.argmax(probs)) if probs is not None else 0
            # Sort unique classes to get stable order
            ordered = list(sorted(classes))
            if len(ordered) == 3:
                # map lowest->negative, middle->neutral, highest->positive
                if classes[idx] == ordered[2]: label = "positive"
                elif classes[idx] == ordered[0]: label = "negative"
                else: label = "neutral"
            elif len(ordered) == 2:
                # binary: highest index -> positive (heuristic)
                label = "positive" if classes[idx] == ordered[1] else "negative"

    # Final fallback: if still none, use rule-based
    if label is None:
        return rule_sentiment(text)

    # Confidence
    conf = None
    if probs is not None:
        conf = float(np.max(probs))
        # For binary models, introduce neutral band for low confidence
        if len(classes) == 2 and conf < 0.60:
            label = "neutral"

    return label, (None if conf is None else round(conf, 2))

# -------------------------
# Streamlit Setup
# -------------------------
st.set_page_config(page_title="Digital Mental Health", layout="wide")
st.title("💬 Digital Mental Health Support")
st.markdown("Analyze feedback, track emotions, and support well-being. If an ML model is in `models/`, it will be used automatically.")

# -------------------------
# Session State
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # (comment, sentiment, confidence, method)
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []  # (total_score, message)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Student Dashboard", "Well-Being Check", "Admin Dashboard"])

st.sidebar.title("⚙️ Sentiment Options")
method_choice = st.sidebar.radio("Prediction method:", ["Rule-based (no dataset)", "ML model (if available)"])
if method_choice == "ML model (if available)" and ml_model is None:
    st.sidebar.warning("ML model not found. Falling back to rule-based analyzer.")

# ====================================================
# STUDENT DASHBOARD
# ====================================================
if page == "Student Dashboard":
    st.header("Student Dashboard")
    user_input = st.text_area("✍️ Enter a comment here:", height=120)

    if st.button("Analyze Sentiment"):
        if not user_input.strip():
            st.warning("Please type a comment before analyzing.")
        else:
            if method_choice == "ML model (if available)" and ml_model is not None:
                try:
                    label, confidence = predict_with_model(user_input)
                    method = "ml_model"
                except Exception:
                    st.error("Error using ML model — falling back to rule-based analyzer.")
                    label, confidence = rule_sentiment(user_input)
                    method = "rule_based"
            else:
                label, confidence = rule_sentiment(user_input)
                method = "rule_based"

            st.session_state.history.append((user_input, label, confidence, method))

            # Show result + tips
            if label == "positive":
                st.success(f"😊 Sentiment: Positive (confidence: {confidence})")
                st.write("✅ Keep it up! Share your joy, keep journaling positives.")
            elif label == "negative":
                st.error(f"😞 Sentiment: Negative (confidence: {confidence})")
                st.write("💡 Try breathing exercises, short walks, or talking to a friend.")
            else:
                st.info(f"😐 Sentiment: Neutral (confidence: {confidence})")
                st.write("🎧 Play calming music, stretch, or hydrate.")

    st.write("---")
    st.subheader("📜 Recent Analyses")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history[::-1], columns=["Comment","Sentiment","Confidence","Method"]).head(10)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.write("No analyses yet.")

    st.write("---")
    st.subheader("📊 Sentiment Distribution")
    if st.session_state.history:
        preds = [p for _, p, _, _ in st.session_state.history]
        counts = {"positive": preds.count("positive"), "negative": preds.count("negative"), "neutral": preds.count("neutral")}
        vals = [counts["positive"], counts["negative"], counts["neutral"]]
        if sum(vals) == 0:
            st.info("No sentiment data yet.")
        else:
            try:
                fig, ax = plt.subplots()
                ax.pie(vals, labels=counts.keys(), autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4))
                ax.axis("equal")
                st.pyplot(fig)
            except Exception:
                st.bar_chart(pd.DataFrame({"count": vals}, index=counts.keys()))
    else:
        st.write("No data yet to show distribution.")

# ====================================================
# WELL-BEING CHECK (MCQ)
# ====================================================
elif page == "Well-Being Check":
    st.header("🌱 Well-being Check")
    st.write("Answer 5 quick questions about your well-being.")

    QUESTIONS = [
        {"q":"How is your mood today?","options":{"😊 Happy/positive":4,"😐 Neutral/okay":2,"😞 Sad/low":1}},
        {"q":"How much energy do you have right now?","options":{"⚡ Full of energy":4,"🙂 Somewhat energetic":3,"😴 Tired/exhausted":1}},
        {"q":"How stressed do you feel?","options":{"😌 Calm/relaxed":4,"😕 A bit stressed":2,"😫 Very stressed":1}},
        {"q":"How was your sleep recently?","options":{"🌙 Slept well":4,"😶 Average sleep":2,"🥱 Poor sleep":1}},
        {"q":"How connected do you feel socially?","options":{"👥 Very connected":4,"🙂 Somewhat connected":2,"😔 Lonely/isolated":1}}
    ]

    responses = []
    for i,q in enumerate(QUESTIONS):
        st.markdown(f"**{q['q']}**")
        choice = st.radio("", list(q["options"].keys()), key=f"quiz_q{i}")
        responses.append(q["options"][choice])

    if st.button("✨ Submit Well-being Check", use_container_width=True):
        total = sum(responses)
        if total >= 17:
            msg = "🌟 You seem happy and doing well! Keep up your positive habits."
            st.success(msg)
        elif 12 <= total < 17:
            msg = "🙂 You seem okay, but could use some self-care."
            st.info(msg)
        else:
            msg = "💡 You may be stressed or feeling low. Try relaxation techniques or seek support."
            st.warning(msg)

        st.session_state.quiz_results.append((total, msg))

        st.write("---")
        st.write("**Note:** This is not a diagnosis, just a self-check. For persistent issues, seek professional help.")

# ====================================================
# ADMIN DASHBOARD
# ====================================================
elif page == "Admin Dashboard":
    st.header("🛠️ Admin Dashboard")

    if not st.session_state.history and not st.session_state.quiz_results:
        st.warning("No student feedback or quiz results available yet.")
    else:
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history, columns=["Comment","Sentiment","Confidence","Method"])
            st.subheader("📋 Student Comments")
            st.dataframe(df, use_container_width=True)

            st.subheader("📊 Sentiment Summary")
            st.bar_chart(df["Sentiment"].value_counts())

            st.subheader("🚩 Flagged Negative Comments")
            negatives = df[df["Sentiment"]=="negative"]["Comment"].tolist()
            if negatives:
                for c in negatives: st.write(f"- {c}")
            else:
                st.write("✅ No negative comments detected.")

        if st.session_state.quiz_results:
            quiz_df = pd.DataFrame(st.session_state.quiz_results, columns=["Score","Result"])
            st.subheader("📝 Well-being Quiz Results")
            st.dataframe(quiz_df, use_container_width=True)
            st.bar_chart(quiz_df["Score"])

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built-in analyzer requires no dataset. If you add a trained ML model to `models/`, the app auto-detects it.")

