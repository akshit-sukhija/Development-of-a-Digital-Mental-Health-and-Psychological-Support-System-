import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import re

# -------------------------
# Rule-based analyzer
# -------------------------
POS_WORDS = {"good","great","happy","joy","love","awesome","fantastic","relieved","satisfied","hopeful","calm","better","improved","positive","grateful"}
NEG_WORDS = {"sad","depressed","angry","upset","anxious","worried","hopeless","terrible","awful","stress","stressed","pain","hurt","bad","negative","lonely","suicidal","angst","panic"}
NEGATIONS = {"not","no","never","n't","hardly","rarely"}
INTENSIFIERS = {"very","extremely","really","so","too","super"}

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()

def rule_sentiment(text):
    tokens = tokenize(text)
    score, i = 0.0, 0
    while i < len(tokens):
        t, weight = tokens[i], 1.0
        if t in INTENSIFIERS and i + 1 < len(tokens):
            weight, i, t = 1.8, i + 1, tokens[i + 1]
        neg_window = tokens[max(0, i - 3):i]
        is_negated = any(w in NEGATIONS for w in neg_window)
        if t in POS_WORDS: score += weight * (-1.0 if is_negated else 1.0)
        elif t in NEG_WORDS: score += weight * (1.0 if is_negated else -1.0)
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
        model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None

ml_model, ml_vectorizer = load_model()

# -------------------------
# Streamlit Setup
# -------------------------
st.set_page_config(page_title="Digital Mental Health", layout="wide")
st.title("💬 Digital Mental Health Support")
st.markdown("Analyze feedback, track emotions, and support well-being.")

# -------------------------
# Session State
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # (comment, sentiment, confidence, method)
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []  # (total_score, message)

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Student Dashboard", "MCQ Quiz", "Admin Dashboard"])

# -------------------------
# Sidebar Options
# -------------------------
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
                    X = ml_vectorizer.transform([user_input])
                    pred_raw = ml_model.predict(X)[0]
                    label = str(pred_raw).lower()
                    try:
                        probs = ml_model.predict_proba(X)[0]
                        confidence = round(max(probs), 2)
                    except Exception:
                        confidence = None
                    method = "ml_model"
                except Exception:
                    st.error("Error using ML model — falling back to rule-based analyzer.")
                    label, confidence = rule_sentiment(user_input)
                    method = "rule_based"
            else:
                label, confidence = rule_sentiment(user_input)
                method = "rule_based"

            st.session_state.history.append((user_input, label, confidence, method))

            # Show result
            if label == "positive":
                st.success(f"😊 Sentiment: Positive (confidence: {confidence})")
                st.write("✅ Keep it up! Share your joy, keep journaling positives.")
            elif label == "negative":
                st.error(f"😞 Sentiment: Negative (confidence: {confidence})")
                st.write("💡 Try breathing exercises, short walks, or talking to a friend.")
            else:
                st.info(f"😐 Sentiment: Neutral (confidence: {confidence})")
                st.write("😐 Neutral mood. Boost it with music, stretching, or hydration.")

    st.write("---")
    st.subheader("📜 Recent Analyses")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history[::-1], columns=["Comment","Sentiment","Confidence","Method"]).head(10)
        st.dataframe(df_hist)
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
                st.pyplot(fig)
            except Exception:
                st.bar_chart(pd.DataFrame({"count": vals}, index=counts.keys()))
    else:
        st.write("No data yet to show distribution.")

# ====================================================
# MCQ QUIZ (Well-being Check)
# ====================================================
elif page == "MCQ Quiz":
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

        # Save quiz result to session
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
            st.dataframe(df)

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
            st.dataframe(quiz_df)
            st.bar_chart(quiz_df["Score"])

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Runs with built-in analyzer (no dataset needed). If you add a trained ML model to `models/`, you can switch to it.")
