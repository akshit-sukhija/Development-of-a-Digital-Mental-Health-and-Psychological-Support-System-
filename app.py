import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import re

# -------------------------
# Rule-based lexicon analyzer (no dataset required)
# -------------------------
POS_WORDS = {
    "good", "great", "happy", "joy", "love", "awesome", "fantastic", "relieved",
    "satisfied", "hopeful", "calm", "better", "improved", "positive", "grateful"
}
NEG_WORDS = {
    "sad", "depressed", "angry", "upset", "anxious", "worried", "hopeless",
    "terrible", "awful", "stress", "stressed", "pain", "hurt", "bad", "negative",
    "lonely", "suicidal", "angst", "panic"
}
NEGATIONS = {"not", "no", "never", "n't", "hardly", "rarely"}
INTENSIFIERS = {"very", "extremely", "really", "so", "too", "super"}

def tokenize(text):
    # lower, remove non-word except apostrophes, split
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()

def rule_sentiment(text):
    tokens = tokenize(text)
    score = 0.0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        weight = 1.0
        # check intensifier next word or current
        if t in INTENSIFIERS and i+1 < len(tokens):
            # apply intensifier to next token
            weight = 1.8
            i += 1
            t = tokens[i]
        # negation handling: if previous token is negation within window of 3 tokens,
        # flip sentiment of this token
        neg_window = tokens[max(0, i-3):i]
        is_negated = any(w in NEGATIONS for w in neg_window)
        if t in POS_WORDS:
            score += weight * (-1.0 if is_negated else 1.0)
        elif t in NEG_WORDS:
            score += weight * (1.0 if is_negated else -1.0)
        i += 1

    # Normalize by token count to avoid very long text bias
    denom = max(1.0, len(tokens))
    normalized = score / denom

    # thresholds tuned for human-friendly output
    if normalized > 0.05:
        label = "positive"
    elif normalized < -0.05:
        label = "negative"
    else:
        label = "neutral"
    # return label and a confidence-like value between 0..1
    confidence = min(1.0, abs(normalized) * 5.0)
    return label, round(confidence, 2)

# -------------------------
# Optional: Try load ML model if user prefers
# -------------------------
@st.cache_resource
def try_load_ml_model():
    try:
        model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        return None, None

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Digital Mental Health", layout="wide")
st.title("💬 Digital Mental Health Support")
st.markdown("A convenient sentiment analyzer for students — no dataset or technical skills required.")

# -------------------------
# Try model load
# -------------------------
ml_model, ml_vectorizer = try_load_ml_model()
if ml_model is None:
    st.info("No ML model found in `models/`. Using a built-in rule-based sentiment analyzer (no dataset needed).")
else:
    st.success("Optional ML model loaded — you can switch to it in the sidebar.")

# -------------------------
# Session State
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of (text, label, confidence, method)
if "quiz_taken" not in st.session_state:
    st.session_state.quiz_taken = False
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = None

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Student Dashboard", "MCQ Quiz", "Admin Dashboard"])

st.sidebar.title("⚙️ Sentiment Options")
method_choice = st.sidebar.radio("Prediction method:", ["Rule-based (no dataset)", "ML model (if available)"])
if method_choice == "ML model (if available)" and ml_model is None:
    st.sidebar.warning("ML model not found. Falling back to rule-based analyzer.")

# ====================================================
# STUDENT DASHBOARD
# ====================================================
if page == "Student Dashboard":
    st.header("Student Dashboard")
    st.write("Enter your comment below. The app will analyze sentiment using the chosen method.")
    user_input = st.text_area("✍️ Enter a comment here:", height=120)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Analyze Sentiment"):
            if not user_input.strip():
                st.warning("Please type a comment before analyzing.")
            else:
                if method_choice == "ML model (if available)" and ml_model is not None:
                    try:
                        X = ml_vectorizer.transform([user_input])
                        pred_raw = ml_model.predict(X)[0]
                        # try to map to our labels
                        label = str(pred_raw).lower()
                        confidence = None
                        # attempt to get probability if available
                        try:
                            probs = ml_model.predict_proba(X)[0]
                            # choose max prob
                            confidence = round(max(probs), 2)
                        except Exception:
                            confidence = None
                        method = "ml_model"
                    except Exception as e:
                        st.error("Error using ML model — falling back to rule-based analyzer.")
                        label, confidence = rule_sentiment(user_input)
                        method = "rule_based"
                else:
                    label, confidence = rule_sentiment(user_input)
                    method = "rule_based"

                # append to history
                st.session_state.history.append((user_input, label, confidence, method))

                # display
                if label == "positive":
                    st.success(f"😊 Sentiment: Positive (confidence: {confidence})")
                elif label == "negative":
                    st.error(f"😞 Sentiment: Negative (confidence: {confidence})")
                else:
                    st.info(f"😐 Sentiment: Neutral (confidence: {confidence})")

    with col2:
        st.markdown("**Quick tips for non-technical users:**")
        st.markdown("""
        - Use short, honest sentences (e.g. *"I feel anxious about exams"*).
        - The rule-based analyzer uses common positive/negative words and handles simple negations like *not*.
        - If you upload a trained ML model to `models/`, you can switch to it from the sidebar.
        """)

    st.write("---")
    st.subheader("📜 Recent Analyses")
    if st.session_state.history:
        # show last 10 analyses
        df_hist = pd.DataFrame(st.session_state.history[::-1], columns=["Comment", "Sentiment", "Confidence", "Method"]).head(10)
        st.dataframe(df_hist)
    else:
        st.write("No analyses yet. Try entering a comment above.")

    st.write("---")
    st.subheader("📊 Sentiment Distribution")
    if st.session_state.history:
        preds = [p for _, p, _, _ in st.session_state.history]
        counts = {
            "positive": preds.count("positive"),
            "negative": preds.count("negative"),
            "neutral": preds.count("neutral"),
        }
        fig, ax = plt.subplots()
        ax.pie([counts["positive"], counts["negative"], counts["neutral"]],
               labels=["positive", "negative", "neutral"], autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.write("No data yet to show distribution.")

# ====================================================
# MCQ QUIZ
# ====================================================
elif page == "MCQ Quiz":
    st.header("MCQ: Mental Health Awareness (short quiz)")
    st.write("This short quiz helps students learn common mental-health facts. No data is collected. Immediate feedback is provided.")

    # Define MCQs
    MCQS = [
        {
            "q": "Which of the following is a common sign of high stress?",
            "choices": ["Increased concentration", "Sleep disturbances", "Steady appetite", "Improved mood"],
            "correct": 1,
            "explain": "Stress often disrupts sleep patterns, causing insomnia or poor-quality sleep."
        },
        {
            "q": "What is a recommended initial step if someone is feeling very low or suicidal?",
            "choices": ["Ignore feelings and hope they pass", "Share feelings with a trusted person or helpline", "Self-medicate with alcohol", "Post about it publicly for attention"],
            "correct": 1,
            "explain": "Talking to a trusted person or a professional/helpline is a safe first step."
        },
        {
            "q": "Which activity is usually helpful for short-term anxiety relief?",
            "choices": ["Deep breathing exercises", "Complete social withdrawal", "Avoiding sleep", "Isolating from loved ones"],
            "correct": 0,
            "explain": "Deep breathing helps calm the nervous system and reduce acute anxiety."
        },
        {
            "q": "Persistent sadness for more than two weeks may indicate:",
            "choices": ["A passing mood", "Clinical depression requiring attention", "Always normal, no action", "Only physical illness"],
            "correct": 1,
            "explain": "While not always, persistent low mood for weeks can be a sign of depression and should be checked."
        },
        {
            "q": "Which is an example of a healthy coping strategy?",
            "choices": ["Talking with a friend", "Binge drinking", "Ignoring the problem", "Self-harm"],
            "correct": 0,
            "explain": "Reaching out to supportive people is a healthy way to cope."
        }
    ]

    # Render quiz
    answers = []
    for i, item in enumerate(MCQS):
        st.markdown(f"**Q{i+1}. {item['q']}**")
        choice = st.radio(f"Select one (Q{i+1})", item["choices"], key=f"q{i}")
        answers.append(item["choices"].index(choice))

    if st.button("Submit Quiz"):
        score = 0
        feedback = []
        for i, item in enumerate(MCQS):
            correct = item["correct"]
            if answers[i] == correct:
                score += 1
                feedback.append((True, item["explain"]))
            else:
                feedback.append((False, item["explain"]))
        st.session_state.quiz_taken = True
        st.session_state.quiz_score = score
        st.success(f"You scored {score} out of {len(MCQS)}")
        st.write("Feedback:")
        for i, (is_correct, expl) in enumerate(feedback):
            if is_correct:
                st.write(f"Q{i+1}: ✅ Correct — {expl}")
            else:
                st.write(f"Q{i+1}: ❌ Incorrect — {expl}")

    if st.session_state.quiz_taken:
        st.info(f"Last quiz score: {st.session_state.quiz_score} / {len(MCQS)}")

# ====================================================
# ADMIN DASHBOARD
# ====================================================
elif page == "Admin Dashboard":
    st.header("🛠️ Admin Dashboard - Mental Health Insights")
    if not st.session_state.history:
        st.warning("No student feedback available yet.")
    else:
        df = pd.DataFrame(st.session_state.history, columns=["Comment", "Sentiment", "Confidence", "Method"])
        st.subheader("📋 Student Comments Data")
        st.dataframe(df)

        st.subheader("📊 Overall Sentiment Summary")
        st.bar_chart(df["Sentiment"].value_counts())

        st.subheader("🚩 Flagged Negative Comments")
        negatives = df[df["Sentiment"] == "negative"]["Comment"].tolist()
        if negatives:
            for c in negatives:
                st.write(f"- {c}")
        else:
            st.write("✅ No negative comments detected yet.")

# Footer / small note
st.markdown("---")
st.caption("This app uses a small built-in lexicon analyzer by default so it can run without uploading any datasets. If you add a trained model in `models/` named `sentiment_model.pkl` + `vectorizer.pkl`, you can toggle to use it.")
