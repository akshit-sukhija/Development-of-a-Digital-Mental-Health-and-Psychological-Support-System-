# app.py
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import re
from typing import List, Dict

# -------------------------
# Streamlit Setup (must be first Streamlit call)
# -------------------------
st.set_page_config(page_title="Digital Mental Health", layout="wide")

# -------------------------
# Global Styles (Centered Card + Soft Colors)
# -------------------------
st.markdown(
    """
    <style>
      .soft-bg { background: #f7f9fc; padding: 24px 0; }
      .centered-card { max-width: 760px; margin: 0 auto; padding: 0 12px; }
      .card {
          background: #ffffff;
          border-radius: 16px;
          box-shadow: 0 6px 22px rgba(0,0,0,0.08);
          padding: 24px;
          border: 1px solid #eef1f6;
      }
      .card h3 { margin-top: 0; }
      .stButton > button {
          background: #4f8bf9 !important;
          color: #ffffff !important;
          border: none !important;
          border-radius: 10px !important;
          padding: 0.6rem 1rem !important;
      }
      .stButton > button:hover {
          background: #3c74e6 !important;
      }
      .subtle { color: #667085; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()

def rule_sentiment(text: str):
    tokens = tokenize(text)
    score = 0.0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        weight = 1.0
        # Handle intensifiers
        if t in INTENSIFIERS and i + 1 < len(tokens):
            weight = 1.8
            i += 1
            t = tokens[i]
        # Windowed negation handling
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
    confidence = min(1.0, abs(normalized) * 5.0)
    return label, round(confidence, 2)

# -------------------------
# Optional: Load ML model (cached resource)
# -------------------------
@st.cache_resource
def try_load_ml_model():
    try:
        model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None

# -------------------------
# App Header
# -------------------------
st.title("💬 Digital Mental Health Support")
st.markdown("A convenient sentiment analyzer and well‑being self‑check for students — no dataset or technical skills required. Optional ML model can be added to 'models/'.", help="Add your own trained model to models/sentiment_model.pkl and models/vectorizer.pkl")

ml_model, ml_vectorizer = try_load_ml_model()
if ml_model is None:
    st.info("No ML model found. Using built-in rule-based sentiment analyzer.")
else:
    st.success("Optional ML model loaded — switch to it in the sidebar.")

# -------------------------
# Session State
# -------------------------
if "history" not in st.session_state:
    # sentiment history: list of tuples (comment, label, confidence, method)
    st.session_state.history = []
if "wellbeing_history" not in st.session_state:
    # well-being results: list of dicts {timestamp, total_score, category, answers:[{question, choice, score}]}
    st.session_state.wellbeing_history = []
if "quiz_taken" not in st.session_state:
    st.session_state.quiz_taken = False
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = None

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Student Dashboard", "Well‑Being Check", "Admin Dashboard"])

st.sidebar.title("⚙️ Sentiment Options")
method_choice = st.sidebar.radio("Prediction method:", ["Rule-based (no dataset)", "ML model (if available)"])
if method_choice == "ML model (if available)" and ml_model is None:
    st.sidebar.warning("ML model not found. Falling back to rule-based analyzer.")

# ====================================================
# STUDENT DASHBOARD
# ====================================================
if page == "Student Dashboard":
    st.header("Student Dashboard")
    st.markdown('<div class="soft-bg"><div class="centered-card"><div class="card">', unsafe_allow_html=True)
    st.markdown("### Comment Sentiment")
    user_input = st.text_area("✍️ Enter a comment here:", height=120, placeholder="e.g., I feel anxious about exams but hopeful after talking to a friend.")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Analyze Sentiment"):
            if not user_input.strip():
                st.warning("Please type a comment before analyzing.")
            else:
                # Try ML, else rule-based
                if method_choice == "ML model (if available)" and ml_model is not None:
                    try:
                        X = ml_vectorizer.transform([user_input])
                        pred_raw = ml_model.predict(X)
                        label = str(pred_raw).lower()
                        try:
                            probs = ml_model.predict_proba(X)
                            confidence = round(float(max(probs)), 2)
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
                if label == "positive":
                    st.success(f"😊 Sentiment: Positive (confidence: {confidence})")
                elif label == "negative":
                    st.error(f"😞 Sentiment: Negative (confidence: {confidence})")
                else:
                    st.info(f"😐 Sentiment: Neutral (confidence: {confidence})")

    with col2:
        st.markdown("**Quick tips for non-technical users:**")
        st.markdown(
            """
            - Use short, honest sentences (e.g. *"I feel anxious about exams"*).
            - The analyzer detects positive/negative words and simple negations.
            - If you upload an ML model to `models/`, you can switch to it from the sidebar.
            """
        )
    st.markdown("</div></div></div>", unsafe_allow_html=True)

    st.write("---")
    st.subheader("📜 Recent Analyses")
    if st.session_state.history:
        df_hist = pd.DataFrame(
            st.session_state.history[::-1],
            columns=["Comment", "Sentiment", "Confidence", "Method"]
        ).head(10)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.write("No analyses yet.")

    st.write("---")
    st.subheader("📊 Sentiment Distribution")
    if st.session_state.history:
        preds = [p for _, p, _, _ in st.session_state.history]
        counts = {
            "positive": preds.count("positive"),
            "negative": preds.count("negative"),
            "neutral": preds.count("neutral")
        }
        keys = ["positive", "negative", "neutral"]
        vals = [max(0, int(counts.get(k, 0))) for k in keys]
        total = sum(vals)
        if total == 0:
            st.info("No sentiment data yet. Enter a comment and analyze it to populate the chart.")
        else:
            try:
                fig, ax = plt.subplots()
                ax.pie(vals, labels=keys, autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig, use_container_width=True)
            except Exception:
                st.warning("Pie chart failed — showing bar chart instead.")
                st.bar_chart(pd.DataFrame({"count": vals}, index=keys))
    else:
        st.write("No data yet to show distribution.")

    st.write("---")
    st.subheader("🧾 Your Well‑Being Results (Recent)")
    if st.session_state.wellbeing_history:
        wb_df = pd.DataFrame(
            [
                {
                    "Timestamp": r["timestamp"],
                    "Category": r["category"],
                    "Total Score": r["total_score"]
                }
                for r in st.session_state.wellbeing_history[::-1]
            ]
        ).head(10)
        st.dataframe(wb_df, use_container_width=True)
    else:
        st.write("No well‑being assessments submitted yet.")

# ====================================================
# WELL‑BEING CHECK (Self-Assessment)
# ====================================================
elif page == "Well‑Being Check":
    st.header("🧠 Well‑Being Check")
    st.markdown('<div class="soft-bg"><div class="centered-card"><div class="card">', unsafe_allow_html=True)
    st.markdown("### How have things been lately?")
    st.markdown("<p class='subtle'>Answer 5 quick questions about mood, energy, stress, sleep, and social connection. Your responses are scored to help reflect your current well‑being.</p>", unsafe_allow_html=True)

    WELLBEING_QUESTIONS: List[Dict] = [
        {
            "q": "Overall mood this week?",
            "options": [("Happy / content", 0), ("Mostly okay", 1), ("Low / sad", 2), ("Very down", 3)]
        },
        {
            "q": "Energy levels?",
            "options": [("Energetic", 0), ("Normal", 1), ("Low", 2), ("Exhausted", 3)]
        },
        {
            "q": "Stress level?",
            "options": [("Calm", 0), ("Manageable", 1), ("Stressed", 2), ("Overwhelmed", 3)]
        },
        {
            "q": "Sleep quality?",
            "options": [("Restful", 0), ("Okay", 1), ("Poor", 2), ("Very poor", 3)]
        },
        {
            "q": "Social connection?",
            "options": [("Connected", 0), ("Somewhat connected", 1), ("Isolated", 2), ("Very isolated", 3)]
        },
    ]

    # Build form for radios + submit button
    with st.form("wellbeing_form"):
        selections = []
        for i, item in enumerate(WELLBEING_QUESTIONS):
            labels = [opt for opt in item["options"]]
            # Default to first choice for simplicity and to avoid None/validation issues
            choice = st.radio(item["q"], labels, index=0, key=f"wb_q_{i}")
            selections.append(choice)

        submitted = st.form_submit_button("Submit Assessment")

    if submitted:
        # Map labels -> scores
        label_to_score_maps = []
        for item in WELLBEING_QUESTIONS:
            label_to_score_maps.append({label: score for (label, score) in item["options"]})

        total_score = 0
        detailed_answers = []
        for i, sel in enumerate(selections):
            score = label_to_score_maps[i][sel]
            total_score += score
            detailed_answers.append({
                "question": WELLBEING_QUESTIONS[i]["q"],
                "choice": sel,
                "score": score
            })

        # Classification: lower is better (0..15)
        if total_score <= 4:
            category = "You seem happy"
            st.success("You seem happy — your responses suggest generally positive well‑being. Keep up supportive routines and connections.")
        elif total_score <= 9:
            category = "You might be feeling sad"
            st.info("You might be feeling sad — consider gentle self‑care, talking with someone you trust, or reflecting on small steps to feel better.")
        else:
            category = "You may be stressed"
            st.warning("You may be stressed — it could help to slow down, practice calming techniques, or reach out for support if this persists.")

        # Store in session history for admin review
        st.session_state.wellbeing_history.append({
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "total_score": int(total_score),
            "category": category,
            "answers": detailed_answers
        })

        # Show summary
        st.write("---")
        st.subheader("Your Result")
        st.write(f"Total Score: {total_score} (0 = most positive, 15 = most distressed)")
        st.write("Breakdown:")
        for ans in detailed_answers:
            st.write(f"- {ans['question']} — {ans['choice']} (score {ans['score']})")

    st.markdown("</div></div></div>", unsafe_allow_html=True)

# ====================================================
# ADMIN DASHBOARD
# ====================================================
elif page == "Admin Dashboard":
    st.header("🛠️ Admin Dashboard")

    # Student comments (Sentiment)
    st.subheader("📋 Student Comments")
    if not st.session_state.history:
        st.warning("No student feedback available yet.")
    else:
        df = pd.DataFrame(st.session_state.history, columns=["Comment", "Sentiment", "Confidence", "Method"])
        st.dataframe(df, use_container_width=True)

        st.subheader("📊 Overall Sentiment Summary")
        st.bar_chart(df["Sentiment"].value_counts())

        st.subheader("🚩 Flagged Negative Comments")
        negatives = df[df["Sentiment"] == "negative"]["Comment"].tolist()
        if negatives:
            for c in negatives:
                st.write(f"- {c}")
        else:
            st.write("✅ No negative comments detected.")

    st.write("---")
    # Well-being aggregation
    st.subheader("🧠 Well‑Being Summary")
    if not st.session_state.wellbeing_history:
        st.info("No well‑being assessments submitted yet.")
    else:
        wb_df = pd.DataFrame(
            [
                {
                    "Timestamp": r["timestamp"],
                    "Category": r["category"],
                    "Total Score": r["total_score"],
                }
                for r in st.session_state.wellbeing_history
            ]
        )

        # Show basic metrics
        colA, colB = st.columns(2)
        with colA:
            st.metric("Assessments Collected", len(wb_df))
        with colB:
            st.metric("Average Total Score", round(float(wb_df["Total Score"].mean()), 2))

        # Distribution by category
        st.write("Category Distribution")
        cat_counts = wb_df["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
        st.bar_chart(cat_counts.set_index("Category"))

        st.write("Recent Assessments")
        st.dataframe(wb_df.sort_values("Timestamp", ascending=False).head(10), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Includes a built-in rule-based analyzer and optional ML model. Add a trained model to 'models/' to enable ML predictions.")
