import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Load Model + Vectorizer
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Digital Mental Health", layout="wide")

# -------------------------
# Session State for History
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Student Dashboard", "Admin Dashboard"])

# ====================================================
# STUDENT DASHBOARD
# ====================================================
if page == "Student Dashboard":
    st.title("💬 Digital Mental Health Support")
    st.subheader("Analyze comments received through e-consultation")

    # Text input
    user_input = st.text_area("✍️ Enter a comment here:")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]

            # Save in history
            st.session_state.history.append((user_input, prediction))

            # Display result
            if prediction == "positive":
                st.success("😊 Sentiment: Positive")
            elif prediction == "negative":
                st.error("😞 Sentiment: Negative")
            else:
                st.info("😐 Sentiment: Neutral")
        else:
            st.warning("⚠️ Please enter a comment before analyzing.")

    # History
    st.write("---")
    st.subheader("📜 Recent Analyses")
    if st.session_state.history:
        for idx, (text, pred) in enumerate(st.session_state.history[-5:][::-1], start=1):
            st.write(f"**{idx}.** *{text}* → **{pred}**")
    else:
        st.write("No analyses yet.")

    # Pie chart
    st.write("---")
    st.subheader("📊 Sentiment Distribution")
    if st.session_state.history:
        preds = [p for _, p in st.session_state.history]
        counts = {
            "positive": preds.count("positive"),
            "negative": preds.count("negative"),
            "neutral": preds.count("neutral"),
        }
        fig, ax = plt.subplots()
        ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.write("No data yet to show distribution.")

# ====================================================
# ADMIN DASHBOARD
# ====================================================
elif page == "Admin Dashboard":
    st.title("🛠️ Admin Dashboard - Mental Health Insights")

    if not st.session_state.history:
        st.warning("No student feedback available yet.")
    else:
        # Convert history to DataFrame
        df = pd.DataFrame(st.session_state.history, columns=["Comment", "Sentiment"])

        # Show raw data
        st.subheader("📋 Student Comments Data")
        st.dataframe(df)

        # Sentiment counts
        st.subheader("📊 Overall Sentiment Summary")
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        st.bar_chart(sentiment_counts.set_index("Sentiment"))

        # Example flagged comments (negative only)
        st.subheader("🚩 Flagged Negative Comments")
        negatives = df[df["Sentiment"] == "negative"]["Comment"].tolist()
        if negatives:
            for c in negatives:
                st.write(f"- {c}")
        else:
            st.write("✅ No negative comments detected yet.")

