import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import re

# -------------------------
# Rule-based lexicon analyzer
# -------------------------
POS_WORDS = {"good", "great", "happy", "joy", "love", "awesome", "fantastic", "relieved", "satisfied", "hopeful", "calm", "better", "improved", "positive", "grateful"}
NEG_WORDS = {"sad", "depressed", "angry", "upset", "anxious", "worried", "hopeless", "terrible", "awful", "stress", "stressed", "pain", "hurt", "bad", "negative", "lonely", "suicidal", "angst", "panic"}
NEGATIONS = {"not", "no", "never", "n't", "hardly", "rarely"}
INTENSIFIERS = {"very", "extremely", "really", "so", "too", "super"}

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
# Try load ML model
# -------------------------
@st.cache_resource
def try_load_ml_model():
    try:
        model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        return model, vectorizer
    except Exception:
        return None, None

ml_model, ml_vectorizer = try_load_ml_model()

# -------------------------
# Streamlit Setup
# -------------------------
st.set_page_config(page_title="Digital Mental Health Support", layout="wide")

# Inject custom CSS
st.markdown("""
<style>
body { background-color: #F5F7FA; color: #333333; }
.navbar { display: flex; justify-content: space-between; align-items: center; padding: 12px 24px; background: white; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);}
.nav-left { font-size: 20px; font-weight: bold; color: #4CAF50; }
.nav-center { display: flex; gap: 24px; }
.nav-item { font-size: 16px; cursor: pointer; padding: 6px 12px; border-radius: 6px; }
.nav-item:hover { background: #E8F5E9; }
.nav-active { background: #4CAF50; color: white; }
.card { background: white; border-radius: 16px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);}
.btn-analyze { background: linear-gradient(90deg, #4CAF50, #2196F3); color: white; border: none; padding: 12px; border-radius: 8px; width: 100%; font-size: 16px; font-weight: bold; cursor: pointer; }
.btn-analyze:hover { opacity: 0.9; }
.footer { text-align: center; margin-top: 30px; color: #777; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Top Navigation
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

nav_items = ["Home", "Student Dashboard", "Admin Dashboard", "About"]
cols = st.columns([2,6,2])
with cols[0]: st.markdown('<div class="nav-left">🌿 Digital Mental Health</div>', unsafe_allow_html=True)
with cols[1]:
    nav_html = '<div class="nav-center">'
    for item in nav_items:
        cls = "nav-item nav-active" if st.session_state.page == item else "nav-item"
        nav_html += f'<span class="{cls}" onclick="window.parent.postMessage({{type: \'streamlit:setSessionState\', key: \'page\', value: \'{item}\' }}, \'*\')">{item}</span>'
    nav_html += '</div>'
    st.markdown(nav_html, unsafe_allow_html=True)

st.write("")  # spacing

# -------------------------
# Home Page
# -------------------------
if st.session_state.page == "Home":
    st.markdown("<div class='card'>"
                "<h2>🌿 Digital Mental Health Support</h2>"
                "<p><i>\"Analyze feedback, track emotions, and support well-being\"</i></p>"
                "</div>", unsafe_allow_html=True)

# -------------------------
# Student Dashboard
# -------------------------
elif st.session_state.page == "Student Dashboard":
    st.markdown("<div class='card'><h3>✍️ Sentiment Analyzer</h3></div>", unsafe_allow_html=True)
    user_input = st.text_area("Type your thoughts here...", height=100)
    if st.button("💡 Analyze Sentiment", use_container_width=True):
        if not user_input.strip():
            st.warning("Please type something.")
        else:
            if ml_model:
                try:
                    X = ml_vectorizer.transform([user_input])
                    pred_raw = ml_model.predict(X)[0]
                    label = str(pred_raw).lower()
                except Exception:
                    label, conf = rule_sentiment(user_input)
            else:
                label, conf = rule_sentiment(user_input)

            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append((user_input, label))

            # Show result
            if label == "positive":
                st.success(f"😊 Positive")
            elif label == "negative":
                st.error(f"😞 Negative")
            else:
                st.info(f"😐 Neutral")

            # Well-being tips
            st.markdown("<div class='card'><h4>🌱 Recommended Tips</h4></div>", unsafe_allow_html=True)
            if label == "positive":
                st.write("✅ Keep up the good vibes! You can:")
                st.write("- Share your happiness with a friend")
                st.write("- Keep journaling positive experiences")
            elif label == "negative":
                st.write("💡 Here are some tips to feel better:")
                st.write("- Take deep breaths for 2 minutes")
                st.write("- Go outdoors for a short walk")
                st.write("- Talk to a trusted friend or counselor")
            else:  # neutral
                st.write("😐 Neutral mood detected. Try these to boost your day:")
                st.write("- Listen to calming music")
                st.write("- Stretch or do light exercise")
                st.write("- Drink water and stay hydrated")

    if "history" in st.session_state and st.session_state.history:
        st.markdown("<div class='card'><h4>📜 Recent Analyses</h4></div>", unsafe_allow_html=True)
        for text, lbl in st.session_state.history[-3:][::-1]:
            if lbl == "positive": st.write(f"😊 **Positive:** {text}")
            elif lbl == "negative": st.write(f"😞 **Negative:** {text}")
            else: st.write(f"😐 **Neutral:** {text}")

        preds = [lbl for _, lbl in st.session_state.history]
        counts = {"Positive": preds.count("positive"), "Negative": preds.count("negative"), "Neutral": preds.count("neutral")}
        vals = list(counts.values())
        if sum(vals) > 0:
            fig, ax = plt.subplots()
            ax.pie(vals, labels=counts.keys(), autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4))
            st.pyplot(fig)

# -------------------------
# Admin Dashboard
# -------------------------
elif st.session_state.page == "Admin Dashboard":
    st.markdown("<div class='card'><h3>📊 Admin Dashboard</h3></div>", unsafe_allow_html=True)
    if "history" not in st.session_state or not st.session_state.history:
        st.warning("No feedback yet.")
    else:
        df = pd.DataFrame(st.session_state.history, columns=["Comment", "Sentiment"])
        st.dataframe(df)

        st.subheader("Aggregate Sentiment")
        st.bar_chart(df["Sentiment"].value_counts())

# -------------------------
# About Page
# -------------------------
elif st.session_state.page == "About":
    st.markdown("<div class='card'><h3>ℹ️ About</h3>"
                "<p>This project helps students reflect on their emotions by analyzing feedback and visualizing sentiments.</p>"
                "<p>Designed with ❤️ for mental well-being.</p></div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Made with ❤️ by Team XYZ</div>", unsafe_allow_html=True)
