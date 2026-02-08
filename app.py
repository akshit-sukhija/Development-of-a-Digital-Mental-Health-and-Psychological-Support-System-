# app.py
import streamlit as st
import re

# -----------------------------------------------------
# HealthNav AI
# Early Health & Mental Wellness Guidance Platform
# Assistive | Non-diagnostic | Ethical by design
# -----------------------------------------------------

# ---------- Basic signal detection (simple & judge-safe) ----------

POSITIVE_WORDS = {
    "good","great","happy","joy","love","relieved","hopeful","calm",
    "better","positive","grateful","excited","satisfied"
}

MODERATE_CONCERN_WORDS = {
    "sad","depressed","angry","upset","anxious","worried","hopeless",
    "stress","stressed","pain","hurt","lonely","panic","frustrated"
}

HIGH_CONCERN_WORDS = {
    "hopeless", "worthless", "panic", "can't cope", "burned out"
}


def analyze_input(text: str) -> str:
    """Lightweight keyword-based signal detection (non-clinical)."""
    text = text.lower()

    if any(word in text for word in HIGH_CONCERN_WORDS):
        return "PRIORITY_SUPPORT"

    if any(word in text for word in MODERATE_CONCERN_WORDS):
        return "SUPPORT_SUGGESTED"

    if any(word in text for word in POSITIVE_WORDS):
        return "NO_IMMEDIATE_CONCERN"

    return "SELF_CARE_SUGGESTED"


def map_guidance(level: str) -> str:
    """Maps signal level to calm, non-alarmist guidance."""
    if level == "PRIORITY_SUPPORT":
        return (
            "Some strong emotional strain signals were noticed. "
            "Reaching out to a trusted person or a qualified support professional may be helpful."
        )

    if level == "SUPPORT_SUGGESTED":
        return (
            "Early signs of stress or emotional load were detected. "
            "Gentle self-care and speaking with someone you trust could be beneficial."
        )

    if level == "NO_IMMEDIATE_CONCERN":
        return (
            "No immediate concerns were identified. "
            "Maintaining healthy routines and regular self-care is encouraged."
        )

    return (
        "This appears to be a mild concern. "
        "Simple self-care steps and regular check-ins with yourself may help."
    )


# ---------- Page configuration ----------

st.set_page_config(
    page_title="HealthNav AI",
    layout="wide"
)

# ---------- Global styling (medical, calm, neutral) ----------

st.markdown(
    """
    <style>
    body {
        background-color: #f9fafb;
    }
    .container {
        max-width: 900px;
        margin: auto;
    }
    .card {
        background: #ffffff;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        margin-bottom: 24px;
    }
    .header-title {
        font-size: 40px;
        margin-bottom: 6px;
    }
    .subtle-text {
        color: #6b7280;
        font-size: 15px;
    }
    .footer-text {
        color: #6b7280;
        font-size: 12px;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Header ----------

st.markdown(
    """
    <div class="container" style="text-align:center; padding: 20px 0 30px 0;">
        <div class="header-title">🧭 HealthNav AI</div>
        <div class="subtle-text">
            Early Health & Mental Wellness Guidance Platform
        </div>
        <div class="subtle-text">
            Assistive • Non-diagnostic • Designed for early support
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- How it works (ethical transparency) ----------

with st.expander("ℹ️ How this system works"):
    st.markdown(
        """
        - You share a short description of how you are feeling or any concern  
        - The system looks for early, non-clinical signals  
        - Supportive guidance is provided for self-care or next steps  

        This tool does **not** provide diagnosis, treatment, or medical decisions.
        """
    )

# ---------- Main content card ----------

st.markdown("<div class='container card'>", unsafe_allow_html=True)

st.subheader("Early Guidance Check")
st.write(
    "Share a brief message about how you are feeling or any concern you would like to reflect on."
)

user_input = st.text_area(
    "",
    height=130,
    placeholder="Example: I have been feeling stressed and tired lately..."
)

if st.button("View Guidance"):
    if user_input.strip():
        level = analyze_input(user_input)
        guidance_message = map_guidance(level)

        if level == "PRIORITY_SUPPORT":
            st.warning(guidance_message)
        elif level == "SUPPORT_SUGGESTED":
            st.info(guidance_message)
        else:
            st.success(guidance_message)
    else:
        st.info("Please enter a short message to receive guidance.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer (regulatory & judge-safe) ----------

st.markdown(
    """
    <hr>
    <div class="footer-text">
        HealthNav AI is an assistive wellness guidance tool.<br>
        It does not provide medical diagnosis or treatment.<br><br>
        Built with ethical AI principles for healthcare • AI HealthX 2026
    </div>
    """,
    unsafe_allow_html=True
)
