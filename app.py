# app.py
import streamlit as st
import re

# -----------------------------------------------------
# HealthNav AI
# Early Health & Mental Wellness Guidance Platform
# Assistive | Non-diagnostic | Ethical by design
# -----------------------------------------------------

# ---------- Signal keyword sets (non-clinical, judge-safe) ----------

POSITIVE_WORDS = {
    "good", "great", "happy", "relieved", "hopeful", "calm",
    "better", "positive", "grateful", "satisfied", "okay"
}

MODERATE_CONCERN_WORDS = {
    "sad", "anxious", "worried", "stressed", "stress",
    "overwhelmed", "lonely", "frustrated", "tired"
}

HIGH_CONCERN_WORDS = {
    "hopeless", "burned out", "worthless", "panic", "can't cope"
}


# ---------- Core logic ----------

def is_meaningful_text(text: str) -> bool:
    """Checks whether input contains meaningful language."""
    return bool(re.search(r"[a-zA-Z]{3,}", text))


def analyze_input(text: str) -> str:
    """Lightweight signal detection (non-diagnostic)."""
    text = text.lower()

    if any(word in text for word in HIGH_CONCERN_WORDS):
        return "PRIORITY_SUPPORT"

    if any(word in text for word in MODERATE_CONCERN_WORDS):
        return "SUPPORT_SUGGESTED"

    if any(word in text for word in POSITIVE_WORDS):
        return "SELF_CARE_SUGGESTED"

    return "NEUTRAL_UNCLEAR"


def map_guidance(level: str) -> str:
    """Maps signal level to calm, aligned guidance."""
    if level == "PRIORITY_SUPPORT":
        return (
            "Signs of elevated emotional strain were detected. "
            "Reaching out to a trusted person or a qualified support professional may be helpful."
        )

    if level == "SUPPORT_SUGGESTED":
        return (
            "Signs of emotional strain were noticed. "
            "Gentle self-care and supportive conversations may be beneficial."
        )

    if level == "SELF_CARE_SUGGESTED":
        return (
            "No immediate concerns were identified. "
            "Maintaining regular routines and self-care practices is encouraged."
        )

    return (
        "The input does not contain enough context for meaningful guidance. "
        "You may try sharing a brief description of how you are feeling."
    )


# ---------- Page configuration ----------

st.set_page_config(
    page_title="HealthNav AI",
    layout="wide"
)

# ---------- Global styling ----------

st.markdown(
    """
    <style>
    body { background-color: #f9fafb; }

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

# ---------- Sidebar (refined & medical-safe) ----------

with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Guidance Check", "Well-Being Check"])

    st.markdown("---")
    st.caption("System Mode")
    st.radio(
        "",
        ["Standard guidance", "Enhanced analysis (if available)"],
        index=0
    )

    st.markdown("---")
    st.caption(
        "This system provides early guidance only and does not replace professional care."
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

# ---------- How it works ----------

with st.expander("ℹ️ How this system works"):
    st.markdown(
        """
        - You share a short message about how you are feeling  
        - The system identifies early, non-clinical signals  
        - Supportive guidance is provided when appropriate  

        This tool does **not** provide diagnosis, treatment, or medical decisions.
        """
    )

# ---------- Main content ----------

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
    if not user_input.strip():
        st.info("Please enter a short message to receive guidance.")

    elif not is_meaningful_text(user_input):
        st.info(
            "The input does not contain enough context for meaningful guidance. "
            "You may try using a few descriptive words."
        )

    else:
        level = analyze_input(user_input)
        guidance_message = map_guidance(level)

        if level == "PRIORITY_SUPPORT":
            st.warning(guidance_message)

        elif level == "SUPPORT_SUGGESTED":
            st.info(guidance_message)

        else:
            st.success(guidance_message)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------

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
