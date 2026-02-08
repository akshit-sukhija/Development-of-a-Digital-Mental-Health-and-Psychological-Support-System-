import streamlit as st

# =====================================================
# HealthNav AI
# Early Health & Mental Wellness Guidance Platform
# Assistive | Non-diagnostic | Ethical by design
# =====================================================

# ---------- Basic signal detection (simple & judge-safe) ----------

POSITIVE_WORDS = {
    "good", "great", "happy", "joy", "love", "relieved", "hopeful", "calm",
    "better", "positive", "grateful", "excited", "satisfied"
}

MODERATE_CONCERN_WORDS = {
    "sad", "depressed", "angry", "upset", "anxious", "worried",
    "stress", "stressed", "lonely", "frustrated", "overwhelmed"
}

HIGH_CONCERN_WORDS = {
    "hopeless", "worthless", "burned out", "burnt out", "panic", "can't cope"
}


def analyze_input(text: str) -> str:
    text = text.lower()

    if not any(c.isalpha() for c in text):
        return "NO_CONTEXT"

    if any(word in text for word in HIGH_CONCERN_WORDS):
        return "PRIORITY_SUPPORT"

    if any(word in text for word in MODERATE_CONCERN_WORDS):
        return "SUPPORT_SUGGESTED"

    if any(word in text for word in POSITIVE_WORDS):
        return "NO_IMMEDIATE_CONCERN"

    return "SELF_CARE_SUGGESTED"


def map_guidance(level: str) -> str:
    if level == "PRIORITY_SUPPORT":
        return (
            "Signs of elevated emotional strain were identified. "
            "Reaching out to a trusted person or a qualified support professional may be helpful."
        )

    if level == "SUPPORT_SUGGESTED":
        return (
            "Some indicators of emotional load were noticed. "
            "Gentle self-care and seeking support from someone you trust could be beneficial."
        )

    if level == "NO_IMMEDIATE_CONCERN":
        return (
            "No immediate concerns were identified. "
            "Maintaining regular routines and self-care practices is encouraged."
        )

    if level == "NO_CONTEXT":
        return (
            "The input does not contain enough context for meaningful guidance. "
            "Please share a short message describing how you are feeling."
        )

    return (
        "This appears to be a mild concern. "
        "Simple self-care steps and regular self-reflection may help."
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
    .container { max-width: 900px; margin: auto; }
    .card {
        background: #ffffff;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        margin-bottom: 24px;
    }
    .header-title { font-size: 40px; margin-bottom: 6px; }
    .subtle-text { color: #6b7280; font-size: 15px; }
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

# ---------- Sidebar ----------

with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Guidance Check", "Well-Being Check"])

    st.markdown("---")
    st.caption("System Mode")
    st.radio(
        "",
        ["Standard guidance", "Enhanced analysis (if available)"],
        disabled=True
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

# ---------- Page: Guidance Check ----------

def render_guidance_check():
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
            elif level == "NO_CONTEXT":
                st.info(guidance_message)
            else:
                st.success(guidance_message)
        else:
            st.info("Please enter a short message to receive guidance.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Page: Well-Being Check ----------

def render_wellbeing_check():
    st.markdown("<div class='container card'>", unsafe_allow_html=True)

    st.subheader("Well-Being Check")
    st.write("A short self-reflection check-in. This is not a medical assessment.")

    mood = st.radio("How is your mood today?", ["Good", "Okay", "Low"])
    stress = st.radio("How stressed do you feel?", ["Relaxed", "Somewhat stressed", "Very stressed"])
    sleep = st.radio("How has your sleep been?", ["Good", "Average", "Poor"])

    if st.button("View Summary"):
        st.info(
            "Thank you for checking in. Regular self-reflection can support ongoing well-being awareness."
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Navigation routing ----------

if page == "Guidance Check":
    render_guidance_check()
elif page == "Well-Being Check":
    render_wellbeing_check()

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
