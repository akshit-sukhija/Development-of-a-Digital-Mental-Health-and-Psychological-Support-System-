# 🧭 HealthNav AI  
### Early Health & Mental Wellness Guidance System

HealthNav AI is an **AI-assisted, non-diagnostic guidance system** designed to support users in understanding **early mental wellness concerns** and **basic health navigation needs**.  
It provides **general guidance, emotional support, and escalation suggestions** without replacing medical or mental health professionals.

> ⚠️ **Important:** HealthNav AI is NOT a medical device.  
> It does **not diagnose conditions**, **assess medical risk**, or **provide treatment or prescriptions**.

---

## 🔍 Problem Statement

Many individuals struggle to understand:
- Whether their **emotional distress** requires professional support  
- How to respond to **early or mild health concerns**
- When to **seek help vs. self-manage**

Mental stress and anxiety often go unexpressed due to stigma, while early physical symptoms create confusion about next steps.  
This gap leads to delayed care, unnecessary panic, or complete avoidance of support.

---

## 💡 Solution Overview

**HealthNav AI** acts as an **early guidance layer** by:
- Providing **supportive, non-clinical feedback**
- Encouraging **self-care and reflection**
- Suggesting **when professional help may be appropriate**
- Supporting **both mental wellness and early health navigation**

The system is intentionally designed to be:
- Assistive, not authoritative  
- Ethical, not diagnostic  
- Simple, not clinical  

---

## 🧠 How It Works

1. User enters a short message describing how they feel or a concern  
2. AI analyzes sentiment using:
   - Rule-based NLP (default)
   - Optional ML sentiment model (if available)
3. The system maps input to **guidance levels**:
   - Self-Care Guidance  
   - Support Suggested  
   - Immediate Support Recommended  
4. Users receive **safe, supportive guidance** and encouragement to seek help when appropriate

No medical decisions are made.

---

## 🧪 Key Features

- 💬 Emotional sentiment understanding  
- 🌱 Well-being self-reflection check  
- 🧭 Early health navigation prompts  
- 🔁 Rule-based fallback (no model dependency)  
- 📊 Simple, explainable logic  

---

## 🛡️ Safety & Ethics

HealthNav AI is built with strict safety boundaries:

- ❌ No diagnosis  
- ❌ No medical advice  
- ❌ No prescriptions  
- ❌ No replacement of professionals  

✔ Clear disclaimers  
✔ Conservative response wording  
✔ Encourages real-world support when needed  

---

## 🧰 Tech Stack

- **Frontend:** Streamlit  
- **Language:** Python  
- **AI Logic:**  
  - Rule-based NLP  
  - Optional ML sentiment classifier  
- **Libraries:** pandas, numpy, matplotlib, joblib  

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
