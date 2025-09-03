import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# -------------------------
# Example dataset (replace with your own)
# -------------------------
data = {
    "text": [
        "I am very happy today",
        "This is awesome, I feel great",
        "Life is good and positive",
        "I am feeling sad",
        "This is terrible, I am depressed",
        "I hate this, very bad",
        "It was okay, not bad but not great",
        "I am anxious about exams",
        "I love this experience",
        "Feeling down and lonely"
    ],
    "label": [
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "neutral",
        "negative",
        "positive",
        "negative"
    ]
}

df = pd.DataFrame(data)

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# -------------------------
# Vectorizer + Model
# -------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# -------------------------
# Save model + vectorizer
# -------------------------
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model and vectorizer saved in /models/")
