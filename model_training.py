import pandas as pd
import re
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
print("Loading dataset...")

df = pd.read_csv("data/Amazon_Reviews.csv", engine="python", encoding="latin-1")


# ---------------------------------------------------
# Extract Numeric Rating
# ---------------------------------------------------
df['Rating'] = df['Rating'].str.extract(r'(\d)')
df = df.dropna(subset=['Rating'])
df['Rating'] = df['Rating'].astype(int)


# ---------------------------------------------------
# Create Sentiment Labels
# ---------------------------------------------------
def convert_rating(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment'] = df['Rating'].apply(convert_rating)


# ---------------------------------------------------
# Keep Required Columns
# ---------------------------------------------------
df = df[['Review Text', 'Sentiment']]
df = df.dropna(subset=['Review Text'])


# ---------------------------------------------------
# Clean Text
# ---------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['Cleaned_Review'] = df['Review Text'].apply(clean_text)


# ---------------------------------------------------
# TF-IDF Vectorization
# ---------------------------------------------------
print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(df['Cleaned_Review'])
y = df['Sentiment']


# ---------------------------------------------------
# Train-Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# Train Logistic Regression
# ---------------------------------------------------
print("Training model...")

model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

model.fit(X_train, y_train)


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------
# Save Model & Vectorizer
# ---------------------------------------------------
print("\nSaving model...")

with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")
