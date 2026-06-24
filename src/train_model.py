import pandas as pd

import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_tweet

# Load dataset
df = pd.read_csv("data/Tweets.csv")

# Keep only required columns
df = df[["text", "airline_sentiment"]]

print("Cleaning tweets...")

# Apply preprocessing
df["clean_text"] = df["text"].apply(clean_tweet)

# Features and labels
X = df["clean_text"]
y = df["airline_sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Save model
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")