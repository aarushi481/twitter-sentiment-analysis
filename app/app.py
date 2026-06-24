import streamlit as st
import pickle

import sys
import os
from wordcloud import WordCloud

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import clean_tweet

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with open(os.path.join(BASE_DIR, "models", "sentiment_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

st.title("Twitter Sentiment Analysis")

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    layout="wide"
)


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/Tweets.csv")



# Load dataset
df = pd.read_csv(os.path.join(BASE_DIR, "data", "Tweets.csv"))

# ==========================
# Statistics Cards
# ==========================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Tweets",
    len(df)
)

col2.metric(
    "Positive",
    (df["airline_sentiment"] == "positive").sum()
)

col3.metric(
    "Neutral",
    (df["airline_sentiment"] == "neutral").sum()
)


col4.metric(
    "Negative",
    (df["airline_sentiment"] == "negative").sum()
)



st.subheader("Dataset Sentiment Distribution")

sentiments = df["airline_sentiment"].value_counts()

fig, ax = plt.subplots()

ax.pie(
    sentiments.values,
    labels=sentiments.index,
    autopct="%1.1f%%"
)

ax.set_title("Sentiment Distribution")

st.pyplot(fig)
st.subheader("Most Frequent Words")

text = " ".join(df["text"].astype(str))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

fig, ax = plt.subplots(figsize=(10, 5))

ax.imshow(wordcloud)

ax.axis("off")

st.pyplot(fig)

tweet = st.text_area("Enter Tweet")

if "history" not in st.session_state:
    st.session_state.history = []


if st.button("Analyze"):
    cleaned = clean_tweet(tweet)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities) * 100

    if prediction == "positive":
        st.success("😊 Positive Sentiment")

    elif prediction == "negative":
        st.error("😠 Negative Sentiment")

    else:
        st.warning("😐 Neutral Sentiment")

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities) * 100

    if confidence < 60:
        st.warning(
            f"Low confidence prediction ({confidence:.2f}%)."
        )
    
    st.session_state.history.append({
        "Tweet": tweet,
        "Prediction": prediction
    })
    



st.subheader("Prediction History")

st.dataframe(st.session_state.history)

if st.button("Clear History"):
    st.session_state.history = []
    st.rerun()