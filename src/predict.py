import pickle
from preprocess import clean_tweet

# Load model
with open("models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_sentiment(tweet):
    cleaned = clean_tweet(tweet)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return prediction

# Test
tweet = input("Enter a tweet: ")

print("Sentiment:", predict_sentiment(tweet))