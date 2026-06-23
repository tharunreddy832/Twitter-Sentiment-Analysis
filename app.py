import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):

    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.lower().split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

st.title("Twitter Sentiment Analysis")

tweet = st.text_area("Enter a tweet")

if st.button("Predict"):

    cleaned = clean_text(tweet)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    if prediction == "positive":
        st.success("😊 Positive Sentiment")
    else:
        st.error("😞 Negative Sentiment")