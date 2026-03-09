import streamlit as st
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# load model
model = pickle.load(open("spam_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

stemmer = PorterStemmer()

def preprocess(text):

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()

    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]

    return " ".join(words)


# Streamlit UI
st.title("📩 SMS Spam Detection")
st.write("Enter a message to check if it is Spam or Not")

message = st.text_area("Enter Message")

if st.button("Check"):

    processed = preprocess(message)

    vector = vectorizer.transform([processed]).toarray()

    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.error("⚠️ Spam Message")
    else:
        st.success("✅ Normal Message")