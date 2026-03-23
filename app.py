import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("resume_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


st.title("AI Resume Classifier")
st.subheader("Paste your resume text below:")

resume_text = st.text_area("Enter Resume Text")

if st.button("Predict Job Role"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:
        cleaned_resume = clean_text(resume_text)
        vectorized_input = vectorizer.transform([cleaned_resume])
        prediction = model.predict(vectorized_input)
        st.success(f"Predicted Job Role: **{prediction[0]}**")
