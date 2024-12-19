import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and preprocessing components
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one space
    return text

# Streamlit app setup
st.title("Text Emotion Classifier")
st.write("A simple app to classify emotions from text.")

# Input text box
user_input = st.text_area("Enter text to analyze:", "")

if st.button("Predict Emotion"):
    if user_input.strip():
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        
        # Transform text using vectorizer
        vectorized_input = vectorizer.transform([processed_text])
        
        # Predict using the pre-trained model
        prediction = model.predict(vectorized_input)
        
        # Display the result
        st.success(f"The predicted emotion is: **{prediction[0]}**")
    else:
        st.warning("Please enter some text to analyze!")
