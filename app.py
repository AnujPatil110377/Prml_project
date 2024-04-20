import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from collections import Counter

label_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Custom Tokenizer Class
class CustomTokenizer:
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = {}
        self.index = 1  # Start index from 1 (0 reserved for padding)

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                if word not in self.word_counts:
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1

        # Sort words by frequency and select top num_words if specified
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        if self.num_words:
            sorted_words = sorted_words[:self.num_words]

        # Assign index to each word
        for word, _ in sorted_words:
            self.word_to_index[word] = self.index
            self.index_to_word[self.index] = word
            self.index += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split():
                if word in self.word_to_index:
                    sequence.append(self.word_to_index[word])
            sequences.append(sequence)
        return sequences

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    clean_text = " ".join(tokens)
    return clean_text

# Function to analyze sentiment
def analyze_sentiment(text):
    tokenizer = CustomTokenizer(num_words=5000)  # Assuming you want to tokenize only top 5000 frequent words
    sample_text = text
    
    # Preprocess the sample text
    preprocessed_sample_text = preprocess_text(sample_text)

    # Load TF-IDF vectorizer
    vectorization= joblib.load("Vector.pkl")
    
    # Transform preprocessed text using TF-IDF vectorizer
    XV_test = vectorization.transform([preprocessed_sample_text])

    # Load logistic regression model
    logistic_regression_model = joblib.load("logistic_Regression_model.pkl")
    
    # Make predictions using the logistic regression model
    predictions = logistic_regression_model.predict(XV_test)
    
    # Convert numerical predictions to labels
    predicted_labels = [label_mapping[prediction] for prediction in predictions]
    
    # Get the most common sentiment
    most_common_sentiment = Counter(predicted_labels).most_common(1)[0][0]
    
    st.write("Sentiment:", most_common_sentiment)

# Function to identify abusive users
def identify_abusive_users(csv_file):
    abusive_users = []

    # Load CSV file
    data = pd.read_csv(csv_file, encoding="latin1")

    # Load TF-IDF vectorizer
    vectorization= joblib.load("Vector.pkl")
    
    # Load logistic regression model
    logistic_regression_model = joblib.load("logistic_Regression_model.pkl")
    
    # Process each comment
    for _, row in data.iterrows():
        comment = row['Tweet']
        preprocessed_comment = preprocess_text(comment)
        XV_comment = vectorization.transform([preprocessed_comment])

        # Predict using logistic regression model
        prediction = logistic_regression_model.predict(XV_comment)[0]

        # Check if user is abusive
        if prediction == 0:
            abusive_users.append(row['Name'])

    return abusive_users

# Streamlit UI
st.title("Prml Project")

# Sentiment Analysis
st.header("Sentiment Analysis")
text_input = st.text_input("Enter your text:")
if text_input:
    analyze_sentiment(text_input)

# Upload CSV File and Identify Abusive Users
with st.expander('Analyze CSV for Abosive Users'):
    csv_file = st.file_uploader('Upload file')

if csv_file:
    abusive_users = identify_abusive_users(csv_file)
    if abusive_users:
        st.write("Abusive Users:")
        for user in abusive_users:
            st.write(user)
    else:
        st.write("No abusive users found in the CSV.")
