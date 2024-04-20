import streamlit as st
from textblob import TextBlob
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import zipfile
import os
import numpy as np
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

# Load models from the zip file
def load_models_from_zip(zip_file_path):
    # Create a temporary directory to extract models
    temp_dir = "temp_models"
    os.makedirs(temp_dir, exist_ok=True)
    # Extract models from zip
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    # Load Keras models and joblib models
    loaded_keras_models = []
    loaded_joblib_models = []
    models_folder = os.path.join(temp_dir, 'extracted_models')
    for model_file in os.listdir(models_folder):
        if model_file.endswith('.h5'):  # Keras model
            loaded_model = tf.keras.models.load_model(os.path.join(models_folder, model_file))
            loaded_keras_models.append(loaded_model)
        elif model_file.endswith('.pkl'):  # Joblib model
            loaded_model = joblib.load(os.path.join(models_folder, model_file))
            loaded_joblib_models.append(loaded_model)
    return loaded_keras_models, loaded_joblib_models

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    clean_text = " ".join(tokens)
    return clean_text

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)
    return polarity, subjectivity

# Function to identify abusive users in CSV
def identify_abusive_users(csv_file, tokenizer, vectorizer, keras_models, joblib_models):
    abusive_users = []

    # Load CSV file
    data = pd.read_csv(csv_file, encoding="latin1")

    # Process each comment
    for _, row in data.iterrows():
        comment = row['Tweet']
        preprocessed_comment = preprocess_text(comment)
        sequences = tokenizer.texts_to_sequences([preprocessed_comment])
        X_comment_padded = pad_sequences(sequences, maxlen=100)

        # Predict using Keras models
        keras_predictions = []
        for model in keras_models:
            predictions = model.predict(X_comment_padded)
            predicted_labels = [np.argmax(prediction) for prediction in predictions]
            keras_predictions.extend(predicted_labels)

        # Predict using joblib models
        joblib_predictions = []
        for model in joblib_models:
            XV_comment = vectorizer.transform([preprocessed_comment])
            predictions = model.predict(XV_comment)
            predicted_labels = [prediction for prediction in predictions]
            joblib_predictions.extend(predicted_labels)

        # Combine predictions from all models
        all_predictions = keras_predictions + joblib_predictions
        majority_prediction = max(set(all_predictions), key=all_predictions.count)

        # Check if user is abusive
        if majority_prediction == 0:
            abusive_users.append(row['Name'])

    return abusive_users

# Streamlit UI
st.title("Abusive User Detection")

# Sentiment Analysis
st.header("Sentiment Analysis")
text_input = st.text_input("Enter your text:")
if text_input:
    polarity, subjectivity = analyze_sentiment(text_input)
    st.write(f"Sentiment Polarity: {polarity}")
    st.write(f"Subjectivity: {subjectivity}")

# Upload CSV File and Identify Abusive Users
st.header("Identify Abusive Users from CSV")
csv_file = st.file_uploader("Upload CSV file", type=['csv'])
if csv_file:
    # Load tokenizer
    tokenizer = CustomTokenizer(num_words=5000)  # Set the num_words parameter accordingly
    # Load vectorization object
    vectorizer = joblib.load("Vector.pkl")
    # Load models from zip file
    keras_models, joblib_models = load_models_from_zip("models.zip")
    abusive_users = identify_abusive_users(csv_file, tokenizer, vectorizer, keras_models, joblib_models)
    if abusive_users:
        st.write("Abusive Users:")
        for user in abusive_users:
            st.write(user)
    else:
        st.write("No abusive users found in the CSV.")
