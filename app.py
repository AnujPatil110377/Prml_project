import streamlit as st


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import zipfile
import os
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
    # blob = TextBlob(text)
    # polarity = round(blob.sentiment.polarity, 2)
    # subjectivity = round(blob.sentiment.subjectivity, 2)
    # return polarity, subjectivity
    tokenizer = CustomTokenizer(num_words=5000)  # Assuming you want to tokenize only top 5000 frequent words
    sample_text = text
    # # Fit the tokenizer on a list of texts (assuming you have a list of texts called `texts`)
    # texts = [sample_text]  # Create a list with the sample text
    # tokenizer.fit_on_texts(texts)

    # # Preprocess the sample text
    # preprocessed_sample_text = preprocess_text(sample_text)

    # # Convert the preprocessed text into sequences of word indices
    # sequences = tokenizer.texts_to_sequences([preprocessed_sample_text])

    # print("Preprocessed Sample Text:", preprocessed_sample_text)
    # print("Sequences:", sequences)
    # import zipfile
    # import os
    # import joblib

    # Step 1: Unzip the models.zip file
    with zipfile.ZipFile('Models.zip', 'r') as zip_ref:
        zip_ref.extractall('models_folder')

    # Step 2: Load Keras models and joblib models
    models_folder = 'models_folder'

    # List all files in the models_folder directory
    model_files = os.listdir(models_folder)

    # Initialize lists to store loaded models
    loaded_keras_models = []
    loaded_joblib_models = []

    # Load Keras models (.h5 files) and append them to the loaded_keras_models list
    for model_file in model_files:
        if model_file.endswith('.h5'):  # Check if the file is a Keras model
            model_path = os.path.join(models_folder, model_file)
            loaded_model = tf.keras.models.load_model(model_path)
            loaded_keras_models.append(loaded_model)

    # Load joblib models (.pkl files) and append them to the loaded_joblib_models list
        elif model_file.endswith('.pkl'):  # Check if the file is a joblib model
            model_path = os.path.join(models_folder, model_file)
            loaded_model = joblib.load(model_path)
            loaded_joblib_models.append(loaded_model)

    # Step 3: Preprocess the sample text
    
    preprocessed_sample_text = preprocess_text(sample_text)

    # Step 4: Convert the preprocessed text into sequences of word indices
    sequences = tokenizer.texts_to_sequences([preprocessed_sample_text])

    # Step 5: Pad the sequences
    maxlen = 100
    X_sample_padded = pad_sequences(sequences, maxlen=maxlen)

    vectorization= joblib.load("Vector.pkl")
    XV_test = vectorization.transform([preprocessed_sample_text])
    

    # Step 6: Use each loaded model to analyze sentiment
    sentiments = []

    for model in loaded_keras_models:
        predictions = model.predict(X_sample_padded)
        # Convert numerical predictions to labels
        predicted_labels = [label_mapping[np.argmax(prediction)] for prediction in predictions]
        sentiments.extend(predicted_labels)

    for model in loaded_joblib_models:
        # Transform preprocessed text using TF-IDF vectorizer
        XV_test = vectorization.transform([preprocessed_sample_text])
        # Make predictions using the joblib model
        predictions = model.predict(XV_test)
        # Convert numerical predictions to labels
        predicted_labels = [label_mapping[prediction] for prediction in predictions]
        sentiments.extend(predicted_labels)

    # Count occurrences of each sentiment label
    sentiment_counts = Counter(sentiments)

    # Find the most common sentiment label
    most_common_sentiment = sentiment_counts.most_common(1)[0][0]
    st.write("sentiment:", most_common_sentiment)
    
    
    

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
