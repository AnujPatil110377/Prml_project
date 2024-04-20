import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os
import zipfile

# Extract models from Models.zip
with zipfile.ZipFile('Models.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_models')

# Load Keras models and joblib models
loaded_keras_models = []
loaded_joblib_models = []
models_folder = 'extracted_models'

label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

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

# Load tokenizer
tokenizer = CustomTokenizer(num_words=5000)  # Set the num_words parameter accordingly

# Load vectorization object
vectorization = joblib.load("Vector.pkl")

# Define preprocess_text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    clean_text = " ".join(tokens)
    return clean_text

# Define maxlen
maxlen = 100  # Set the value accordingly

# Load data from CSV
data = pd.read_csv("tweets.csv",encoding="latin1")

# Initialize predictions_counts dictionary
predictions_counts = {}

# Count predictions per user

# Process each comment
for index, row in data.iterrows():
    user = row['Name']
    comment = row['Tweet']
    
    preprocessed_comment = preprocess_text(comment)
    sequences = tokenizer.texts_to_sequences([preprocessed_comment])
    X_comment_padded = pad_sequences(sequences, maxlen=maxlen)
    
    # Predict using Keras models
    keras_predictions = []
    for model_file in os.listdir(models_folder):
        if model_file.endswith('.h5'):
            model_path = os.path.join(models_folder, model_file)
            loaded_model = tf.keras.models.load_model(model_path)
            loaded_keras_models.append(loaded_model)
            predictions = loaded_model.predict(X_comment_padded)
            predicted_labels = [np.argmax(prediction) for prediction in predictions]
            keras_predictions.extend(predicted_labels)
    
    # Predict using joblib models
    joblib_predictions = []
    for model_file in os.listdir(models_folder):
        if model_file.endswith('.pkl'):
            model_path = os.path.join(models_folder, model_file)
            loaded_model = joblib.load(model_path)
            loaded_joblib_models.append(loaded_model)
            XV_comment = vectorization.transform([preprocessed_comment])
            predictions = loaded_model.predict(XV_comment)
            predicted_labels = [prediction for prediction in predictions]
            joblib_predictions.extend(predicted_labels)
    
    # Combine predictions from all models
    all_predictions = keras_predictions + joblib_predictions
    majority_prediction = max(set(all_predictions), key = all_predictions.count)
    
    # Count predictions for each user
    if user in predictions_counts:
        predictions_counts[user].append(majority_prediction)
    else:
        predictions_counts[user] = [majority_prediction]

# Identify abusive users
abusive_users = [user for user, predictions in predictions_counts.items() if predictions.count(0) > 4]

print("Abusive or Vulgar Users:")
print(abusive_users)
