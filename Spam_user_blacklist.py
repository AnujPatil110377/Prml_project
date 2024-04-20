import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os
loaded_keras_models = []
loaded_joblib_models = []
models_folder = 'models_folder'
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


for model_file in os.listdir(models_folder):
    if model_file.endswith('.h5'):
        model_path = os.path.join(models_folder, model_file)
        loaded_model = tf.keras.models.load_model(model_path)
        loaded_keras_models.append(loaded_model)

    elif model_file.endswith('.pkl'):
        model_path = os.path.join(models_folder, model_file)
        loaded_model = joblib.load(model_path)
        loaded_joblib_models.append(loaded_model)

vectorization = joblib.load("Vector.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    clean_text = " ".join(tokens)
    return clean_text

data = pd.read_csv("comments.csv")
negative_counts = {}

for index, row in data.iterrows():
    user = row['user']
    comment = row['comment']
    
    preprocessed_comment = preprocess_text(comment)
    sequences = tokenizer.texts_to_sequences([preprocessed_comment])
    X_comment_padded = pad_sequences(sequences, maxlen=maxlen)
    
    for model in loaded_keras_models:
        predictions = model.predict(X_comment_padded)
        predicted_labels = []
        for prediction in predictions:
            predicted_labels.append(label_mapping[np.argmax(prediction)])
        negative_count = predicted_labels.count('negative')
        if user in negative_counts:
            negative_counts[user] += negative_count
        else:
            negative_counts[user] = negative_count
        
    XV_comment = vectorization.transform([preprocessed_comment])
    
    for model in loaded_joblib_models:
        predictions = model.predict(XV_comment)
        predicted_labels = []
        for prediction in predictions:
            predicted_labels.append(label_mapping[prediction])
        negative_count = predicted_labels.count('negative')
        if user in negative_counts:
            negative_counts[user] += negative_count
        else:
            negative_counts[user] = negative_count

abusive_users = []
for user, count in negative_counts.items():
    if count > 6:
        abusive_users.append(user)

print("Abusive or Vulgar Users:")
print(abusive_users)
