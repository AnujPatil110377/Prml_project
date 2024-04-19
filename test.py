import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# loaded_model = tf.keras.models.load_model("pretrained_ann_model.h5")
# Sample text for classification
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
sample_text = "Get a discount on our limited time offer! Click here for more details."
import re
import nltk
from nltk.corpus import stopwords
label_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

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
            sequence = [self.word_to_index[word] for word in text.split() if word in self.word_to_index]
            sequences.append(sequence)
        return sequences

# Define a custom preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters, punctuation, and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    clean_text = " ".join(tokens)
    return clean_text
# Initialize the CustomTokenizer
tokenizer = CustomTokenizer(num_words=5000)  # Assuming you want to tokenize only top 5000 frequent words

# Fit the tokenizer on a list of texts (assuming you have a list of texts called `texts`)
texts = [sample_text]  # Create a list with the sample text
tokenizer.fit_on_texts(texts)

# Preprocess the sample text
preprocessed_sample_text = preprocess_text(sample_text)

# Convert the preprocessed text into sequences of word indices
sequences = tokenizer.texts_to_sequences([preprocessed_sample_text])

print("Preprocessed Sample Text:", preprocessed_sample_text)
print("Sequences:", sequences)
import zipfile
import os
import joblib

# Step 1: Unzip the models.zip file
with zipfile.ZipFile('models.zip', 'r') as zip_ref:
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
sample_text = "Get a discount on our limited time offer! Click here for more details bitch fuck hurt kill rip destory no mercy hit stupid hate bully asshole wrong stupid hate hate fuck idiot fire ."
preprocessed_sample_text = preprocess_text(sample_text)

# Step 4: Convert the preprocessed text into sequences of word indices
sequences = tokenizer.texts_to_sequences([preprocessed_sample_text])

# Step 5: Pad the sequences
maxlen = 100
X_sample_padded = pad_sequences(sequences, maxlen=maxlen)

# Step 6: Use each loaded model to make predictions
for model in loaded_keras_models:
    predictions = model.predict(X_sample_padded)
    # Convert numerical predictions to labels
    predicted_labels = [label_mapping[np.argmax(prediction)] for prediction in predictions]
    print("Keras Model Predictions:", predicted_labels)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization= joblib.load("Vector.pkl")
XV_test = vectorization.transform([preprocessed_sample_text])
# print(XV_test)
for model in loaded_joblib_models:
    # Assuming loaded_joblib_models contains only DecisionTreeClassifier models
    # Transform preprocessed text using TF-IDF vectorizer
    
    # Make predictions using the joblib model
    predictions = model.predict(XV_test)
    # Convert numerical predictions to labels
    predicted_labels = [label_mapping[prediction] for prediction in predictions]
    print("Joblib Model Predictions:", predicted_labels)

# random
