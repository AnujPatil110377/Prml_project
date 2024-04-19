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
import pandas as pd

df1 = pd.read_csv('train.csv',encoding='latin1')
df2 = pd.read_csv('test.csv',encoding='latin1')

# Merge the DataFrames
twitter_data = pd.concat([df1, df2], ignore_index=True)

# Write the merged DataFrame to a new CSV file
twitter_data.to_csv('merged_file.csv', index=False)

# Drop rows with NaN or null values in the 'text' column
twitter_data.dropna(subset=['text'], inplace=True)

# Apply the preprocessing function to the 'text' column
twitter_data['text'] = twitter_data['text'].apply(preprocess_text)

# Create an instance of CustomTokenizer
custom_tokenizer = CustomTokenizer(num_words=5000)
custom_tokenizer.fit_on_texts(twitter_data['text'])

# Convert sentiment labels to numeric form
sentiment_mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
twitter_data['sentiment_encoded'] = twitter_data['sentiment'].map(sentiment_mapping)

# Split the dataset into training and testing sets
X = twitter_data['text']
y = twitter_data['sentiment_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert texts to sequences
X_train_seq = custom_tokenizer.texts_to_sequences(X_train)
X_test_seq = custom_tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
maxlen = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_padded = pad_sequences(X_test_seq, maxlen=maxlen)

# Convert labels to categorical format
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Define a function to create the model with specified hyperparameters
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=len(custom_tokenizer.word_to_index) + 1, output_dim=100, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# best
#{'optimizer': 'rmsprop', 'epochs': 3, 'batch_size': 128}
# # Define the hyperparameters to tune
# optimizers = ['adam', 'rmsprop', 'sgd']
# epochs_list = [2,3, 5, 8]
# batch_sizes = [64, 128, 256]
optimizers = ['rmsprop']
epochs_list = [3]
batch_sizes = [128]

# Perform hyperparameter tuning
best_accuracy = 0
best_hyperparameters = {}


print(f"Training model with optimizer: {'rmsprop'}, epochs: {3}, batch_size: {128}")
            
            # Create and compile the model with current hyperparameters
model = create_model(optimizer='rmsprop')
            
            # Train the model
history = model.fit(X_train_padded, y_train_categorical, epochs=3, batch_size=128, validation_split=0.3, verbose=1)
            
            # Evaluate the model on validation data
accuracy = model.evaluate(X_test_padded, y_test_categorical, verbose=0)
print(f"Validation Accuracy: {accuracy}")
            
 

model.save("pretrained_ann_model.h5")