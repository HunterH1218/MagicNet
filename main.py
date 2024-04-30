import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import numpy as np
import os
import pickle

model_path = 'chatbot_model.h5'
tokenizer_path = 'chatbot_tokenizer.pickle'
answers_index_path = 'chatbot_answers_index.pickle'
max_length_path = 'chatbot_max_length.pickle'  # Path to save the max_length variable

# Function to save max_length along with tokenizer and answers_index
def save_model_utilities(tokenizer, answers_index, max_length):
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(answers_index_path, 'wb') as handle:
        pickle.dump(answers_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(max_length_path, 'wb') as handle:
        pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)  # Saving the max_length

# Function to load tokenizer, answers_index, and max_length
def load_model_utilities():
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(answers_index_path, 'rb') as handle:
        answers_index = pickle.load(handle)
    with open(max_length_path, 'rb') as handle:
        max_length = pickle.load(handle)  # Loading the max_length
    return tokenizer, answers_index, max_length

train = False

# Load data
with open('data.json') as file:
    data = json.load(file)

# Check if the model and utilities already exist
if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(answers_index_path) and os.path.exists(max_length_path) and not train:
    print("Loading trained model and utilities...")
    model = tf.keras.models.load_model(model_path)
    tokenizer, answers_index, max_length = load_model_utilities()
else:
    questions = [item["Q"] for item in data]
    answers = [item["A"] for item in data]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(questions + answers)
    vocab_size = len(tokenizer.word_index) + 1
    
    question_sequences = tokenizer.texts_to_sequences(questions)
    max_length = max(len(x) for x in question_sequences)
    question_padded = pad_sequences(question_sequences, maxlen=max_length, padding='post')

    answers_index = dict((c, i) for i, c in enumerate(set(answers)))
    answers_seq = [answers_index[a] for a in answers]

    model = Sequential([
        Embedding(vocab_size, 16, input_length=max_length),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(len(answers_index), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    answers_seq = np.array(answers_seq)

    print("Training model...")
    model.fit(question_padded, answers_seq, epochs=200, verbose=1)
    
    model.save(model_path)
    save_model_utilities(tokenizer, answers_index, max_length)

def ask_chatbot(question):
    sequence = tokenizer.texts_to_sequences([question])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')  # Using the loaded/max_length
    prediction = model.predict(padded)
    answer_index = np.argmax(prediction)
    for answer, index in answers_index.items():
        if index == answer_index:
            return answer
    return "I'm not sure how to answer that."

print(ask_chatbot("What's your name?"))
