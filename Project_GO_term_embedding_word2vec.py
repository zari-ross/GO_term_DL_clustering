import json

input_file = "cleaned_names.json"

with open(input_file, "r") as f:
    cleaned_terms = json.load(f)

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

from gensim.models import Word2Vec

# Tokenize the cleaned_names data
tokenized_cleaned_names = [name.split() for name in cleaned_terms]

# Train a Word2Vec model
model = Word2Vec(tokenized_cleaned_names, vector_size=50, window=5, min_count=1, workers=4)

# Save the model
model.save("word2vec_go_terms.model")

# Get word embeddings
word_embeddings = {word: model.wv[word] for word in model.wv.key_to_index}

import pickle

# Save the word_embeddings dictionary to a file
with open('word_embeddings.pkl', 'wb') as f:
    pickle.dump(word_embeddings, f)
