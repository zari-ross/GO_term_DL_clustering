import tensorflow as tf
import numpy as np
import json
import pickle
from tensorflow.keras.utils import plot_model

input_file = "cleaned_names_MF_rat.json"

with open(input_file, "r") as f:
    cleaned_terms = json.load(f)

# Load the trained model
trained_model = tf.keras.models.load_model("mask_go_terms.model")

# Generate a PNG image of the model architecture
plot_model(trained_model, to_file='mask_model_architecture.png',
           show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True)
#
# # Load the word_embeddings dictionary
# with open('word_embeddings_mask_trained_on_abstracts.pkl', 'rb') as f:
#     word_embeddings_abstracts = pickle.load(f)
#
# # Create the vectorizer using the vocabulary from the word_embeddings dictionary
# vocab_abstracts = list(word_embeddings_abstracts.keys())
# vectorizer_abstracts = tf.keras.layers.TextVectorization(output_sequence_length=7, vocabulary=vocab_abstracts)
#
# # Vectorize the cleaned_terms
# vectorized_cleaned_terms = vectorizer_abstracts(cleaned_terms)
#
# # Get the embeddings for the cleaned_terms
# embeddings_cleaned_terms = trained_model.layers[0](vectorized_cleaned_terms)
#
# # Create a dictionary of word embeddings for cleaned_terms
# word_embeddings_cleaned_terms = {word: embeddings_cleaned_terms.numpy()[idx] for idx, word in enumerate(cleaned_terms)}
#
# # Save the word_embeddings dictionary to a file
# with open('word_embeddings_mask_MF_rat_trained_on_abstracts.pkl', 'wb') as f:
#     pickle.dump(word_embeddings_cleaned_terms, f)
