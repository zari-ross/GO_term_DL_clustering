import tensorflow as tf
import json
import pickle
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

with open("rat_cleaned_terms.json", "r") as f:
    cleaned_terms = json.load(f)

# Load the trained model
trained_model = tf.keras.models.load_model("mask_go_terms.model")

# Generate a PNG image of the model architecture
plot_model(trained_model, to_file='mask_model_architecture.png',
           show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True)

# Load the word_embeddings dictionary
with open('word_embeddings_mask_trained_on_abstracts.pkl', 'rb') as f:
    word_embeddings_abstracts = pickle.load(f)

# Create the vectorizer using the vocabulary from the word_embeddings dictionary
vocab_abstracts = list(word_embeddings_abstracts.keys())
vectorizer_abstracts = tf.keras.layers.TextVectorization(output_sequence_length=7, vocabulary=vocab_abstracts)

# Create a list of cleaned term names
cleaned_term_names = [term_info['cleaned_name'] for term_info in cleaned_terms.values()]

# Calculate the lengths of the cleaned term names
lengths = [len(name.split()) for name in cleaned_term_names]

# Plot the histogram
plt.hist(lengths, bins='auto', edgecolor='black')
plt.xlabel('Length of Cleaned Term Names')
plt.ylabel('Frequency')
plt.title('Histogram of Cleaned Term Names Lengths')
plt.show()

# Vectorize the cleaned_term_names
vectorized_cleaned_terms = vectorizer_abstracts(cleaned_term_names)

# Get the embeddings for the cleaned_terms
embeddings_cleaned_terms = trained_model.layers[0](vectorized_cleaned_terms)

# Update the cleaned_terms dictionary with the embeddings
for idx, (term_id, term_info) in enumerate(cleaned_terms.items()):
    term_info['embedding'] = embeddings_cleaned_terms.numpy()[idx].tolist()

# Save the updated cleaned_terms dictionary to a file
with open('rat_cleaned_terms_with_embeddings.json', 'w') as f:
    json.dump(cleaned_terms, f)
