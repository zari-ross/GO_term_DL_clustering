import argparse
import tensorflow as tf
import json
import pickle
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description='Compute embeddings')
parser.add_argument('terms', type=str, help='Input JSON file with terms')
parser.add_argument('model', type=str, help='Input model file')
parser.add_argument('pkl', type=str, help='Input pickle file with word embeddings')
parser.add_argument('out_json', type=str, help='Output JSON file with terms and embeddings')
parser.add_argument('png', type=str, help='Output PNG file with model architecture')
args = parser.parse_args()

with open(args.terms, "r") as f:
    cleaned_terms = json.load(f)

# Load the trained model
trained_model = tf.keras.models.load_model(args.model)

# Generate a PNG image of the model architecture
plot_model(trained_model, to_file=args.png,
           show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True)

# Load the word_embeddings dictionary
with open(args.pkl, 'rb') as f:
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
with open(args.out_json, 'w') as f:
    json.dump(cleaned_terms, f)
