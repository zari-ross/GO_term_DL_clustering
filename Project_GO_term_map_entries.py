import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pronto
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from Project_GO_term_cleaning_names import clean_name
from gensim.models import Word2Vec

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load data from a file
with open('cluster_data.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings_2d = data['embeddings_2d']
words = data['words']
clusters = data['clusters']
cluster_representatives = data['cluster_representatives']

# Select 50 random words
selected_words = random.sample(words, 150)
selected_indices = [words.index(word) for word in selected_words]

# Plot the 2D embeddings for all words
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis')

# Add labels for the selected words
for i in selected_indices:
    x, y = embeddings_2d[i, :]
    plt.annotate(words[i], xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

# Add cluster representative labels
for cluster_id, cluster_word in cluster_representatives.items():
    cluster_centroid = np.mean(embeddings_2d[clusters == cluster_id], axis=0)
    plt.annotate(cluster_word, xy=(cluster_centroid[0], cluster_centroid[1]), xytext=(-15, 15),
                 textcoords='offset points',
                 ha='center', va='center', fontsize=12, color='red',
                 bbox=dict(boxstyle='round, pad=0.5', edgecolor='red', facecolor='white', alpha=0.7))

plt.show()

filename = "GO_terms_Example1.txt"

# Read the file into a DataFrame
df = pd.read_csv(filename, sep='\s+', engine='python')

# Extract the GeneGroup column as a list of GO IDs
go_ids = df['GeneGroup'].tolist()

# Load OBO file
file_path = "go-basic.obo"
ontology = pronto.Ontology(file_path)

# Get term names for the GO IDs in your experiment
term_names = [ontology[go_id].name for go_id in go_ids if go_id in ontology]

filtered_names = [name for name in term_names if "obsolete" not in name.lower() and "unknown" not in name.lower() and
                  "uncharacterized" not in name.lower()]

# Clean all filtered names
cleaned_names = [clean_name(name) for name in filtered_names]

print(len(cleaned_names))

# Tokenize the new cleaned_names data
tokenized_cleaned_names = [name.split() for name in cleaned_names]

# Load the saved Word2Vec model
model = Word2Vec.load("word2vec_go_terms.model")

# Get word embeddings for the new cleaned_names
experiment_word_embeddings = {}
for name in tokenized_cleaned_names:
    for word in name:
        if word in model.wv.key_to_index:
            experiment_word_embeddings[word] = model.wv[word]

print(experiment_word_embeddings)

# # 5. Calculate the enrichment in the clusters
# # One way is to calculate the frequency of each cluster in the experiment
# cluster_frequencies = np.bincount(experiment_clusters)
# total_clusters = len(cluster_frequencies)
#
# # Print the enrichment results
# for cluster_id, freq in enumerate(cluster_frequencies):
#     enrichment = freq / total_clusters
#     print(f"Enrichment of cluster {cluster_id}: {enrichment}")