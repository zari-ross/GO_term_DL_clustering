import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pronto
from Project_GO_term_cleaning_names import clean_name
from collections import Counter
from sklearn.cluster import KMeans

# Choose the number of clusters
num_clusters = 20
kmeans = KMeans(n_clusters=num_clusters)

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

# # Get term names for the GO IDs in your experiment
# term_names = [ontology[go_id].name for go_id in go_ids if go_id in ontology]

# Get term names for the GO IDs in your experiment with namespace "molecular_function"
term_names = [ontology[go_id].name for go_id in go_ids if go_id in ontology and ontology[go_id].namespace == "molecular_function"]

filtered_names = [name for name in term_names if "obsolete" not in name.lower() and "unknown" not in name.lower() and
                  "uncharacterized" not in name.lower()]

# Clean all filtered names
cleaned_names = [clean_name(name) for name in filtered_names]

print(len(cleaned_names))

# Tokenize the new cleaned_names data
tokenized_cleaned_names = [name.split() for name in cleaned_names]

# Flatten the list of lists and get unique words
unique_words_in_cleaned_names = list(set([word for name in tokenized_cleaned_names for word in name]))

# Get the 2D embeddings for the experiment words
experiment_word_2d_embeddings = [embeddings_2d[words.index(word)] for word in unique_words_in_cleaned_names]

# Predict the cluster assignments for the experiment words
experiment_word_clusters = kmeans.predict(experiment_word_2d_embeddings)

# Count the number of experiment words in each cluster
experiment_word_counts = Counter(experiment_word_clusters)

# Count the number of all words in each cluster
all_word_counts = Counter(clusters)

# Calculate the enrichment score for each cluster
enrichment_scores = {}
for cluster_id in range(num_clusters):
    experiment_prop = experiment_word_counts[cluster_id] / len(tokenized_cleaned_names)
    all_prop = all_word_counts[cluster_id] / len(words)
    enrichment_scores[cluster_id] = experiment_prop / all_prop

print(enrichment_scores)

# Plot the enrichment scores
plt.bar(range(num_clusters), [enrichment_scores[i] for i in range(num_clusters)])
plt.xlabel("Cluster ID")
plt.ylabel("Enrichment Score")
plt.title("Cluster Enrichment")
plt.show()

# Plot the 2D embeddings for all words
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis')

# Add cluster representative labels
for cluster_id, cluster_word in cluster_representatives.items():
    cluster_centroid = np.mean(embeddings_2d[clusters == cluster_id], axis=0)
    plt.annotate(cluster_word, xy=(cluster_centroid[0], cluster_centroid[1]), xytext=(-15, 15),
                 textcoords='offset points',
                 ha='center', va='center', fontsize=12, color='red',
                 bbox=dict(boxstyle='round, pad=0.5', edgecolor='red', facecolor='white', alpha=0.7))

# Plot the experiment_word_2d_embeddings points in red
experiment_word_2d_embeddings_np = np.array(experiment_word_2d_embeddings)
plt.scatter(experiment_word_2d_embeddings_np[:, 0], experiment_word_2d_embeddings_np[:, 1], c='red', marker='x')

# Add black labels for the experiment_word_2d_embeddings
for i, word in enumerate(unique_words_in_cleaned_names):
    x, y = experiment_word_2d_embeddings_np[i, :]
    plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', color='black')

plt.show()
