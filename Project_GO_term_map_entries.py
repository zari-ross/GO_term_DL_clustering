import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data from a file
    data = pickle.load(f)
with open('cluster_data.pkl', 'rb') as f:

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
    plt.annotate(cluster_word, xy=(cluster_centroid[0], cluster_centroid[1]), xytext=(-15, 15), textcoords='offset points',
                 ha='center', va='center', fontsize=12, color='red',
                 bbox=dict(boxstyle='round, pad=0.5', edgecolor='red', facecolor='white', alpha=0.7))

plt.show()