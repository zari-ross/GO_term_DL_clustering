import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import random
import pickle
from sklearn.cluster import KMeans

# # Load the word_embeddings dictionary from the file
with open('word_embeddings_mask.pkl', 'rb') as f:
    word_embeddings = pickle.load(f)

# Prepare data for t-SNE
words = list(word_embeddings.keys())
embeddings = np.array([word_embeddings[word] for word in words])

# # Select a smaller number of words (e.g., 300)
# num_words = 300
# check_words = random.sample(words, num_words)
#
# embeddings = np.array([word_embeddings[word] for word in check_words])

# # Load the term_embeddings dictionary from the file
# with open('term_embeddings_bert.pkl', 'rb') as f:
#     term_embeddings = pickle.load(f)
#
# # Prepare data for t-SNE
# words = list(term_embeddings.keys())
# embeddings = np.array([term_embeddings[term] for term in words])
#
# # Flatten the embeddings
# embeddings = embeddings.reshape(embeddings.shape[0], -1)

# Load the term_embeddings dictionary from the file
with open('word_embeddings_mask_MF_rat_trained_on_abstracts.pkl', 'rb') as f:
    term_embeddings = pickle.load(f)

# Prepare data for t-SNE
words = list(term_embeddings.keys())
embeddings = np.array([term_embeddings[term] for term in words])

# Flatten the embeddings
embeddings = embeddings.reshape(embeddings.shape[0], -1)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Apply UMAP
# reducer = umap.UMAP(n_components=2, random_state=1)
# embeddings_2d = reducer.fit_transform(embeddings)

sse = []
num_clusters_range = range(1, 11)
for k in num_clusters_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings_2d)
    sse.append(kmeans.inertia_)

plt.plot(num_clusters_range, sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

# Choose the number of clusters
num_clusters = 20
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(embeddings_2d)
kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)

# Save the kmeans model to a file
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# # Find the representative word for each cluster
# cluster_representatives = {}
# for cluster_id in range(num_clusters):
#     # Get the mean (centroid) of the 2D embeddings for the current cluster
#     cluster_centroid = np.mean(embeddings_2d[clusters == cluster_id], axis=0)
#
#     # Calculate the Euclidean distance between the centroid and all points in the cluster
#     distances = np.linalg.norm(embeddings_2d[clusters == cluster_id] - cluster_centroid, axis=1)
#
#     # Find the index of the point with the minimum distance
#     closest_point_index = np.argmin(distances)
#
#     # Get the word corresponding to the closest point
#     cluster_word = words[np.where(clusters == cluster_id)[0][closest_point_index]]
#
#     # Save the representative word for the current cluster
#     cluster_representatives[cluster_id] = cluster_word

# Find the representative word for each cluster based on frequency
cluster_representatives = {}
for cluster_id in range(num_clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]

    # Calculate the average distance for each word to all other points in the cluster
    avg_distances = []
    for idx in cluster_indices:
        word_distances = np.linalg.norm(embeddings_2d[cluster_indices] - embeddings_2d[idx], axis=1)
        avg_distance = np.mean(word_distances)
        avg_distances.append(avg_distance)

    # Find the index of the word with the lowest average distance
    best_word_index = np.argmin(avg_distances)

    # Get the word corresponding to the lowest average distance
    cluster_word = words[cluster_indices[best_word_index]]

    # Save the representative word for the current cluster
    cluster_representatives[cluster_id] = cluster_word

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
    cluster_centroid = kmeans.cluster_centers_[cluster_id]
    plt.annotate(cluster_word, xy=(cluster_centroid[0], cluster_centroid[1]), xytext=(-15, 15),
                 textcoords='offset points',
                 ha='center', va='center', fontsize=12, color='red',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='red', facecolor='white', alpha=0.7))

plt.show()

# Save data to a file
data = {
    'embeddings_2d': embeddings_2d,
    'words': words,
    'clusters': clusters,
    'cluster_representatives': cluster_representatives
}

with open('cluster_data.pkl', 'wb') as f:
    pickle.dump(data, f)
