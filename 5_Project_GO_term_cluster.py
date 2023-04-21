import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import random
import json
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# Load the term_embeddings dictionary from the file
with open('rat_cleaned_terms_with_embeddings_and_tsne.json', 'r') as f:
    term_embeddings = json.load(f)

# Extract the t-SNE embeddings
embeddings_2d = np.array([term_embeddings[term]['embedding_2d_tsne'] for term in term_embeddings])
cleaned_terms = [term_embeddings[term_id]['name'] for term_id in term_embeddings]

# # Initialize the DBSCAN algorithm
# dbscan = DBSCAN(eps=3.5, min_samples=15)
#
# # Fit the DBSCAN algorithm to the t-SNE embeddings
# dbscan.fit(embeddings_2d)
#
# # Calculate the number of clusters
# n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
#
# # Print the results
# print(f"Number of clusters: {n_clusters}")
#
# # Create a scatter plot of the t-SNE embeddings
# colors = plt.cm.viridis((dbscan.labels_ + 1) / n_clusters)
# colors[dbscan.labels_ == -1] = (0, 0, 0, 1)  # Set the color of outliers to black
#
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.5)
#
# # Add a colorbar
# cbar = plt.colorbar()
# cbar.set_label('Cluster Label')
#
# # Set the title and axis labels
# plt.title(f"t-SNE Plot with {n_clusters} Clusters (DBSCAN)")
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
#
# # Display the plot
# plt.show()

# Choose the number of clusters
num_clusters = 40
kmeans = KMeans(n_clusters=num_clusters)
clusters = kmeans.fit_predict(embeddings_2d)
kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)

# Find the representative word for each cluster
cluster_representatives = {}
for cluster_id in range(num_clusters):
    # Get the mean (centroid) of the 2D embeddings for the current cluster
    cluster_centroid = np.mean(embeddings_2d[clusters == cluster_id], axis=0)

    # Calculate the Euclidean distance between the centroid and all points in the cluster
    distances = np.linalg.norm(embeddings_2d[clusters == cluster_id] - cluster_centroid, axis=1)

    # Find the index of the point with the minimum distance
    closest_point_index = np.argmin(distances)

    # Get the word corresponding to the closest point
    cluster_word = cleaned_terms[np.where(clusters == cluster_id)[0][closest_point_index]]

    # Save the representative word for the current cluster
    cluster_representatives[cluster_id] = cluster_word

# # Find the representative word for each cluster based on frequency
# cluster_representatives = {}
# for cluster_id in range(num_clusters):
#     cluster_indices = np.where(clusters == cluster_id)[0]
#
#     # Calculate the average distance for each word to all other points in the cluster
#     avg_distances = []
#     for idx in cluster_indices:
#         word_distances = np.linalg.norm(embeddings_2d[cluster_indices] - embeddings_2d[idx], axis=1)
#         avg_distance = np.mean(word_distances)
#         avg_distances.append(avg_distance)
#
#     # Find the index of the word with the lowest average distance
#     best_word_index = np.argmin(avg_distances)
#
#     # Get the word corresponding to the lowest average distance
#     cluster_word = cleaned_terms[cluster_indices[best_word_index]]
#
#     # Save the representative word for the current cluster
#     cluster_representatives[cluster_id] = cluster_word

# # Select 150 random words
selected_words = random.sample(cleaned_terms, 50)
selected_indices = [cleaned_terms.index(word) for word in selected_words]

# Plot the 2D embeddings for all words
plt.figure(figsize=(20, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis')

# Add labels for the selected words
for i in selected_indices:
    x, y = embeddings_2d[i, :]
    plt.annotate(cleaned_terms[i], xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

# Add cluster representative labels
for cluster_id, cluster_word in cluster_representatives.items():
    cluster_centroid = np.mean(embeddings_2d[clusters == cluster_id], axis=0)
    plt.annotate(cluster_word, xy=(cluster_centroid[0], cluster_centroid[1]), xytext=(-15, 15),
                 textcoords='offset points',
                 ha='center', va='center', fontsize=8, color='red',
                 bbox=dict(boxstyle='round, pad=0.1', edgecolor='red', facecolor='white', alpha=0.7))

plt.title("Random words in black")
plt.show()

# Create a reverse mapping from cluster ID to the corresponding term ID
cluster_to_term_id = {}
for term_id, term_name, cluster in zip(term_embeddings.keys(), cleaned_terms, clusters):
    term_embeddings[term_id]['cluster'] = int(cluster)

# Add the cluster representatives to the term_embeddings dictionary
for cluster_id, cluster_word in cluster_representatives.items():
    representative_term_index = cleaned_terms.index(cluster_word)
    representative_term_id = list(term_embeddings.keys())[representative_term_index]
    cluster_to_term_id[cluster_id] = representative_term_id
    term_embeddings[representative_term_id]['representative'] = cluster_word

# Print the cluster representatives
# print(cluster_representatives)

# Save the updated term_embeddings dictionary to a file
with open('rat_cleaned_terms_with_embeddings_clusters_and_tsne.json', 'w') as f:
    json.dump(term_embeddings, f)
