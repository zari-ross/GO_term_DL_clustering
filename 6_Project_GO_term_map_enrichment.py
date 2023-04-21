import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from Project.Step5_Project_GO_term_cluster import num_clusters
num_clusters = 40
from collections import Counter
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def custom_formatter(x, pos):
    return f'{x / 100}'


formatter = FuncFormatter(custom_formatter)

# Load the term_embeddings dictionary from the file
with open('rat_cleaned_terms_with_embeddings_clusters_and_tsne.json', 'r') as f:
    term_embeddings = json.load(f)

# Extract the t-SNE embeddings
embeddings_2d = np.array([term_embeddings[term]['embedding_2d_tsne'] for term in term_embeddings])
go_ids = list(term_embeddings.keys())

# Extract clusters
clusters = [term_embeddings[term]['cluster'] for term in term_embeddings]

# Create a reverse mapping from cluster ID to the corresponding term ID
cluster_to_term_id = {}
for term_id, cluster in zip(term_embeddings.keys(), clusters):
    cluster_to_term_id[int(cluster)] = term_id

# Extract cluster_representatives
cluster_representatives = {i: term_embeddings[cluster_to_term_id[i]]['representative'] for i in range(num_clusters)}
print(cluster_representatives)

filename = "GO_terms_Example1.txt"

# Read the file into a DataFrame
df = pd.read_csv(filename, sep='\s+', engine='python')

# Extract the GeneGroup column as a list of GO IDs
experiment_go_ids = df['GeneGroup'].tolist()

# Create a list to store experiment words for each cluster
experiment_go_ids_per_cluster = {cluster_id: [] for cluster_id in range(num_clusters)}

for go_id, cluster_id in zip(go_ids, clusters):
    if go_id in experiment_go_ids:
        experiment_go_ids_per_cluster[cluster_id].append(go_id)

all_terms_in_cluster = {}
for cluster_id in range(num_clusters):
    all_terms_in_cluster[cluster_id] = clusters.count(cluster_id)

# Calculate the enrichment score for each cluster
enrichment_scores = {}
for cluster_id in range(num_clusters):
    cluster_word = cluster_representatives[cluster_id]
    experiment_prop = len(experiment_go_ids_per_cluster[cluster_id]) / len(experiment_go_ids)
    all_prop = all_terms_in_cluster[cluster_id] / len(go_ids)
    enrichment_score = experiment_prop / all_prop
    experiment_count = len(experiment_go_ids_per_cluster[cluster_id])
    enrichment_scores[cluster_id] = {'representative': cluster_word, 'enrichment_score': enrichment_score,
                                     'experiment_go_id_count': experiment_count,
                                     'experiment_go_ids': ', '.join(experiment_go_ids_per_cluster[cluster_id])}

# Create a DataFrame from the enrichment scores dictionary
enrichment_scores_df = pd.DataFrame(enrichment_scores).T.reset_index().rename(columns={'index': 'cluster_id'})

# Save the DataFrame to a CSV file
enrichment_scores_df.to_csv('enrichment_scores.csv', index=False)

# Plot the enrichment scores
fig, ax = plt.subplots(figsize=(12, 8))

# Get the data for plotting
y = list(enrichment_scores.keys())
x = [v['enrichment_score'] for v in enrichment_scores.values()]
sizes = [v['experiment_go_id_count'] * 100 for v in enrichment_scores.values()]
labels = [v['representative'] for v in enrichment_scores.values()]

# Map the bubble sizes to colors using the viridis colormap
norm = plt.Normalize(min(sizes), max(sizes))
colors = cm.viridis(norm(sizes))

# Create a scatter plot with variable bubble sizes
scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.5)

# Set the axis labels and title
ax.set_xlabel("Enrichment Score")
ax.set_ylabel("Cluster ID")
ax.set_title("Cluster Enrichment")

# Set the y ticks to cluster representatives
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)

# Add a colorbar as a legend
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
cbar.ax.yaxis.set_major_formatter(formatter)  # Use custom formatter for colorbar values
cbar.set_label("Size proportional to number of words from experiment")

plt.savefig('cluster_enrichment.png')
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

# Extract 2D embeddings for experiment_go_ids
experiment_go_ids_2d = np.array([term_embeddings[go_id]['embedding_2d_tsne'] for go_id in experiment_go_ids if go_id in term_embeddings])  # Add this line

# Plot the experiment_go_ids_2d points in red
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='red', marker='x')  # Change variable to experiment_go_ids_2d

# Add black labels for the experiment_go_ids_2d
for i, go_id in enumerate(experiment_go_ids):
    if go_id in term_embeddings:  # Add this condition
        x, y = experiment_go_ids_2d[i, :]
        plt.annotate(go_id, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', color='black')

plt.show()

plt.figure(figsize=(10, 10))

# Create the 2D kernel density estimation plot
sns.kdeplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=clusters, palette='viridis', common_norm=False)

# Add cluster representative labels
for cluster_id, cluster_word in cluster_representatives.items():
    cluster_centroid = np.mean(embeddings_2d[clusters == cluster_id], axis=0)
    plt.annotate(cluster_word, xy=(cluster_centroid[0], cluster_centroid[1]), xytext=(-15, 15),
                 textcoords='offset points',
                 ha='center', va='center', fontsize=12, color='red',
                 bbox=dict(boxstyle='round, pad=0.5', edgecolor='red', facecolor='white', alpha=0.7))

# Plot the experiment_go_ids_2d points in red
plt.scatter(experiment_go_ids_2d[:, 0], experiment_go_ids_2d[:, 1], c='red', marker='x')  # Change variable to experiment_go_ids_2d

# Add black labels for the experiment_go_ids_2d
for i, go_id in enumerate(experiment_go_ids):
    if go_id in term_embeddings:  # Add this condition
        x, y = experiment_go_ids_2d[i, :]
        plt.annotate(go_id, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', color='black')

plt.savefig('cluster_visualization.png')
plt.show()
