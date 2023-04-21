import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import json
import random

# Load the term_embeddings dictionary from the file
with open('rat_cleaned_terms_with_embeddings.json', 'r') as f:
    term_embeddings = json.load(f)

# Print the overall number of embeddings
print(f"Overall number of embeddings: {len(term_embeddings)}")

# Prepare data for t-SNE
words = list(term_embeddings.keys())

# Randomly select 1000 words
random_words = random.sample(words, 4625)  # 1000

embeddings = np.array([term_embeddings[term]['embedding'] for term in random_words])

# Flatten the embeddings
embeddings = embeddings.reshape(embeddings.shape[0], -1)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d_tsne = tsne.fit_transform(embeddings)

# Apply UMAP
# reducer = umap.UMAP(n_components=2, random_state=42,  n_neighbors=100, min_dist=0.5, init='random')
# embeddings_2d_umap = reducer.fit_transform(embeddings)

# Create a side-by-side plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ax1.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], alpha=0.3)
ax1.set_title('t-SNE Plot')

# ax2.scatter(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], alpha=0.3)
# ax2.set_title('UMAP Plot')

plt.show()

# Add t-SNE embeddings to the original dictionary
for i, word in enumerate(random_words):
    term_embeddings[word]['embedding_2d_tsne'] = embeddings_2d_tsne[i].tolist()

# Save the updated dictionary to a file
with open('rat_cleaned_terms_with_embeddings_and_tsne.json', 'w') as f:
    json.dump(term_embeddings, f)
