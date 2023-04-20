import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import json

# Load the name-embedding dictionary
with open("name_embeddings_2d.json", "r") as f:
    name_embedding_dict = json.load(f)

# Convert the embeddings back to numpy arrays
for name in name_embedding_dict:
    name_embedding_dict[name] = np.array(name_embedding_dict[name])

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot()

# Load the embeddings
X_embed = np.load('embeddings_2d.npy')

# 1. Dimension der Wort-Vektoren auf X-Achse,
# 2. Dimension auf y-Achse, 3. auf die Z-Achse abbilden
ax.scatter(X_embed[:, 0], X_embed[:, 1])

num_names_to_show = 100

for i, name in enumerate(name_embedding_dict.keys()):
    if i >= num_names_to_show:
        break
    embedding = name_embedding_dict[name]
    ax.text(embedding[0], embedding[1], name)


ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')

plt.show()
