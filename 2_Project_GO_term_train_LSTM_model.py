import argparse
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import CSVLogger
import pickle

# Argument parsing
parser = argparse.ArgumentParser(description='Train LSTM model')
parser.add_argument('in_json', type=str, help='Input JSON file with cleaned abstracts')
parser.add_argument('log', type=str, help='Output log file')
parser.add_argument('model', type=str, help='Output model file')
parser.add_argument('pkl', type=str, help='Output pickle file with word embeddings')
args = parser.parse_args()


with open(args.in_json, "r") as f:
    cleaned_abstracts = json.load(f)

vectorizer = tf.keras.layers.TextVectorization(output_sequence_length=7)  # max_tokens=Inf,

# vectorizer.adapt(cleaned_terms)
vectorizer.adapt(cleaned_abstracts)


def generator(text, batch_size=1):
    while True:
        x = vectorizer(text)
        mask = tf.reduce_max(x) + 1

        # Get the indices for a random batch of data
        batch_indices = np.random.choice(len(text), batch_size, replace=False)
        x_batch = tf.gather(x, batch_indices)

        lengths = tf.argmin(x_batch, axis=1)
        lengths = tf.cast(lengths, tf.float32)

        masks = tf.random.uniform(shape=(x_batch.shape[0],), minval=0, maxval=lengths)
        masks = tf.cast(masks, tf.int32)

        masks = tf.one_hot(masks, x_batch.shape[1], dtype=tf.int32)
        masks = tf.cast(masks, tf.bool)

        y_batch = x_batch[masks]
        masks = tf.cast(masks, tf.int64)
        x_batch = x_batch * (1 - masks) + mask * masks
        yield x_batch, y_batch


# Describe model architecture
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(vectorizer.vocabulary_size()+1, 50))

model.add(tf.keras.layers.LSTM(50, return_sequences=False, activation='sigmoid'))
model.add(tf.keras.layers.Dense(vectorizer.vocabulary_size(), activation='softmax'))

# model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=500, min_lr=1e-6)

# Create a CSVLogger callback and specify the filename for the log file
csv_logger = CSVLogger(args.log, append=True)

# model.fit(generator(cleaned_terms, batch_size=50), steps_per_epoch=1, epochs=3000, verbose=2,
#           callbacks=[lr_reduce, csv_logger])
model.fit(generator(cleaned_abstracts, batch_size=50), steps_per_epoch=1, epochs=3000, verbose=2,
          callbacks=[lr_reduce, csv_logger])

# Save the model
model.save(args.model)

# Get the weights from the Embedding layer
embeddings = model.layers[0].get_weights()[0]

# Get the vocabulary from the vectorizer
vocab = vectorizer.get_vocabulary()

# Create a dictionary of word embeddings
word_embeddings = {word: embeddings[idx] for idx, word in enumerate(vocab)}

# Save the word_embeddings dictionary to a file
with open(args.pkl, 'wb') as f:
    pickle.dump(word_embeddings, f)
