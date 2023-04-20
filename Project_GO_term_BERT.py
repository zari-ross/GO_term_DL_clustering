import pickle
import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

input_file = "cleaned_names_MF_rat.json"

with open(input_file, "r") as f:
    cleaned_terms = json.load(f)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
model = TFBertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1', from_pt=True)


# Function to get embeddings for a single term
def get_term_embedding(term, tokenizer, model):
    # Tokenize the term
    inputs = tokenizer(term, return_tensors="tf", padding=True, truncation=True, max_length=128)

    # Get the output from the BERT model
    outputs = model(inputs)

    # Get the pooled output (the [CLS] token's output)
    embedding = outputs.pooler_output.numpy()

    return embedding


# Get embeddings for all cleaned_terms
term_embeddings = {}

# GO term will be represented by a fixed-size vector
# (768-dimensional in the case of 'dmis-lab/biobert-base-cased-v1.1').
for term in cleaned_terms:
    term_embeddings[term] = get_term_embedding(term, tokenizer, model)

# Save the term_embeddings dictionary to a file
with open('term_embeddings_bert.pkl', 'wb') as f:
    pickle.dump(term_embeddings, f)
