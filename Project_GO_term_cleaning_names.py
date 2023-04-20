import pronto
import re


# Load OBO file
file_path = "go-basic.obo"
ontology = pronto.Ontology(file_path)

# Print some information about the ontology
# print(f"Number of terms: {len(ontology)}")

# # Access a specific term using its ID
# term_id = "GO:0008150"  # Replace with your term ID
# term = ontology[term_id]
#
# # Print information about the term
# print(f"ID: {term.id}")
# print(f"Name: {term.name}")
# print(f"Definition: {term.definition}")
# print(f"Subsets: {term.subsets}")

# Initialize an empty list for storing names
all_names = []

# Iterate through all terms in the ontology
for term_id, term in ontology.items():
    # Get the term's name
    name = term.name
    # Check if the term belongs to the molecular_function namespace
    if term.namespace == "molecular_function":
        # Get the term's name
        name = term.name
        # Check if the name is not empty
        if name:
            # Add term name to the list
            all_names.append(str(name))

# Check the result
# print(all_names[:5])  # Print the first 5 names

filtered_names = [name for name in all_names if "obsolete" not in name.lower() and "unknown" not in name.lower() and "uncharacterized" not in name.lower()]

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_name(name):
    # Remove punctuation and special characters
    cleaned = name.translate(str.maketrans('', '', string.punctuation))

    # Convert all characters to lowercase
    cleaned = cleaned.lower()

    # Remove stopwords
    words = nltk.word_tokenize(cleaned)
    words = [word for word in words if word not in stop_words]

    # Perform lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    # Reconstruct cleaned name
    cleaned = ' '.join(words)

    return cleaned


# Clean all filtered names
cleaned_names = [clean_name(name) for name in filtered_names]

# Check the result
# print(cleaned_names[:5])
# print(len(cleaned_names))

import json

output_file = "cleaned_names.json"

with open(output_file, "w") as f:
    json.dump(cleaned_names, f)

