import pronto
import re


# Load OBO file
file_path = "go-basic.obo"
ontology = pronto.Ontology(file_path)

# Print some information about the ontology
print(f"Number of terms: {len(ontology)}")

# # Access a specific term using its ID
# term_id = "GO:0008150"  # Replace with your term ID
# term = ontology[term_id]
#
# # Print information about the term
# print(f"ID: {term.id}")
# print(f"Name: {term.name}")
# print(f"Definition: {term.definition}")
# print(f"Subsets: {term.subsets}")

# Initialize an empty list for storing definitions
all_definitions = []

# Iterate through all terms in the ontology
for term_id, term in ontology.items():
    # Get the term's definition
    definition = term.definition

    # Check if the definition is not empty
    if definition:
        # Add term definition to the list
        all_definitions.append(str(definition))

# Check the result
print(all_definitions[:5])  # Print the first 5 definitions

filtered_definitions = [definition for definition in all_definitions if "obsolete" not in definition.lower() and "unknown" not in definition.lower() and "uncharacterized" not in definition.lower()]

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')


def is_verb(word):
    if not word:
        return False
    pos = pos_tag([word])[0][1]
    return pos.startswith('VB')


def clean_definition(definition):
    # Remove punctuation and special characters
    cleaned = definition.translate(str.maketrans('', '', string.punctuation))

    # Convert all characters to lowercase
    cleaned = cleaned.lower()

    # Remove stopwords
    words = nltk.word_tokenize(cleaned)
    words = [word for word in words if word not in stop_words]

    # Remove verbs
    words = [word for word in words if not is_verb(word)]

    # Perform lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    # Reconstruct cleaned definition
    cleaned = ' '.join(words)

    return cleaned


# Clean all filtered definitions
cleaned_definitions = [clean_definition(definition) for definition in filtered_definitions]

# Check the result
print(cleaned_definitions[:5])

import json

output_file = "cleaned_definitions.json"

with open(output_file, "w") as f:
    json.dump(cleaned_definitions, f)

