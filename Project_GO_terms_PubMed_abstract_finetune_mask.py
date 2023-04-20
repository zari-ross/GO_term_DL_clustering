import json

with open("cleaned_abstracts.json", "r") as f:
    cleaned_abstracts = json.load(f)

vectorizer.adapt(cleaned_abstracts)