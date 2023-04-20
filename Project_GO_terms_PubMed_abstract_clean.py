import json

from Project.Project_GO_term_cleaning_names import clean_name

with open("abstracts.json", "r") as f:
    abstracts = json.load(f)

cleaned_abstracts = [clean_name(abstract) for abstract in abstracts]

with open("cleaned_abstracts.json", "w") as f:
    json.dump(cleaned_abstracts, f)
