import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
import pronto
import json

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

file_path = "go-basic.obo"
ontology = pronto.Ontology(file_path)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_name(name):
    cleaned = name.translate(str.maketrans('', '', string.punctuation))
    cleaned = cleaned.lower()
    words = nltk.word_tokenize(cleaned)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned = ' '.join(words)
    return cleaned


# Define column names for the GAF file
column_names = [
    "DB",
    "DB_Object_ID",
    "DB_Object_Symbol",
    "Qualifier",
    "GO_ID",
    "DB:Reference",
    "Evidence_Code",
    "With_or_From",
    "Aspect",
    "DB_Object_Name",
    "DB_Object_Synonym",
    "DB_Object_Type",
    "Taxon",
    "Date",
    "Assigned_By",
    "Annotation_Extension",
    "Gene_Product_Form_ID",
]

# Load GAF file into a pandas DataFrame
file_path = "rgd.gaf"
gaf_df = pd.read_csv(file_path, sep="\t", comment="!", names=column_names, skiprows=12)

# Extract unique GO terms from the DataFrame
unique_go_terms = gaf_df['GO_ID'].unique()

# Convert the result to a list
unique_go_terms_list = list(unique_go_terms)

# Get term names and cleaned names for rat-specific molecular function GO terms
all_term_names = [ontology[go_id].name for go_id in unique_go_terms_list if
                  go_id in ontology and ontology[go_id].namespace == "molecular_function"]

all_terms = {}

for term_id in unique_go_terms_list:
    if term_id.startswith("GO:") and term_id in ontology:
        term = ontology[term_id]
        if term.namespace == "molecular_function":
            name = term.name
            if name and "obsolete" not in name.lower() and "unknown" not in name.lower() \
                    and "uncharacterized" not in name.lower():
                all_terms[term_id] = {'name': str(name), 'cleaned_name': clean_name(name)}

output_file = "rat_cleaned_terms.json"

with open(output_file, "w") as f:
    json.dump(all_terms, f)

# Print the first 10 elements of rat_terms
for i, (term_id, term_info) in enumerate(all_terms.items()):
    if i >= 10:
        break
    print(f"{term_id}: {term_info}")


with open("abstracts.json", "r") as f:
    abstracts = json.load(f)

cleaned_abstracts = [clean_name(abstract) for abstract in abstracts]

with open("cleaned_abstracts.json", "w") as f:
    json.dump(cleaned_abstracts, f)
