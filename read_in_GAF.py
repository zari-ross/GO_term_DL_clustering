import pandas as pd
import pronto
import json

from Project.Project_GO_term_cleaning_names import clean_name

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

# Print the first few rows of the DataFrame
# print(gaf_df.head())

# Extract unique GO terms from the DataFrame
unique_go_terms = gaf_df['GO_ID'].unique()

# Convert the result to a list
unique_go_terms_list = list(unique_go_terms)

# Print the list of unique GO terms
# print(unique_go_terms_list)

go_ids = unique_go_terms_list

# Load OBO file
file_path = "go-basic.obo"
ontology = pronto.Ontology(file_path)

# # Get term names for the GO IDs in your experiment
# term_names = [ontology[go_id].name for go_id in go_ids if go_id in ontology]

# Get term names for the GO IDs in your experiment with namespace "molecular_function"
term_names = [ontology[go_id].name for go_id in go_ids if
              go_id in ontology and ontology[go_id].namespace == "molecular_function"]

filtered_names = [name for name in term_names if "obsolete" not in name.lower() and "unknown" not in name.lower() and
                  "uncharacterized" not in name.lower()]

# Clean all filtered names
cleaned_names = [clean_name(name) for name in filtered_names]

# print(cleaned_names)
# print(len(cleaned_names))

output_file = "cleaned_names_MF_rat.json"

with open(output_file, "w") as f:
    json.dump(cleaned_names, f)
