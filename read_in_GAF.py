import pandas as pd

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
print(gaf_df.head())
