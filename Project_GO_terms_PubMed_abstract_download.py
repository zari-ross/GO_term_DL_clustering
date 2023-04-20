import json
import time
from Bio import Entrez
from tqdm import tqdm

# Set your email address (required by NCBI)
Entrez.email = "NITrushina@hotmail.com"

# Set your API key (optional but recommended)
Entrez.api_key = "74baba2f0a49e12f5ce2cb11159c86fbf609"

# Define the search query
query = "\"gene function\""

# Search PubMed for the query and get the total number of results
search_handle = Entrez.esearch(db="pubmed", term=query, retmax=100000)
search_results = Entrez.read(search_handle)
search_handle.close()
total_results = int(search_results["Count"])
print(f"Total number of results: {total_results}")

# Get the list of PubMed IDs
pubmed_ids = search_results["IdList"]

# Fetch the abstracts using the list of PubMed IDs
abstracts = []
start_time = time.time()
for pubmed_id in tqdm(pubmed_ids, desc="Fetching abstracts"):
    try:
        fetch_handle = Entrez.efetch(db="pubmed", id=pubmed_id, retmode="xml")
        fetch_data = Entrez.read(fetch_handle)
        fetch_handle.close()

        # Extract the abstract text
        abstract = fetch_data['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
        abstracts.append(str(abstract))
    except IndexError:
        # Skip if the abstract is not available
        pass
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred while fetching abstract for PubMed ID {pubmed_id}: {e}")

    time.sleep(0.05)

# Save the abstracts to a file
with open("abstracts.json", "w") as f:
    json.dump(abstracts, f)

# Print the elapsed time and number of downloaded abstracts
end_time = time.time()
num_abstracts = len(abstracts)
print(f"Downloaded {num_abstracts} abstracts in {end_time - start_time:.2f} seconds")
