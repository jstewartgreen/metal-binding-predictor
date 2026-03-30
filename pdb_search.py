import json
import requests

QUERY_PATH = "queries/pdb_query.json"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

with open(QUERY_PATH, "r") as f:
    query = json.load(f)

response = requests.post(SEARCH_URL, json=query)
response.raise_for_status()

results = response.json()["result_set"]
pdb_ids = [r["identifier"] for r in results]

with open("data/pdb_ids.txt", "w") as out:
    out.write("\n".join(pdb_ids))

print(f"Retrieved {len(pdb_ids)} PDB IDs")
