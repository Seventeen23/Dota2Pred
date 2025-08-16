import requests
import pandas as pd
import json
from tqdm import tqdm
import os

DATA_FILE = "pro_matches_10k.json"

def get_pro_matches(n=10000, save_path=DATA_FILE):
    """Fetch pro matches from OpenDota and save to local file."""
    matches = []
    url = "https://api.opendota.com/api/proMatches"
    
    while len(matches) < n:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"Error: {r.status_code}")
            break

        data = r.json()
        matches.extend(data)
        last_match_id = data[-1]['match_id']
        url = f"https://api.opendota.com/api/proMatches?less_than_match_id={last_match_id}"

        print(f"Fetched {len(matches)} matches...")

    # Save to local JSON
    with open(save_path, "w") as f:
        json.dump(matches[:n], f)
    print(f"Saved {len(matches[:n])} matches to {save_path}")


# Call the the fetch hehe
get_pro_matches()