import requests
import pandas as pd
import json
from tqdm import tqdm
import os

DATA_FILE = "pro_matches_10k.json"

def load_matches(save_path=DATA_FILE):
    """Load locally saved matches."""
    if not os.path.exists(save_path):
        print("No local file found. Please fetch data first.")
        return None
    with open(save_path, "r") as f:
        matches = json.load(f)
    return matches

# Step 1: Fetch and save (run this once)
# get_pro_matches()

# Step 2: Load from local file
matches = load_matches()
if matches is None:
    exit()

df = pd.DataFrame(matches)
print(df.head())

# Step 3: Continue with your hero + training code
heroes = requests.get("https://api.opendota.com/api/heroes").json()
hero_ids = sorted([h["id"] for h in heroes])
print(f"Total heroes: {len(hero_ids)}")

def create_draft_vector(picks_bans):
    vec = {hid: 0 for hid in hero_ids}
    for pb in picks_bans:
        if pb.get("is_pick", True):
            if pb["team"] == 0:  # Radiant
                vec[pb["hero_id"]] = 1
            else:  # Dire
                vec[pb["hero_id"]] = -1
    return list(vec.values())

import numpy as np
X, y = [], []

for match_id in tqdm(df["match_id"]):
    details = requests.get(f"https://api.opendota.com/api/matches/{match_id}").json()
    if "picks_bans" not in details or details["picks_bans"] is None:
        continue
    X.append(create_draft_vector(details["picks_bans"]))
    y.append(1 if details["radiant_win"] else 0)

X = np.array(X)
y = np.array(y)
print("Dataset size:", X.shape, len(y))

# Train the model
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model and hero IDs
import joblib
joblib.dump(model, "testmodel.pkl")
joblib.dump(hero_ids, "hero_ids.pkl")
