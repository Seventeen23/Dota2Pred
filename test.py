import requests
import pandas as pd
from tqdm import tqdm

def get_pro_matches(n=10):
    matches = []
    url = "https://api.opendota.com/api/proMatches"
    
    while len(matches) < n:
        r = requests.get(url)
        data = r.json()
        matches.extend(data)
        last_match_id = data[-1]['match_id']
        url = f"https://api.opendota.com/api/proMatches?less_than_match_id={last_match_id}"
    
    return matches[:n]

# Get 500 matches
matches = get_pro_matches(10)
df = pd.DataFrame(matches)
print(df.head())


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

X, y = [], []

for match_id in tqdm(df["match_id"]):
    details = requests.get(f"https://api.opendota.com/api/matches/{match_id}").json()
    if "picks_bans" not in details or details["picks_bans"] is None:
        continue
    X.append(create_draft_vector(details["picks_bans"]))
    y.append(1 if details["radiant_win"] else 0)

import numpy as np
X = np.array(X)
y = np.array(y)

print("Dataset size:", X.shape, len(y))


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))


import joblib
joblib.dump(model, "testmodel.pkl")
joblib.dump(hero_ids, "hero_ids.pkl")


example_draft = [
    {"is_pick": True, "team": 0, "hero_id": 1},
    {"is_pick": True, "team": 0, "hero_id": 2},
    {"is_pick": True, "team": 0, "hero_id": 3},
    {"is_pick": True, "team": 0, "hero_id": 4},
    {"is_pick": True, "team": 0, "hero_id": 5},
    {"is_pick": True, "team": 1, "hero_id": 6},
    {"is_pick": True, "team": 1, "hero_id": 7},
    {"is_pick": True, "team": 1, "hero_id": 8},
    {"is_pick": True, "team": 1, "hero_id": 9},
    {"is_pick": True, "team": 1, "hero_id": 10},
]

vec = np.array([create_draft_vector(example_draft)])
prob = model.predict_proba(vec)[0][1]
print(f"Radiant win probability: {prob:.2%}")
