from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model + hero IDs
model = joblib.load("model.pkl")
hero_ids = joblib.load("hero_ids.pkl")

def create_draft_vector(picks_bans):
    vec = {hid: 0 for hid in hero_ids}
    for pb in picks_bans:
        if pb.get("is_pick", True):
            if pb["team"] == 0:
                vec[pb["hero_id"]] = 1
            else:
                vec[pb["hero_id"]] = -1
    return list(vec.values())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    draft_vector = np.array([create_draft_vector(data["picks"])])
    prob = model.predict_proba(draft_vector)[0][1]
    return jsonify({"radiant_win_probability": float(prob)})

if __name__ == "__main__":
    app.run(debug=True)
