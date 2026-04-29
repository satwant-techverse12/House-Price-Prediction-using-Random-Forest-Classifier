from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model & column structure
model = joblib.load("model/house_price_model.pkl")
scaler = joblib.load("model/scaler.save")  # keep if used during training
columns = joblib.load("model/columns.pkl")  # IMPORTANT

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Get inputs
        data = {
            "area": float(request.form["area"]),
            "bedrooms": int(request.form["bedrooms"]),
            "bathrooms": int(request.form["bathrooms"]),
            "stories": int(request.form["stories"]),
            "mainroad": int(request.form["mainroad"]),
            "guestroom": int(request.form["guestroom"]),
            "basement": int(request.form["basement"]),
            "hotwaterheating": int(request.form["hotwaterheating"]),
            "airconditioning": int(request.form["airconditioning"]),
            "parking": int(request.form["parking"]),
            "prefarea": int(request.form["prefarea"]),
        }

        furnishingstatus = request.form["furnishingstatus"]

        # ✅ One-hot encoding (REPLACE LabelEncoder)
        data["furnishingstatus_semi-furnished"] = 1 if furnishingstatus == "semi-furnished" else 0
        data["furnishingstatus_unfurnished"] = 1 if furnishingstatus == "unfurnished" else 0

        # Prepare input safely
        input_data = [0] * len(columns)

        for key in data:
            if key in columns:
                index = columns.index(key)
                input_data[index] = data[key]

        features = np.array([input_data])

        # Scale if required
        try:
            features = scaler.transform(features)
        except:
            pass  # ignore if scaler not needed

        # Predict
        prediction = model.predict(features)[0]

    return render_template("index.html", prediction=prediction)


# ✅ REST API (IMPORTANT FOR JOB)
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json

    # One-hot encoding
    data["furnishingstatus_semi-furnished"] = 1 if data["furnishingstatus"] == "semi-furnished" else 0
    data["furnishingstatus_unfurnished"] = 1 if data["furnishingstatus"] == "unfurnished" else 0

    # Prepare input
    input_data = [0] * len(columns)

    for key in data:
        if key in columns:
            index = columns.index(key)
            input_data[index] = data[key]

    features = np.array([input_data])

    try:
        features = scaler.transform(features)
    except:
        pass

    prediction = model.predict(features)[0]

    return jsonify({
        "predicted_price": float(prediction)
    })


# ✅ Production-ready run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    



if not os.path.exists("model/house_price_model.pkl"):
    print("Training model...")
    import train_model