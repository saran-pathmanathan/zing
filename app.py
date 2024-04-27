from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

# Define the unique values for income_group seen during training
income_group_categories = ["LOW", "LOWMID", "UPMID", "HIGH"]


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert the dictionary to dataframe
    input_data = pd.DataFrame([data])

    # One-hot encoding
    input_data = pd.get_dummies(input_data)

    # Add the missing columns with 0s
    for category in income_group_categories:
        if f"income_group_{category}" not in input_data.columns:
            input_data[f"income_group_{category}"] = 0

    # Make prediction using the saved model
    prediction = model.predict(input_data)
    rounded_prediction = round(prediction[0], 3)
    # Send back the result as json
    return jsonify([rounded_prediction])


if __name__ == "__main__":
    # Use the PORT environment variable if it's set, otherwise default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
