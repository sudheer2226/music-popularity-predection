from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    try:
        user_input = request.form.to_dict()

        # Convert input into DataFrame
        input_df = pd.DataFrame(columns=columns)
        input_df.loc[0] = 0  # default all features = 0

        # Fill only provided values
        for key, value in user_input.items():
            if key in input_df.columns:
                input_df[key] = float(value)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        result = "🔥 Popular Song" if prediction == 1 else "😐 Not Popular Song"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)