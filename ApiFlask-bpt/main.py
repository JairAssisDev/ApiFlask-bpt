from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Carregue o modelo e outros dados necessários
model = joblib.load("LR.joblib")

def predict(sex, redo, cpb, age, bsa, hb):
    sex = 1 if sex == "Male" else 0
    redo = 1 if redo == "Yes" else 0
    cpb = 1 if cpb == "Yes" else 0
    instance = [sex, age, bsa, redo, cpb, hb]
    prediction = model.predict([instance])
    return bool(prediction[0])

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.json  # Suponha que você está enviando os dados JSON para a API

        sex = data["sex"]
        redo = data["redo"]
        cpb = data["cpb"]
        age = data["age"]
        bsa = data["bsa"]
        hb = data["hb"]

        prediction_result = predict(sex, redo, cpb, age, bsa, hb)

        result = {
            "prediction": prediction_result
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Retorna um erro 500 com uma mensagem de erro JSON

if __name__ == "__main__":
    app.run(debug=True)
