from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

# Carregue o modelo e outros dados necessários
model = joblib.load("LR.joblib")
train = pd.read_csv("X_train.csv", usecols=range(1, 7)).to_numpy()
class_names = model.classes_
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=None, class_names=class_names, discretize_continuous=True)


def predict_and_explain(sex, redo, cpb, age, bsa, hb):
    sex = 1 if sex == "Male" else 0
    redo = 1 if redo == "Yes" else 0
    cpb = 1 if cpb == "Yes" else 0
    instance = [sex, age, bsa, redo, cpb, hb]
    prediction = model.predict([instance])
    exp = explainer.explain_instance(np.array(instance), model.predict_proba, num_features=6)

    # Converte a explicação em uma imagem
    explanation_image = exp.as_pyplot_figure()

    # Salva a imagem em um buffer de bytes
    image_buffer = io.BytesIO()
    explanation_image.savefig(image_buffer, format="png")
    image_buffer.seek(0)

    return prediction, image_buffer.read()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Suponha que você está enviando os dados JSON para a API

        sex = data["sex"]
        redo = data["redo"]
        cpb = data["cpb"]
        age = data["age"]
        bsa = data["bsa"]
        hb = data["hb"]

        prediction_result, explanation_image = predict_and_explain(sex, redo, cpb, age, bsa, hb)

        result = {
            "prediction": prediction_result[0],  # Se o modelo retornar uma matriz, pegue o primeiro elemento
        }

        # Crie um gráfico da explicação usando Matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(explanation_image)
        plt.axis('off')  # Desative os eixos
        plt.tight_layout()

        # Salve o gráfico em um buffer de bytes
        graph_buffer = io.BytesIO()
        plt.savefig(graph_buffer, format="png")
        graph_buffer.seek(0)

        return jsonify(result), 200, {'Content-Type': 'image/png'}, graph_buffer.read()

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Retorna um erro 500 com uma mensagem de erro JSON

if __name__ == "__main__":
    app.run(debug=True)
