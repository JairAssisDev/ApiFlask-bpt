import io
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import base64
import pandas as pd
from PIL import Image


# Carregue o modelo e outros dados necessários
model = joblib.load("LR.joblib")
train = pd.read_csv("X_train.csv", usecols=range(1, 7)).to_numpy()
feature_names = model.feature_names_in_
class_names = model.classes_
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=feature_names, class_names=class_names, discretize_continuous=True)

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

    # Converte a imagem em uma string codificada em base64
    image_base64 = base64.b64encode(image_buffer.read()).decode("utf-8")

    return prediction, image_base64

# Exemplo de uso
sex = "Male"
redo = "Yes"
cpb = "Yes"
age = 1.0
bsa = 1.0
hb = 1.0

prediction_result, explanation_image = predict_and_explain(sex, redo, cpb, age, bsa, hb)

# Converter a imagem de base64 para formato de imagem
image_data = base64.b64decode(explanation_image)
image = Image.open(io.BytesIO(image_data))

# Exibir a previsão e a imagem da explicação
print("Previsão:", prediction_result[0])
plt.imshow(image)
plt.show()
