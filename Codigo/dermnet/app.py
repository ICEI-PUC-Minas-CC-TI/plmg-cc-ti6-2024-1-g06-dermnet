from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image

import numpy as np
import base64
import os


app = Flask(__name__)

model = load_model('model_dermnet.h5')
model.make_predict_function()

dic = [
    "Acne e Rosácea",
    "Queratose Actínica Carcinoma Basocelular e outras Lesões Malignas",
    "Dermatite Atópica",
    "Doença Bolhosa",
    "Celulite Impetigo e outras Infecções Bacterianas",
    "Eczema",
    "Exantemas e Erupções Cutâneas Relacionadas a Medicamentos",
    "Queda de Cabelo Alopecia e outras Doenças Capilares",
    "Herpes HPV e outras DSTs",
    "Doenças Leves e Distúrbios de Pigmentação",
    "Lúpus e outras Doenças do Tecido Conjuntivo",
    "Câncer de Pele Melanoma Nevos e Sardas",
    "Fungo nas Unhas e outras Doenças das Unhas",
    "Herbívora venenosa e outras Dermatites de Contato",
    "Psoríase, Liquen Plano e Doenças Relacionadas",
    "Escabiose Doença de Lyme e outras Infestações e Picadas",
    "Ceratoses Seborreicas e outros Tumores Benignos",
    "Doença Sistêmica",
    "Tinha, Candidíase e outras Infecções Fúngicas",
    "Urticária",
    "Tumores Vasculares",
    "Vasculite",
    "Verrugas Molluscum e outras Infecções Virais"
]


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST' and 'imageFile' in request.files:
        img = request.files['imageFile']
        img_path = './uploads/' + img.filename
        img.save(img_path)
        print(img)
        i = image.load_img(img_path, target_size=(224,224))
        i = image.img_to_array(i)
        i = np.expand_dims(i, axis=0)
        i = preprocess_input(i)
        p = model.predict(i)[0]
        with open(img_path, "rb") as imagefile:
            convert = base64.b64encode(imagefile.read()).decode('utf-8')
        os.remove(img_path)
       
        return render_template("index.html", img_base64 = convert, prediction= dic[np.argmax(p)])     
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(port=3000, debug=True)

