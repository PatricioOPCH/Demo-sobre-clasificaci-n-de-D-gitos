import os

from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
import base64
from PIL import Image
import requests
import json
# Carga el modelo desde el archivo
model = keras.models.load_model('CreacionModelo/mnist_model.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/update', methods=['POST'])
def update():
    # Obtiene la imagen dibujada en el canvas
    image_data = request.form.get('image_data')
    # Elimina el encabezado de la cadena base64
    image_data = image_data.replace('data:image/png;base64,', '')
    # Decodifica la cadena base64 en bytes
    image_bytes = base64.b64decode(image_data)

    # Guarda la imagen en un archivo temporal
    with open('temp.png', 'wb') as f:
        f.write(image_bytes)


    # Carga la imagen desde el archivo temporal
    image = open('temp.png', 'rb')

    # Envía la imagen al endpoint de predicción
    response = requests.post('http://localhost:5000/predict', files={'image': image})
    if response.ok:
        prediction = response.json()['result']
        
    else:
        prediction = 'Error en la respuesta del servidor'

    prediction = json.dumps(prediction)
    # Muestra la predicción como una alerta en el navegador
    print("\n\n\n ", prediction,"\n\n")
    return prediction


# Define una ruta para procesar imágenes
@app.route('/predict', methods=['POST'])
def predict():
    # Obtiene la imagen enviada en la solicitud
    image = Image.open(request.files['image'])
    image = image.resize((28, 28))
    image = image.convert('L')

    # Preprocesa la imagen para que tenga el mismo formato que los datos de entrenamiento
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    # Realiza la predicción utilizando el modelo cargado
    prediction = model.predict(image)

    # Convierte el objeto numpy.ndarray a un objeto int
    # En el servidor Flask
    result = int(prediction[0].argmax())
    
    return jsonify({'result': result})






if __name__ == '__main__':
    app.run(debug=True)
