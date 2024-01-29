import os
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model(r'disease.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(img):
    img = img.resize((225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    img = Image.open(file.stream)
    predictions = getResult(img)
    predicted_label = labels[np.argmax(predictions)]
    return str(predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
