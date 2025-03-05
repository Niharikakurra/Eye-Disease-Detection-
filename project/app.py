import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create 'uploads/' inside 'static/' if it doesn't exist

BACKGROUND_IMAGE = "static/eye disease.jpeg"


# Load the trained model
MODEL_PATH = "evgg.h5"
model = load_model(MODEL_PATH)

# Preprocess the image before passing it to the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/inp')
def inp():
    return render_template('img_input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']

    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)  # Save the image in the "uploads" folder
    
    # Preprocess and make prediction
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)

    classes = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
    predicted_class = classes[np.argmax(predictions)]
    
    return render_template('output.html', prediction=predicted_class, img_path=url_for('static', filename='uploads/' + file.filename))

if __name__ == '__main__':
    app.run(debug=True)

