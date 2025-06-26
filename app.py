from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from flask_ngrok import run_with_ngrok # Import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app) # Use run_with_ngrok to create a public URL

# Load your trained model (ensure the file is in the same directory or provide the full path)
model = load_model("fresh_rotten_model.h5")

# Define class labels (make sure these match your training)
class_labels = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load and preprocess image
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((128, 128))  # resize based on model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        confidence = prediction[predicted_index] * 100

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run() # Remove debug=True here when using flask-ngrok
