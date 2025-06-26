import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import os

st.title("🍎 Fresh or Rotten Fruit/Veggie Classifier")
st.write("Upload an image and I will tell you if it's **Fresh** or **Rotten**.")

# Auto-download model from GitHub (optional if model is large)
MODEL_PATH = "fresh_rotten_model.h5"
MODEL_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/fresh_rotten_model.h5"  # Replace this!

if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'wb') as f:
        f.write(requests.get(MODEL_URL).content)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Fresh 🍏" if prediction < 0.5 else "Rotten 🥀"
    confidence = 100 * (1 - prediction if prediction < 0.5 else prediction)

    st.subheader("🔍 Prediction:")
    st.success(f"**{label}** (Confidence: {confidence:.2f}%)")
