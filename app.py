import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from pathlib import Path

# Add a background color to the app and update button styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5dc;  
    }
    .stButton>button {
        background-color: #8B4513;  
        color: white;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to download and load the model
@st.cache_resource
def download_and_load_model():
    model_url = "https://drive.google.com/uc?export=download&id=1iO013Rqp0dOlHHoFuQsNQEWsBS2f11mj"
    model_path = Path("cnn_model.tflite")
    
    if not model_path.exists():
        with st.spinner("Downloading the model..."):
            response = requests.get(model_url)
            model_path.write_bytes(response.content)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter

# Load the TFLite model
model = download_and_load_model()

# Define classes as per training
classes = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Function to preprocess image and make prediction
def predict(image):
    input_details = model.get_input_details()
    input_shape = input_details[0]['shape']

    image = image.resize((input_shape[1], input_shape[2]))  
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  

    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    prediction = model.get_tensor(model.get_output_details()[0]['index'])
    return prediction

# App Layout
st.title("🧠 Alzheimer's MRI Classification")
st.markdown("""
Welcome to the **Alzheimer Classification Tool**! 🏥

This app classifies MRI scans into stages of Alzheimer's disease using deep learning. Upload an MRI scan below, and the model will predict whether the scan indicates:
- **Mild Demented**: Noticeable cognitive impairment, impacting daily life and decision-making.
- **Moderate Demented**: Significant cognitive impairment, requiring assistance with daily tasks.
- **Non Demented**: Normal brain function without signs of dementia.
- **Very Mild Demented**: Early signs of cognitive decline, minimal impact on daily activities.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to trigger prediction
    if st.button("Classify MRI Scan"):
        with st.spinner("Classifying..."):
            # Make prediction
            prediction = predict(image)
            predicted_class = classes[np.argmax(prediction)]

            # Display prediction result
            st.subheader("Prediction Result")
            st.write(f"{predicted_class}")
            
            # Explanation of predicted class
            explanations = {
                "Mild Demented": "Noticeable cognitive impairment, impacting daily life and decision-making.",
                "Moderate Demented": "Significant cognitive impairment, requiring assistance with daily tasks.",
                "Non Demented": "Normal brain function without signs of dementia.",
                "Very Mild Demented": "Early signs of cognitive decline, minimal impact on daily activities."
            }
            st.write(f"*Explanation:* {explanations[predicted_class]}")
