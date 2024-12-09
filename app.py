import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from pathlib import Path

# Function to download and load the TFLite model
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

# Define the correct class order as in training
classes = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Function to preprocess image and make a prediction
def predict(image):
    # Preprocess image
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension

    # Get input and output details from the model
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Set the input tensor
    model.set_tensor(input_details[0]['index'], image)

    # Run inference
    model.invoke()

    # Get prediction from the output tensor
    prediction = model.get_tensor(output_details[0]['index'])
    return prediction

# Streamlit app UI
st.title("Alzheimer Classification Using MRI Scans")
st.markdown("""
This application classifies MRI scans into four categories:
- **Mild Demented**: Noticeable cognitive impairment, impacting daily life and decision-making.
- **Moderate Demented**: Significant cognitive impairment, requiring assistance with daily tasks.
- **Non Demented**: Normal brain function without signs of dementia.
- **Very Mild Demented**: Early signs of cognitive decline, minimal impact on daily activities.

Upload an MRI scan image below to classify it.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction = predict(image)
    predicted_class = classes[np.argmax(prediction)]

    # Display prediction result
    st.subheader("Prediction Result")
    st.write(f"**{predicted_class}**")
    
    # Explanation of predicted class
    explanations = {
        "Mild Demented": "Noticeable cognitive impairment, impacting daily life and decision-making.",
        "Moderate Demented": "Significant cognitive impairment, requiring assistance with daily tasks.",
        "Non Demented": "Normal brain function without signs of dementia.",
        "Very Mild Demented": "Early signs of cognitive decline, minimal impact on daily activities."
    }
    st.write(f"**Explanation:** {explanations[predicted_class]}")
