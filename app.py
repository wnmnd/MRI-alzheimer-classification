import streamlit as st
import tensorflow as tf
import requests
from pathlib import Path

# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="alzheimer_densenet121.tflite")
    interpreter.allocate_tensors()
    return interpreter

model = load_model()

# Preprocess and make prediction
def predict(image):
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Preprocess image
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # Run inference
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])
    return prediction


# Load the model
model = download_and_load_model()

# App UI
st.title("Alzheimer Classification Using MRI Scans")
st.markdown("""
This application classifies MRI scans into four categories:
- **Non-Demented**: Normal brain function without signs of dementia.
- **Very Mild Demented**: Early signs of cognitive decline, minimal impact on daily activities.
- **Mild Demented**: Noticeable cognitive impairment, impacting daily life and decision-making.
- **Moderate Demented**: Significant cognitive impairment, requiring assistance with daily tasks.

Upload an MRI scan image below to classify it.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    from PIL import Image
    import numpy as np

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.resize((224, 224))  # Resize to the model's input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(image)
    classes = ["Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
    result = classes[np.argmax(prediction)]
    
    # Display the prediction result
    st.subheader("Prediction Result")
    st.write(f"**{result}**")
    
    # Explanation of predicted class
    explanations = {
        "Non-Demented": "Normal brain function without signs of dementia.",
        "Very Mild Demented": "Early signs of cognitive decline, minimal impact on daily activities.",
        "Mild Demented": "Noticeable cognitive impairment, impacting daily life and decision-making.",
        "Moderate Demented": "Significant cognitive impairment, requiring assistance with daily tasks."
    }
    st.write(f"**Explanation:** {explanations[result]}")
