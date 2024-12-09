import streamlit as st
import tensorflow as tf
import requests
from pathlib import Path

# Function to download the model from Google Drive
@st.cache_resource
def download_and_load_model():
    model_path = "alzheimer_densenet121.h5"
    
    # Check if the model file already exists locally
    if not Path(model_path).exists():
        st.write("Downloading model... Please wait.")
        url = "https://drive.google.com/uc?export=download&id=1zKTzDOI5xBVzU7NGvifgmtrDe-uxwBq5"
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # Ensure the chunk isn't empty
                    f.write(chunk)
        st.success("Model downloaded successfully!")
    
    # Load the model
    return tf.keras.models.load_model(model_path)

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
