import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import requests

# Download and load the TensorFlow Lite model
@st.cache_resource
def load_tflite_model():
    # URL to download the model
    model_url = "https://drive.google.com/uc?export=download&id=1iO013Rqp0dOlHHoFuQsNQEWsBS2f11mj"
    model_path = Path("alzheimer_densenet121.tflite")

    # Download the model if it doesn't exist
    if not model_path.exists():
        st.info("Downloading the TensorFlow Lite model...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully.")
        else:
            st.error("Failed to download the model. Check the URL.")
            return None

    # Load the TFLite model
    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        st.success("Model loaded successfully.")
        return interpreter
    except Exception as e:
        st.error(f"Failed to load the TensorFlow Lite model: {e}")
        return None


# Perform predictions using the TFLite interpreter
def predict_with_tflite(interpreter, image):
    # Preprocess image
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    image = tf.image.resize(image, (input_shape[1], input_shape[2]))  # Resize to input shape
    image = tf.cast(image, dtype=tf.float32) / 255.0  # Normalize
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], image.numpy())
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


# Load the model
interpreter = load_tflite_model()
if interpreter is None:
    st.stop()  # Stop if model loading fails


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
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to Tensor
    image = np.array(image)

    # Make prediction
    if st.button("Classify MRI"):
        predictions = predict_with_tflite(interpreter, image)
        class_names = ["Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
        predicted_class = class_names[np.argmax(predictions[0])]
        st.subheader("Prediction Result")
        st.write(f"**{predicted_class}**")

        # Explanation
        explanations = {
            "Non-Demented": "Normal brain function without signs of dementia.",
            "Very Mild Demented": "Early signs of cognitive decline, minimal impact on daily activities.",
            "Mild Demented": "Noticeable cognitive impairment, impacting daily life and decision-making.",
            "Moderate Demented": "Significant cognitive impairment, requiring assistance with daily tasks."
        }
        st.write(f"**Explanation:** {explanations[predicted_class]}")
