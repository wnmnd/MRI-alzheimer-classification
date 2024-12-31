# MRI Alzheimer Classification

This project utilizes the **DenseNet121** model to classify Alzheimer's disease from MRI scans with an impressive accuracy of **99.2%**. The model has been deployed as a **Streamlit app**, providing a user-friendly interface to detect and classify Alzheimer's disease, showcasing the transformative potential of AI in healthcare diagnostics.

## Live Demo
Access the deployed application here: [MRI Alzheimer Detection App](https://mri-alzheimer-detection.streamlit.app/)

## Features
- **High Accuracy:** Achieves 99.2% accuracy using DenseNet121, ensuring reliable predictions.
- **Interactive UI:** A simple and intuitive web app built with Streamlit for effortless usage.
- **Real-World Impact:** Demonstrates the potential of machine learning in healthcare by aiding in early detection of Alzheimer's disease.

## Classification Categories
The app classifies MRI scans into the following stages of Alzheimer's disease:
1. **Mild Demented:** Noticeable cognitive impairment, impacting daily life and decision-making.  
2. **Moderate Demented:** Significant cognitive impairment, requiring assistance with daily tasks.  
3. **Non Demented:** Normal brain function without signs of dementia.  
4. **Very Mild Demented:** Early signs of cognitive decline, minimal impact on daily activities.  

Simply upload an MRI scan, and the model will provide its prediction.

## Technology Stack
- **Frameworks:** TensorFlow, Streamlit
- **Model Architecture:** DenseNet121
- **Programming Language:** Python
- **Deployment Platform:** Streamlit Cloud
