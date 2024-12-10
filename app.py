# prompt: i have a cnn model i need a streamlit app for it

#!pip install streamlit
#!pip install pyngrok

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('CNN_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (255, 255))
    img = img / 255
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app
st.title("Pneumonia Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Preprocess the image
    preprocessed_image = preprocess_image(opencv_image)

    # Make a prediction
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)

    # Display the results
    st.image(opencv_image, caption="Uploaded Image", use_column_width=True)
    st.write("Prediction:")
    if predicted_class == 0:
        st.write("Normal")
    else:
        st.write("Pneumonia")

    st.write(f"Confidence: {prediction[0][predicted_class]:.2f}")