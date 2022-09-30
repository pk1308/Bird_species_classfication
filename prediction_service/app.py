import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from s3 import get_best_model_s3

"""
# deep Classifier project

"""
class_name_path = "classes.npz"
with open(class_name_path, "rb") as file_obj:
    classes_name = np.load(file_obj, allow_pickle=True)


os.makedirs("models", exist_ok=True)
model_path = get_best_model_s3(
    best_model_path="models/best_model.h5",
    key="Best_model",
    bucket_name="projectbirdclassifier",
)
model = tf.keras.models.load_model(model_path)


def pred_classes_(model, img, class_names):
    """
    Imports an image located at IMG, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it

    # Make a prediction
    pred = model.predict(img)

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = class_names[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # if only one output, round

    # Plot the image and predicted class
    return pred_class


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    image = Image.open(uploaded_file)
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predicted = pred_classes_(model, img_array, classes_name)
    st.image(image, caption=f"predicted: {predicted}")
