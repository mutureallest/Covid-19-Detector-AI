# StreamLit stuff
import streamlit as st
from PIL import Image
from tensorflow.keras.models import model_from_json
import keras.backend.tensorflow_backend as K
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

import os

html_temp = """
    <div style="background-color:black;"><p style="color:white;font-size:70px">Covid-19 Detector AI</p></div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image...", type=("jpg", "png", "jpeg"))
st.text("Please upload chest radiograph scans | only jpg, jpeg or png files")


# showing the uploaded image
a = st.checkbox("Show image")
if uploaded_file is not None:
    if a:
        Image = Image.open(uploaded_file)
        st.image(Image, width=500, caption='Image uploaded by you.')


@st.cache(allow_output_mutation=True)
def load_model():
    model_weights = 'covid.h5'
    model_json = 'model.json'
    with open(model_json) as json_file:
        loaded_model = model_from_json(json_file.read())
    loaded_model.load_weights(model_weights)
    loaded_model.summary()  # included to make it visible when model is reloaded
    # session = K.get_session()
    return loaded_model


def preprocess_image(Image, target_size):
    if Image.mode != "RGB":
        Image = Image.convert("RGB")
    Image = Image.resize(target_size)
    Image = img_to_array(Image)
    Image = np.expand_dims(Image, axis=0)
    return Image


if __name__ == "__main__":
    # load the saved model
    if st.button('Check for Covid'):
        if uploaded_file is not None:
            # prepare the input data for prediction
            processed_image = preprocess_image(Image, target_size=(224, 224))
            model = load_model()
            # K.set_session(session)
            rslt = model.predict(processed_image)

            label = "I detect traces of Covid-19.." if rslt == [[0]] else "This seems Normal to me.."
            # display input and results
            st.success(label)
            st.write(rslt)

    if st.button("About AI"):
        st.info("Built by George Muturi")
        st.info("..of   Hue_man AI")


























