import streamlit as st
import tensorflow as tf
from streamlit_lottie import st_lottie  # import the necessary packages for image recognition
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
import json
import cv2
from PIL import Image
from io import BytesIO
import pandas as pd
import requests 
import urllib
from streamlit_image_select import image_select
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# set page layout
st.set_page_config(
    page_title="Image Classification App",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    )

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
        
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.title("A CAD system using CNN")
first_anime = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_VfWq5Z.json")

st_lottie(
    first_anime,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    height=300,
    width=300,
    key=None,
)
st.subheader('Introduction.')
st.write('This project aims to help dermatologists to diagnose skin lesions images and determines wether it is cancerous or not. This is done using Convolutional Neural Network')
st.write('''
# Dataset Description.
''')
second_anime = load_lottiefile("data4.json")

st_lottie(
    second_anime,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    height=300,
    width=800,
    key=None,
)

data = pd.read_csv('train.csv')
all_columns=['NV','BCC','AK','BKL','DF','VASC','SCC','MEL']
st.write('The dataset used for training models consists of more than 88000 skin lesion images belonging to 8 classes. Each class contains number of images ranging from 4500 and 12800 images.')
st.image('data_full.png')
st.write('The dataset used')
st.image('data_after.png')           
st.sidebar.subheader("Options")
models_list = ["Xception", "EffecientNetB5", "VGG19"]
network = st.sidebar.selectbox("Select the Model", models_list)

@st.cache(allow_output_mutation=True)
def get_model(model_name):
    load = tf.keras.models.load_model(model_name)
    return (load)

model_1 = get_model('Xcep_multi290_new.hdf5')
model_2 = get_model('EFNETB5_model.hdf5')
model_3 = get_model('VGG_model.hdf5')

# define a dictionary that maps model names to their classes

MODELS = {
    "Xception": model_1,
    "EffecientNetB5": model_2,
    "VGG19": model_3
}



uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg"]
)

model = MODELS[network]
class_names = {0: 'Melanocytic Nevi',
               1: 'Basal Cell Carcinoma',
               2: 'Actinic Keratosis',
               3: 'Benign Keratosis',
               4: 'Dermatofibroma',
               5: 'Vascular Lesion',
               6: 'Squamous Cell Carcinoma ',
               7: 'Melanoma' }  

class_list = ['Melanocytic Nevi', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma', 'Vascular Lesion', 'Squamous Cell Carcinoma ', 'Melanoma']

if uploaded_file:
    bytes_data = uploaded_file.read()

    inputShape = (224, 224)
    preprocess = preprocess_input

    if network in ("Xception"):
        inputShape = (290, 290)
        preprocess = preprocess_input



    image = Image.open(BytesIO(bytes_data))
    image = image.convert("RGB")
    image = image.resize(inputShape)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.


    preds = model.predict(image)  
    index = np.argmax(preds) 
    st.image(bytes_data, width=600)
    s = pd.Series(preds[0],index=class_list)
    df = pd.DataFrame(s, columns=['confidence'])   
    st.title("Prediction - {}".format(class_names[index])) 

if st.sidebar.checkbox("try on image sample"):
    img = image_select(
    label="Select an image",
    images=[
        "ISIC_NV.jpg",
        "ISIC_MEL.jpg",
        'ISIC_BKL.jpg',
        'ISIC_AK.jpg',
        'ISIC_BCC.jpg',
        'ISIC_DF.jpg',
        'ISIC_VASC.jpg',
        'ISIC_SCC.jpg',
    ],
    captions=["NV", "MEL", "BKL", "AK", "BCC", "DF", "VASC", "SCC"],
     use_container_width=False)
    st.image(img)

    inputShape = (224, 224)
    preprocess = preprocess_input

    if network in ("Xception"):
        inputShape = (290, 290)
        preprocess = preprocess_input


    image = Image.open(img)
    image = image.convert("RGB")
    image = image.resize(inputShape)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.


    preds = model.predict(image)
    index = np.argmax(preds) 
    s = pd.Series(preds[0],index=class_list)
    df = pd.DataFrame(s, columns=['confidence'])   
    st.title("Prediction - {}".format(class_names[index]))

if st.sidebar.checkbox("show top 5 predictions"):
    st.subheader(f"Top Predictions from {network} model")
    st.dataframe(
        df.sort_values(by=['confidence'], ascending=False).head(5)
        )

if st.sidebar.checkbox("Show classification report"):

    if network in ("Xception"):
        st.image('Xcep_metrics.PNG', width=700, caption='Xception model classification report')

    elif network in ("EffecientNetB5"):
        st.image('efn_metrics.PNG', width=700, caption='EffecientNetB5 model classification report')

    else:
        st.image('VGG_metrics.PNG', width=700, caption='VGG19 model classification report')


if st.sidebar.checkbox("Show confusion matrix"):

    if network in ("Xception"):
        st.image('Xcep_cnf.png', width=700, caption='Xception model Confusion Matrix')

    elif network in ("EffecientNetB5"):
        st.image('efn_cnf.png', width=700, caption='EffecientNetB5 model Confusion Matrix')

    else:
        st.image('VGG_cnf.png', width=700, caption='VGG19 model Confusion Matrix')

