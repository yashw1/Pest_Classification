import streamlit as st
import tensorflow as tf
import keras.utils as image
import numpy as np
from PIL import Image, ImageOps  # Streamlit works with PIL library very easily for Images
import cv2
#import utils.SQLiteDB as dbHandler
#from app import prediction
import os

path = 'C:\\Users\\kapse\PycharmProjects\\pythonProject\\upload'
model_path = 'C:\\Users\\kapse\PycharmProjects\\pythonProject\\PestImageClassificationInception.h5'
#model_path = 'C:\\Users\\asus\\PycharmProjects\\PestClassification\\PestImageClassificationCNN.h5'

def save_uploadedfile(uploadedfile, path):
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    print("Saved File:{} to upload".format(uploadedfile.name))


st.title("Pest Image Classification using Deep Learning")
upload = st.file_uploader('Upload a pest image')

def prediction(savedModel, inputImage):
    test_image = image.load_img( inputImage, target_size=(256, 256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = savedModel.predict(test_image)
    print("Predicted result", result)
    return result




if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)

    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # Color from BGR to RGB
    print("type of opencv", type(opencv_image))
    img = Image.open(upload)
    st.image(img, caption='Uploaded Image', width=300)
    if st.button('Predict'):
        # Load pretrained Model
        model = tf.keras.models.load_model(model_path)

        path_dir = os.path.join(os.getcwd(), 'upload')
        print("path_dir =", path_dir)
        upload_path = os.path.join(path_dir, upload.name)
        print("upload_path=", upload_path)

        # Save uploaded file
        save_uploadedfile(upload, upload_path)

        # Prediction on uploaded image
        result = prediction(model, upload_path)
        #map_result = {1: 'beetle', 3: 'Black hairy', 0: 'corn earworm', 4: 'Field Cricket', 2: 'Termite'}//CNN
        map_result = {0:'Black hairy',1:'Field Cricket',2:'Termite',3:'beetle',4:'corn earworm'} #INCEPTION
        print('np array',np.argmax(result))
        print("Predicted result", result)
        print("Output labels = ", map_result)
        print("output[np.argmax(result)] = ", map_result[np.argmax(result)])
        st.title( map_result[np.argmax(result)])
        detail_result = {0:'This is a Black hairy',1:'This is a Field Cricket',2:'This is a Termite',3:'This is a beetle',4:'This is a corn earworm',}
        st.text(detail_result[np.argmax(result)])