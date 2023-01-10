import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks

@st.cache
def apply_um(image):
  gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
  enhanced = cv2.addWeighted(image, 2.0, gaussian, -1.0, 0)  
  return enhanced

@st.cache
def preprocess_input(image):
  open_cv_image = np.array(image)
  open_cv_image = open_cv_image[:, :, ::-1].copy()

  preprocessed = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
  preprocessed = cv2.resize(preprocessed, (256, 256))
  preprocessed = apply_um(preprocessed)
  return preprocessed

@st.cache
def classify(model, image):
  image = tf.convert_to_tensor(image)
  result = model.predict(image)
  return "Pneumonia" if result > 0.5 else "Normal"


def main():
  model = models.load_model('model_um', compile=False)

  st.title('Chest X-ray Pneumonia Classification')

  upload = st.file_uploader('Insert image for classification', type=['png', 'jpeg', 'jpg'])
  c1, c2= st.columns(2)
  if upload is not None:
    im = Image.open(upload).convert('RGB')
    img = np.asarray(im)
    img = preprocess_input(img)
    img = np.expand_dims(img, 0)
    c1.header('Input Image')
    c1.image(im)

    st.header("Classification Results")

    result = classify(model, img)
    st.write(result)


if __name__ == '__main__':
  main()