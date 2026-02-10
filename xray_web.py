
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    # Load model without compilation to avoid compatibility issues with old Keras versions
    model = tf.keras.models.load_model("model/xray_model.hdf5", compile=False)
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

# Load class names
class_names = ['NORMAL', 'PNEUMONIA']

st.write("""
         # Pneumonia Identification System
         """
         )

file = st.file_uploader("Please upload a chest scan file", type=["jpg","jpeg", "png"])

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write("**Raw Predictions:**", predictions)
    st.write("**Confidence Scores:**", score.numpy())
    
    # Display the final prediction
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    st.markdown("---")
    st.subheader("Prediction Result:")
    if predicted_class == "PNEUMONIA":
        st.error(f"🔴 **{predicted_class}** detected with **{confidence:.2f}%** confidence")
    else:
        st.success(f"🟢 **{predicted_class}** - No pneumonia detected with **{confidence:.2f}%** confidence")
