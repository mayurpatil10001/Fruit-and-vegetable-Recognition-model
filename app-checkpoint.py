import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image  # Import the Image class from PIL

st.header('Image Classification Model')

# Load the pre-trained model
model = load_model('C:/Python/Image_Classification/Image_classify.keras')

# List of categories (fruits and vegetables)
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
            'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
            'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
            'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
            'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
            'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Define the input image dimensions
img_height = 180
img_width = 180

# Drag-and-drop functionality for image upload
uploaded_file = st.file_uploader("Drag and drop an image here...", type="jpg")

# Process the image and make predictions once the image is uploaded
if uploaded_file is not None:
    # Load the uploaded image using PIL
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Resize the image to match the input dimensions of the model
    image = image.resize((img_height, img_width))
    
    # Convert the image to a numpy array and add a batch dimension
    img_arr = np.array(image)
    img_bat = np.expand_dims(img_arr, axis=0)  # Convert to batch format
    
    # Normalize the image if necessary (depending on how the model was trained)
    img_bat = img_bat / 255.0  # Normalization
    
    # Make a prediction using the model
    predict = model.predict(img_bat)
    
    # Get the probability scores and label
    score = tf.nn.softmax(predict[0])
    
    # Display the prediction result
    st.write(f"The AI model predicts that the image is a **{data_cat[np.argmax(score)]}**.")
    st.write(f"Prediction confidence: **{np.max(score) * 100:.2f}%**.")
