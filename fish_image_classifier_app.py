
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os

# Streamlit app setup
st.title("AIML - Multiclass Fish Image Classification")
st.write("Upload an image of a fish to classify it among multiple classes using a deep learning model.")

# Load model if exists, otherwise train a new one
MODEL_PATH = "fish_model.h5"
CLASS_NAMES = ['Betta', 'Guppy', 'Goldfish', 'Molly', 'Tetra']  # Example classes

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = train_model()
    return model

# Function to train a model (can be replaced with your own trained model)
def train_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(CLASS_NAMES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Data preparation (replace with your dataset path)
    train_dir = 'fish_dataset/train'
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    model.fit(train_generator, epochs=5)  # Reduce epochs for faster training
    model.save(MODEL_PATH)
    return model

# Load the model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 150, 150, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class} ({confidence * 100:.2f}% confidence)")
