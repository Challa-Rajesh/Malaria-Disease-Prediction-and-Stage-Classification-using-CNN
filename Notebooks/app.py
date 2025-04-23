import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from matplotlib.patches import Rectangle
import io
import base64
import os

# Set Streamlit config
st.set_page_config(page_title="Malaria Classifier", layout="wide")

# 🖼️ Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: black;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span, label {{
        color: black !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ✅ Set your background image path here
bg_image_path = os.path.expanduser("~/Downloads/national-institute-of-allergy-and-infectious-diseases-Ykv2gm43QFc-unsplash.jpg")  # Change if needed
set_background(bg_image_path)

# Load the trained model
model = tf.keras.models.load_model('./Models/malaria_cnn_checkpoint.h5')

# Class labels
class_mapping = {1: "Infected", 0: "Uninfected"}

# --- Common helper: Preprocess image ---
def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.expand_dims(image, axis=0), image

# --- For Single Cell: CAM Heatmap ---
def class_activation_map(img_array, model, layer_name='conv2d_7'):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img_array, axis=0))
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (64, 64))
    return heatmap

# --- For Multiple Image Classification ---
def predict_label(image):
    preprocessed, _ = preprocess_image(image)
    prediction = model.predict(preprocessed)
    class_idx = int(np.round(prediction[0][0]))
    return class_mapping[class_idx], class_idx

# --- Streamlit UI ---
st.title("🧬 Malaria Detection App")


# Custom CSS to increase radio button font size
st.markdown("""
    <style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Radio buttons
mode = st.radio("Choose mode:", ["🔍 Single Cell Classification", "📊 Stage Prediction (Multiple Images)"])





if mode == "🔍 Single Cell Classification":
    st.subheader("Upload a blood smear image to detect infection and view activation heatmap")
    uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        preprocessed, original = preprocess_image(image)
        prediction = model.predict(preprocessed)
        predicted_class_idx = int(np.round(prediction[0][0]))
        predicted_label = class_mapping[predicted_class_idx]

        st.markdown(f"### 🧪 Prediction: `{predicted_label}`")

        heatmap = class_activation_map(original, model)
        peak_coords = peak_local_max(heatmap, num_peaks=5, threshold_rel=0.5, min_distance=10)

        # Plot with heatmap overlay and rectangles
        fig, ax = plt.subplots()
        ax.imshow(original)
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        for y, x in peak_coords:
            rect = Rectangle((x - 5, y - 5), 10, 10, linewidth=1, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        st.image(buf, caption='🧠 Activation Heatmap', width=300)

elif mode == "📊 Stage Prediction (Multiple Images)":
    st.subheader("Upload individual cell images to classify and predict disease stage")
    uploaded_files = st.file_uploader("📤 Upload one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        infected_count = 0
        uninfected_count = 0
        total = len(uploaded_files)

        st.markdown("### 🔍 Classification Results:")
        cols = st.columns(3)

        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            label, class_idx = predict_label(image)

            with cols[i % 3]:
                st.image(image, caption=f"{uploaded_file.name}\nPredicted: `{label}`", width=300)

            if class_idx == 1:
                infected_count += 1
            else:
                uninfected_count += 1

        infection_ratio = infected_count / total if total else 0

        if infection_ratio <= 0.10:
            stage = "🟢 Initial Stage"
        elif infection_ratio <= 0.40:
            stage = "🟡 Moderate Stage"
        else:
            stage = "🔴 Advanced Stage"

        st.markdown("---")
        st.markdown("### 📊 Summary")
        st.markdown(f"- 🦠 Infected: `{infected_count}`")
        st.markdown(f"- ✅ Uninfected: `{uninfected_count}`")
        st.markdown(f"- 📌 Total Cells: `{total}`")
        st.markdown(f"### 🧬 Predicted Disease Stage: **{stage}**")

