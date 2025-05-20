# app.py

import streamlit as st
from PIL import Image
from model import predict_image

# ——— Page configuration ———
st.set_page_config(
    page_title="🍽️ Food Vision Classifier",
    page_icon="🍔",
    layout="centered",
)

# ——— UI ———
st.title("🍽️ Food Vision Classifier")
st.write("Upload an image of pizza or steak, and I'll tell you what it is!")

uploaded_file = st.file_uploader(
    label="Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((300, 300))  # Resize for display

    st.image(image, caption="Uploaded image", use_container_width=False, width=300)

    if st.button("Classify"):
        with st.spinner("Analyzing..."):
            result = predict_image(image)
        st.success(result)
