import streamlit as st
from PIL import Image
from model.py import predict_image

st.set_page_config(
    page_title="🍽️ Food Vision Classifier",
    page_icon="🍔",
    layout="centered",
)

st.title("🍽️ Food Vision Classifier")
st.write("Upload an image of pizza or steak, and I'll tell you what it is!")

uploaded_file = st.file_uploader(
    label="Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Analyzing..."):
            label, confidence = predict_image(image)
        st.success(f"Prediction: **{label}** ({confidence * 100:.1f}% confidence)")
