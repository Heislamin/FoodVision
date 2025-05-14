# app.py
import streamlit as st
from PIL import Image
#from model import predict

# ——— Page config ———
st.set_page_config(
    page_title="🍔 Food Vision",
    page_icon="🍽️",
    layout="centered",
)

# ——— Title & instructions ———
st.title("🍽️ Food Vision Classifier")
st.write("Upload a photo of a dish, and I'll tell you what it is!")

# ——— File uploader & display ———
uploaded_file = st.file_uploader(
    label="Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Classification button
    if st.button("Classify"):
        with st.spinner("Analyzing..."):
            #label = predict(image)
        st.success(f"**Prediction:** {label}")
