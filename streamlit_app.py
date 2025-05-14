import streamlit as st
from PIL import Image
from mymodel import predict

st.set_page_config(
    page_title="🍔 Food Vision",
    page_icon="🍽️",
    layout="centered",
)

st.title("🍽️ Food Vision Classifier")
st.write("Upload a photo of a dish, and I'll tell you what it is!")

uploaded_file = st.file_uploader(
    label="Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Analyzing..."):
            label = predict(image)               # ← make sure this line is active
        st.success(f"**Prediction:** {label}")
