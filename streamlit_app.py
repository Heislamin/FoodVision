import streamlit as st
from PIL import Image
from model import predict_image

# ——— Page config ———
st.set_page_config(
    page_title="🍽️ Food Vision Classifier",
    page_icon="🍔",
    layout="centered",
)

# ——— Title & instructions ———
st.title("🍽️ Food Vision Classifier")
st.write("Upload an image of pizza or steak, and I'll tell you what it is!")

# ——— File uploader ———
uploaded_file = st.file_uploader(
    label="Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Load and down‑scale the image
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((300, 300))  # max width/height = 300px

    # Display the thumbnail
    st.image(image, caption="Uploaded image", use_column_width=False, width=300)

    # Classification button
    if st.button("Classify"):
        with st.spinner("Analyzing..."):
            label, confidence = predict_image(image)
        st.success(f"Prediction: **{label}** ({confidence * 100:.1f}% confidence)")
