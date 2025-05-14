from model import predict_image

# …

if st.button("Classify"):
    with st.spinner("Analyzing..."):
        label, confidence = predict_image(image)
    st.success(f"Prediction: **{label}** ({confidence*100:.1f}% confidence)")
