import streamlit as st
from PIL import Image
from imageCaptioning import generate_caption


st.title("AI Image Caption Generator")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
    # st.write("")

    with st.spinner("Generating caption..."):
        
        image = Image.open(uploaded_image)
        
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        caption = generate_caption(image)
        caption = caption.capitalize()
        st.subheader("Generated Caption:")
        st.write(caption)


