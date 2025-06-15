import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model with error handling
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_trained_model()

st.title("Skin Cancer Prediction System")

if model is None:
    st.error("Model could not be loaded. Please check if 'model.keras' file exists.")
    st.stop()

uploaded = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    try:
        image = Image.open(uploaded)
        st.image(image, caption="Input Cell Image", width=200)

        def preprocess_image(uploaded_image):
            # Convert to RGB if needed
            if uploaded_image.mode != 'RGB':
                uploaded_image = uploaded_image.convert('RGB')
            resized_image = uploaded_image.resize((256, 256))
            image_array = img_to_array(resized_image)
            image_array /= 255.0
            return image_array

        def prediction(image_array):
            pred = model.predict(np.expand_dims(image_array, axis=0))
            return pred

        with st.spinner('Processing image...'):
            inp = preprocess_image(image)
            ans = prediction(inp)
            classes = ['Benign', 'Malignant']

            pred_class = np.argmax(ans)
            confidence = ans[0][pred_class]
            pred_class_name = classes[pred_class]

            confidence_percentage = round(confidence * 100, 2)
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Class", pred_class_name)
            with col2:
                st.metric("Confidence", f"{confidence_percentage}%")
            
            # Show detailed results
            data = {
                'Class': classes,
                'Probability (%)': [round(ans[0][0] * 100, 2), round(ans[0][1] * 100, 2)]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        
st.markdown("---")
st.markdown("⚠️ **Disclaimer**: This is for educational purposes only. Always consult healthcare professionals for medical advice.")
