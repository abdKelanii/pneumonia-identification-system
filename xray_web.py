
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import os
import glob

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add sidebar with information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/lungs.png", width=100)
    st.title("About")
    st.markdown("""
    This AI-powered system uses deep learning to detect pneumonia from chest X-ray images.
    
    **Features:**
    - 🧠 ResNet-based model
    - 🎯 High accuracy detection
    - ⚡ Real-time predictions
    - 📊 Confidence scores
    
    **How to use:**
    1. Upload your X-ray or choose a sample
    2. Click 'Analyze X-Ray'
    3. View the results
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning("This is a research prototype. NOT for medical diagnosis. Always consult healthcare professionals.")

@st.cache_resource
def load_model():
    # Load model without compilation to avoid compatibility issues with old Keras versions
    model = tf.keras.models.load_model("model/xray_model.hdf5", compile=False)
    return model

with st.spinner('🔄 Loading AI model...'):
    model = load_model()

# Load class names
class_names = ['NORMAL', 'PNEUMONIA']

# Main title
st.title("🫁 Chest X-Ray Pneumonia Detection System")
st.markdown("### AI-Powered Medical Image Analysis")
st.markdown("---")

# Create tabs for upload and sample selection
tab1, tab2 = st.tabs(["📤 Upload Your Image", "🖼️ Choose Sample Image"])

selected_image = None
image_source = None

with tab1:
    st.markdown("### Upload a Chest X-Ray Image")
    file = st.file_uploader("Please upload a chest scan file", type=["jpg","jpeg", "png"], key="file_uploader")
    if file is not None:
        selected_image = Image.open(file)
        image_source = "uploaded"

with tab2:
    st.markdown("### Select from Sample Images")
    st.info("💡 These are example X-rays you can use to test the system")
    
    # Get sample images
    normal_images = sorted(glob.glob("sample images/NORMAL/*.jpeg"))
    pneumonia_images = sorted(glob.glob("sample images/NEUMONIA/*.jpeg"))
    
    # Create two columns for Normal and Pneumonia
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Normal Chest X-Rays")
        st.caption(f"{len(normal_images)} samples available")
        normal_choice = st.radio(
            "Select a normal X-ray:",
            options=["None"] + [os.path.basename(img) for img in normal_images],
            key="normal_radio",
            label_visibility="collapsed"
        )
        
        if normal_choice != "None":
            selected_path = f"sample images/NORMAL/{normal_choice}"
            if os.path.exists(selected_path):
                selected_image = Image.open(selected_path)
                image_source = "sample"
                st.image(selected_image, caption=f"Preview: {normal_choice}", use_column_width=True)
    
    with col2:
        st.markdown("#### 🔴 Pneumonia Chest X-Rays")
        st.caption(f"{len(pneumonia_images)} samples available")
        pneumonia_choice = st.radio(
            "Select a pneumonia X-ray:",
            options=["None"] + [os.path.basename(img) for img in pneumonia_images],
            key="pneumonia_radio",
            label_visibility="collapsed"
        )
        
        if pneumonia_choice != "None":
            selected_path = f"sample images/NEUMONIA/{pneumonia_choice}"
            if os.path.exists(selected_path):
                selected_image = Image.open(selected_path)
                image_source = "sample"
                # Detect if it's bacterial or viral from filename
                img_type = "Bacterial" if "bacteria" in pneumonia_choice.lower() else "Viral"
                st.image(selected_image, caption=f"Preview: {pneumonia_choice} ({img_type})", use_column_width=True)

st.markdown("---")

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


# Process and display prediction
if selected_image is None:
    st.info("👆 Please upload an image or select a sample image from the tabs above to get started!")
else:
    # Display the selected image
    st.markdown("### 📸 Selected Image")
    st.image(selected_image, use_column_width=True)
    
    # Add a predict button
    if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
        with st.spinner("Analyzing the chest X-ray..."):
            predictions = import_and_predict(selected_image, model)
            score = tf.nn.softmax(predictions[0])
            
            # Display the final prediction
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)
            
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Create columns for better layout
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                if predicted_class == "PNEUMONIA":
                    st.error(f"### 🔴 **{predicted_class}** Detected")
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    st.warning("⚠️ This X-ray shows signs of pneumonia. Please consult a healthcare professional for proper diagnosis and treatment.")
                else:
                    st.success(f"### 🟢 **{predicted_class}**")
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    st.info("✅ No signs of pneumonia detected in this X-ray.")
            
            with result_col2:
                # Display confidence meter
                st.metric(label="Confidence Level", value=f"{confidence:.1f}%")
            
            # Expandable section for technical details
            with st.expander("🔬 View Technical Details"):
                st.write("**Raw Predictions:**", predictions)
                st.write("**Confidence Scores:**", score.numpy())
                st.write("**Predicted Class Index:**", np.argmax(score))
                
                # Show prediction breakdown
                st.markdown("**Class Probabilities:**")
                for i, class_name in enumerate(class_names):
                    prob = score.numpy()[i] * 100
                    st.write(f"- {class_name}: {prob:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🏥 <strong>Pneumonia Detection System</strong> | Powered by Deep Learning & TensorFlow</p>
        <p style='font-size: 0.9em;'>Dataset: Guangzhou Women and Children's Medical Center</p>
        <p style='font-size: 0.8em; color: #999;'>⚠️ For research and educational purposes only - Not for clinical diagnosis</p>
    </div>
    """, unsafe_allow_html=True)
