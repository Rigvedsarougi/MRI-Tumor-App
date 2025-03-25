import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from utils.preprocessing import preprocess_image, explore_dataset
from utils.models import load_model, predict_tumor, train_models
from utils.visualization import plot_sample_images, plot_metrics

# Page configuration
st.set_page_config(
    page_title="Brain MRI Tumor Classification",
    page_icon=":brain:",
    layout="wide"
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["Dataset Exploration", "Image Preprocessing", "Model Training", "Prediction"])

# Main content
st.title("Brain MRI Tumor Classification")

if page == "Dataset Exploration":
    st.header("Dataset Exploration")
    
    if st.button("Load Sample Images"):
        with st.spinner("Loading images..."):
            sample_images = explore_dataset("data")
            fig = plot_sample_images(sample_images)
            st.pyplot(fig)
            
        st.success("Dataset exploration complete!")
        
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**'Yes' folder (Tumor):**")
            yes_count = len(os.listdir("data/yes"))
            st.write(f"Number of images: {yes_count}")
            
        with col2:
            st.info("**'No' folder (No Tumor):**")
            no_count = len(os.listdir("data/no"))
            st.write(f"Number of images: {no_count}")
            
        st.write("**Key Challenges:**")
        st.markdown("""
        - Variations in image resolution and dimensions
        - Differences in contrast and brightness
        - Potential artifacts in MRI scans
        - Class imbalance (if present)
        """)

elif page == "Image Preprocessing":
    st.header("Image Preprocessing")
    
    uploaded_file = st.file_uploader("Upload an MRI image to see preprocessing", 
                                   type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original MRI", use_column_width=True)
            
        with col2:
            st.subheader("Preprocessed Image")
            with st.spinner("Processing image..."):
                processed_img = preprocess_image(np.array(image))
                st.image(processed_img, caption="Processed MRI", use_column_width=True)
                
        st.subheader("Preprocessing Steps")
        st.markdown("""
        1. **Resizing**: All images standardized to 256x256 pixels
        2. **Normalization**: Pixel values scaled to [0,1] range
        3. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        4. **Denoising**: Median filtering to reduce noise
        5. **Edge Enhancement**: Sharpening important features
        """)

elif page == "Model Training":
    st.header("Model Training")
    
    st.write("""
    We'll train and compare three CNN architectures:
    1. **Basic CNN**: Simple 4-layer architecture
    2. **VGG-like**: Deeper network with more filters
    3. **ResNet**: Residual network with skip connections
    """)
    
    if st.button("Train Models"):
        with st.spinner("Training models (this may take several minutes)..."):
            history, metrics = train_models("data")
            
        st.success("Training complete!")
        
        st.subheader("Training Metrics")
        fig = plot_metrics(history)
        st.pyplot(fig)
        
        st.subheader("Model Performance Comparison")
        df = pd.DataFrame(metrics)
        st.table(df.style.highlight_max(axis=0))
        
        st.write("""
        **Key Observations:**
        - ResNet typically performs best but takes longer to train
        - Basic CNN is faster but may underfit on complex patterns
        - VGG-like offers a good balance between speed and accuracy
        """)

elif page == "Prediction":
    st.header("Tumor Prediction")
    
    model_option = st.selectbox(
        "Select model for prediction",
        ("Basic CNN", "VGG-like", "ResNet")
    )
    
    uploaded_file = st.file_uploader("Upload an MRI image for tumor detection", 
                                   type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and model_option:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="MRI Scan", use_column_width=True)
            
        with col2:
            st.subheader("Prediction")
            with st.spinner("Analyzing image..."):
                processed_img = preprocess_image(np.array(image))
                prediction, confidence = predict_tumor(model_option, processed_img)
                
                if prediction == "Tumor Detected":
                    st.error(f"**{prediction}** (Confidence: {confidence:.2%})")
                else:
                    st.success(f"**{prediction}** (Confidence: {confidence:.2%})")
                
                st.image(processed_img, caption="Processed Image", use_column_width=True)
                
        st.subheader("Model Information")
        if model_option == "Basic CNN":
            st.code("""
            Model architecture:
            - Conv2D (32 filters) -> ReLU -> MaxPooling
            - Conv2D (64 filters) -> ReLU -> MaxPooling
            - Conv2D (128 filters) -> ReLU -> MaxPooling
            - Flatten -> Dense (128) -> ReLU -> Dropout
            - Output (sigmoid)
            """)
        elif model_option == "VGG-like":
            st.code("""
            Model architecture:
            - 2x Conv2D (64 filters) -> ReLU -> MaxPooling
            - 2x Conv2D (128 filters) -> ReLU -> MaxPooling
            - 3x Conv2D (256 filters) -> ReLU -> MaxPooling
            - Flatten -> Dense (512) -> ReLU -> Dropout
            - Output (sigmoid)
            """)
        else:  # ResNet
            st.code("""
            Model architecture:
            - Initial Conv + MaxPool
            - 2x Residual blocks (64 filters)
            - 2x Residual blocks (128 filters)
            - 2x Residual blocks (256 filters)
            - Global Average Pooling
            - Output (sigmoid)
            """)

# Footer
st.markdown("---")
st.markdown("""
**Brain MRI Tumor Classification**  
*AIMIL Ltd. Data Scientist Assignment*  
[GitHub Repository](https://github.com/yourusername/brain-mri-tumor-classification)
""")
