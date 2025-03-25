import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, 
                                   Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score)

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

# Utility Functions
def explore_dataset(data_path):
    """Load and return sample images from the dataset."""
    samples = []
    for class_name in ["yes", "no"]:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            images = os.listdir(class_path)
            for _ in range(min(4, len(images))):  # Max 4 per class
                img_name = random.choice(images)
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path)
                samples.append((img, class_name))
    return samples

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess a single MRI image using PIL only."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize
    image = image.resize(target_size)
    
    # Contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Denoising
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    # Edge enhancement
    image = image.filter(ImageFilter.SHARPEN)
    
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    image = np.stack((image,)*3, axis=-1)  # Convert to 3 channels
    
    return image

def plot_sample_images(samples, figsize=(12, 8)):
    """Plot sample images from the dataset."""
    plt.figure(figsize=figsize)
    for i, (image, label) in enumerate(samples):
        plt.subplot(2, 4, i+1)
        plt.imshow(image, cmap='gray' if image.mode == 'L' else None)
        plt.title(f"Class: {label}")
        plt.axis('off')
    plt.tight_layout()
    return plt.gcf()

def create_basic_cnn(input_shape=(256, 256, 3)):
    """Create a simple CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def train_models(data_dir, epochs=10, batch_size=32):
    """Train models and return their histories and metrics."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    model = create_basic_cnn()
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        verbose=1
    )
    
    # Evaluate
    val_pred = (model.predict(val_gen) > 0.5).astype(int)
    val_true = val_gen.classes
    
    metrics = {
        'Accuracy': accuracy_score(val_true, val_pred),
        'Precision': precision_score(val_true, val_pred),
        'Recall': recall_score(val_true, val_pred),
        'F1 Score': f1_score(val_true, val_pred)
    }
    
    return model, history, metrics

def predict_tumor(model, image):
    """Predict tumor presence using the model."""
    img_array = np.expand_dims(image, axis=0)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        return "Tumor Detected", prediction
    else:
        return "No Tumor Detected", 1 - prediction

# Main App
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
            yes_count = len(os.listdir("data/yes")) if os.path.exists("data/yes") else 0
            st.write(f"Number of images: {yes_count}")
            
        with col2:
            st.info("**'No' folder (No Tumor):**")
            no_count = len(os.listdir("data/no")) if os.path.exists("data/no") else 0
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
                st.image(processed_img, caption="Processed MRI", use_column_width=True, 
                         channels="RGB")
                
        st.subheader("Preprocessing Steps")
        st.markdown("""
        1. **Resizing**: All images standardized to 256x256 pixels
        2. **Grayscale Conversion**: Convert to single channel
        3. **Contrast Enhancement**: PIL's contrast enhancement
        4. **Denoising**: Median filtering
        5. **Edge Enhancement**: Sharpening filter
        6. **Normalization**: Pixel values scaled to [0,1] range
        """)

elif page == "Model Training":
    st.header("Model Training")
    
    st.write("""
    We'll train a basic CNN architecture:
    - Conv2D (32 filters) -> ReLU -> MaxPooling
    - Conv2D (64 filters) -> ReLU -> MaxPooling
    - Conv2D (128 filters) -> ReLU -> MaxPooling
    - Flatten -> Dense (128) -> ReLU -> Dropout
    - Output (sigmoid)
    """)
    
    if st.button("Train Model"):
        if not os.path.exists("data/yes") or not os.path.exists("data/no"):
            st.error("Please ensure you have both 'yes' and 'no' folders in the data directory")
        else:
            with st.spinner("Training model (this may take several minutes)..."):
                model, history, metrics = train_models("data", epochs=5)
                
            st.success("Training complete!")
            
            st.subheader("Training Metrics")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['accuracy'], label='Train Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend()
            
            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            
            st.pyplot(fig)
            
            st.subheader("Model Performance")
            df = pd.DataFrame([metrics])
            st.table(df.style.highlight_max(axis=1))
            
            # Save the model for prediction page
            model.save("models/basic_cnn.h5")
            st.session_state.model = model

elif page == "Prediction":
    st.header("Tumor Prediction")
    
    uploaded_file = st.file_uploader("Upload an MRI image for tumor detection", 
                                   type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load model if not already loaded
        if 'model' not in st.session_state:
            if os.path.exists("models/basic_cnn.h5"):
                st.session_state.model = tf.keras.models.load_model("models/basic_cnn.h5")
            else:
                st.error("Please train the model first from the Model Training page")
                st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="MRI Scan", use_column_width=True)
            
        with col2:
            st.subheader("Prediction")
            with st.spinner("Analyzing image..."):
                processed_img = preprocess_image(np.array(image))
                prediction, confidence = predict_tumor(st.session_state.model, processed_img)
                
                if prediction == "Tumor Detected":
                    st.error(f"**{prediction}** (Confidence: {confidence:.2%})")
                else:
                    st.success(f"**{prediction}** (Confidence: {confidence:.2%})")
                
                st.image(processed_img, caption="Processed Image", use_column_width=True, 
                        channels="RGB")
                
        st.subheader("Model Information")
        st.code("""
        Model architecture:
        - Conv2D (32 filters) -> ReLU -> MaxPooling
        - Conv2D (64 filters) -> ReLU -> MaxPooling
        - Conv2D (128 filters) -> ReLU -> MaxPooling
        - Flatten -> Dense (128) -> ReLU -> Dropout
        - Output (sigmoid)
        """)

# Footer
st.markdown("---")
st.markdown("""
**Brain MRI Tumor Classification**  
*AIMIL Ltd. Data Scientist Assignment*  
[GitHub Repository](https://github.com/yourusername/brain-mri-tumor-classification)
""")
