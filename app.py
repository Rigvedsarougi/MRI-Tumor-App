import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, 
                                   Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score)

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

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
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize
    image = image.resize(target_size)
    
    # Contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Increase contrast by 2x
    
    # Denoising
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    # Edge enhancement
    image = image.filter(ImageFilter.SHARPEN)
    
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.repeat(image, 3, axis=-1)   # Repeat to make 3 channels
    
    return image

def plot_sample_images(samples, figsize=(12, 8)):
    """Plot sample images from the dataset."""
    plt.figure(figsize=figsize)
    for i, (image, label) in enumerate(samples):
        plt.subplot(2, 4, i+1)
        if isinstance(image, Image.Image):
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image, cmap='gray' if image.shape[-1] == 1 else None)
        plt.title(f"Class: {label}")
        plt.axis('off')
    plt.tight_layout()
    return plt.gcf()

def create_cnn_model(input_shape=(256, 256, 3)):
    """Create a CNN model with better architecture."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy', 
                         tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall()])
    return model

def train_model(data_dir, epochs=10, batch_size=32):
    """Train model and return history and metrics."""
    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        color_mode='rgb'
    )
    
    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        color_mode='rgb'
    )
    
    model = create_cnn_model()
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on validation set
    val_pred = (model.predict(val_gen) > 0.5).astype(int)
    val_true = val_gen.classes
    
    metrics = {
        'Accuracy': accuracy_score(val_true, val_pred),
        'Precision': precision_score(val_true, val_pred),
        'Recall': recall_score(val_true, val_pred),
        'F1 Score': f1_score(val_true, val_pred)
    }
    
    # Save the model
    model.save("models/brain_mri_cnn.h5")
    
    return model, history, metrics

def predict_with_model(model, image):
    """Make prediction with the trained model."""
    # Ensure image is in correct format
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Preprocess the image
    processed_img = preprocess_image(image)
    
    # Add batch dimension
    img_array = np.expand_dims(processed_img, axis=0)
    
    # Make prediction
    prediction_prob = model.predict(img_array)[0][0]
    prediction = "Tumor Detected" if prediction_prob > 0.5 else "No Tumor Detected"
    confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
    
    return prediction, confidence, processed_img

# Main App
st.title("Brain MRI Tumor Classification")

if page == "Dataset Exploration":
    st.header("Dataset Exploration")
    
    if st.button("Load Sample Images"):
        if not os.path.exists("data/yes") or not os.path.exists("data/no"):
            st.error("Please ensure you have both 'yes' and 'no' folders in the data directory")
        else:
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
                # Convert back to PIL for display
                display_img = Image.fromarray((processed_img[:, :, 0] * 255).astype(np.uint8))
                st.image(display_img, caption="Processed MRI", use_column_width=True)
                
        st.subheader("Preprocessing Steps")
        st.markdown("""
        1. **Resizing**: Standardized to 256×256 pixels
        2. **Grayscale Conversion**: Convert to single channel
        3. **Contrast Enhancement**: 2× contrast boost
        4. **Denoising**: Median filtering (3×3 kernel)
        5. **Edge Enhancement**: Sharpening filter
        6. **Normalization**: Pixel values scaled to [0,1] range
        7. **Channel Expansion**: Convert to 3-channel for model compatibility
        """)

elif page == "Model Training":
    st.header("Model Training")
    
    st.write("""
    **CNN Architecture:**
    - 4 Convolutional Blocks (32 → 64 → 128 → 256 filters)
    - MaxPooling after each block
    - 512-unit Dense layer with Dropout
    - Binary output (sigmoid activation)
    """)
    
    epochs = st.slider("Number of epochs", 5, 30, 15)
    batch_size = st.slider("Batch size", 16, 64, 32)
    
    if st.button("Train Model"):
        if not os.path.exists("data/yes") or not os.path.exists("data/no"):
            st.error("Please ensure you have both 'yes' and 'no' folders in the data directory")
        else:
            with st.spinner("Training model (this may take several minutes)..."):
                model, history, metrics = train_model("data", epochs=epochs, batch_size=batch_size)
                
            st.success("Training complete!")
            
            st.subheader("Training Metrics")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Accuracy plot
            ax1.plot(history.history['accuracy'], label='Train Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend()
            
            # Loss plot
            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            
            st.pyplot(fig)
            
            st.subheader("Model Performance Metrics")
            df = pd.DataFrame([metrics])
            st.dataframe(df.style.format("{:.2%}").highlight_max(axis=0))
            
            # Store model in session state
            st.session_state.model = model

elif page == "Prediction":
    st.header("Tumor Prediction")
    
    # Load model if not already loaded
    if 'model' not in st.session_state:
        if os.path.exists("models/brain_mri_cnn.h5"):
            st.session_state.model = load_model("models/brain_mri_cnn.h5")
        else:
            st.error("No trained model found. Please train the model first.")
            st.stop()
    
    uploaded_file = st.file_uploader("Upload an MRI image for tumor detection", 
                                   type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            st.image(image, caption="MRI Scan", use_column_width=True)
            
        with col2:
            st.subheader("Prediction")
            with st.spinner("Analyzing image..."):
                prediction, confidence, processed_img = predict_with_model(st.session_state.model, image)
                
                if prediction == "Tumor Detected":
                    st.error(f"**{prediction}** (Confidence: {confidence:.2%})")
                else:
                    st.success(f"**{prediction}** (Confidence: {confidence:.2%})")
                
                # Display processed image
                display_img = Image.fromarray((processed_img[:, :, 0] * 255).astype(np.uint8))
                st.image(display_img, caption="Processed Image", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Brain MRI Tumor Classification**  
*Debugged Implementation*  
[GitHub Repository](https://github.com/yourusername/brain-mri-tumor-classification)
""")
