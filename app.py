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
                                   Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                      ReduceLROnPlateau)
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split

# Page configuration
st.set_page_config(
    page_title="Brain MRI Tumor Classification",
    page_icon=":brain:",
    layout="wide"
)

# Constants
IMG_SIZE = (224, 224)  # Standard size for many CNN models
BATCH_SIZE = 32
EPOCHS = 30  # Increased from original
PATIENCE = 5  # For early stopping

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

def preprocess_image(image, target_size=IMG_SIZE):
    """Enhanced preprocessing with PIL only."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize with better interpolation
    image = image.resize(target_size, Image.LANCZOS)
    
    # Contrast enhancement with more control
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # More subtle enhancement
    
    # Denoising with different filter
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    # Edge enhancement with custom kernel
    kernel = ImageFilter.Kernel((3, 3), 
                              (-1, -1, -1, 
                               -1, 9, -1, 
                               -1, -1, -1), 
                              scale=1)
    image = image.filter(kernel)
    
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    image = np.stack((image,)*3, axis=-1)  # Convert to 3 channels
    
    return image

def plot_sample_images(samples, figsize=(12, 8)):
    """Improved visualization of samples."""
    plt.figure(figsize=figsize)
    for i, (image, label) in enumerate(samples):
        plt.subplot(2, 4, i+1)
        plt.imshow(image, cmap='gray' if image.mode == 'L' else None)
        plt.title(f"Class: {label}\nSize: {image.size}")
        plt.axis('off')
    plt.tight_layout()
    return plt.gcf()

def create_enhanced_model(input_shape=(224, 224, 3)):
    """More powerful CNN architecture."""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Classification block
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy', 
                         tf.keras.metrics.Precision(),
                         tf.keras.metrics.Recall()])
    return model

def train_enhanced_model(data_dir):
    """Enhanced training process with callbacks."""
    # More aggressive augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint('models/best_model.h5', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    
    model = create_enhanced_model()
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    model = load_model('models/best_model.h5')
    
    # Evaluate
    val_pred = (model.predict(val_gen) > 0.5).astype(int)
    val_true = val_gen.classes
    
    metrics = {
        'Accuracy': accuracy_score(val_true, val_pred),
        'Precision': precision_score(val_true, val_pred),
        'Recall': recall_score(val_true, val_pred),
        'F1 Score': f1_score(val_true, val_pred),
        'Confusion Matrix': confusion_matrix(val_true, val_pred)
    }
    
    return model, history, metrics

def plot_training_history(history):
    """Enhanced training history visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
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
    
    plt.tight_layout()
    return fig

def predict_tumor(model, image):
    """Enhanced prediction with confidence thresholds."""
    img_array = np.expand_dims(image, axis=0)
    prediction_prob = model.predict(img_array)[0][0]
    
    # More nuanced confidence thresholds
    if prediction_prob > 0.7:
        return "Tumor Detected (High Confidence)", prediction_prob
    elif prediction_prob > 0.55:
        return "Tumor Likely Present", prediction_prob
    elif prediction_prob > 0.45:
        return "Uncertain - Possibly Normal", prediction_prob
    elif prediction_prob > 0.3:
        return "Likely Normal", 1 - prediction_prob
    else:
        return "Normal (High Confidence)", 1 - prediction_prob

# Main App
st.title("Enhanced Brain MRI Tumor Classification")

if page == "Dataset Exploration":
    st.header("Enhanced Dataset Exploration")
    
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
        - Need for precise tumor boundary detection
        """)

elif page == "Image Preprocessing":
    st.header("Enhanced Image Preprocessing")
    
    uploaded_file = st.file_uploader("Upload an MRI image to see preprocessing", 
                                   type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Original MRI\nSize: {image.size}", 
                    use_column_width=True)
            
        with col2:
            st.subheader("Preprocessed Image")
            with st.spinner("Processing image..."):
                processed_img = preprocess_image(np.array(image))
                st.image(processed_img, 
                        caption=f"Processed MRI\nSize: {processed_img.shape[:2]}", 
                        use_column_width=True, 
                        channels="RGB")
                
        st.subheader("Enhanced Preprocessing Steps")
        st.markdown("""
        1. **Resizing**: Standardized to 224x224 pixels (LANCZOS interpolation)
        2. **Grayscale Conversion**: Convert to single channel
        3. **Contrast Enhancement**: Controlled enhancement (1.5x)
        4. **Denoising**: Median filtering (3x3)
        5. **Edge Enhancement**: Custom sharpening kernel
        6. **Normalization**: Pixel values scaled to [0,1] range
        7. **Channel Expansion**: Convert to 3 channels for model compatibility
        """)

elif page == "Model Training":
    st.header("Enhanced Model Training")
    
    st.write("""
    **Enhanced CNN Architecture:**
    - 3 Convolutional Blocks with Batch Normalization
    - Increased dropout rates (0.2-0.5) for better regularization
    - Larger dense layer (256 units)
    - Adam optimizer with learning rate 0.0001
    - Early stopping, model checkpointing, and LR reduction
    """)
    
    if st.button("Train Enhanced Model"):
        if not os.path.exists("data/yes") or not os.path.exists("data/no"):
            st.error("Please ensure you have both 'yes' and 'no' folders in the data directory")
        else:
            # Create models directory if not exists
            os.makedirs("models", exist_ok=True)
            
            with st.spinner("Training enhanced model (this may take 10-20 minutes)..."):
                model, history, metrics = train_enhanced_model("data")
                
            st.success("Training complete!")
            
            st.subheader("Training Metrics")
            fig = plot_training_history(history)
            st.pyplot(fig)
            
            st.subheader("Model Performance Metrics")
            df = pd.DataFrame([{k:v for k,v in metrics.items() if k != 'Confusion Matrix'}])
            st.table(df.style.format("{:.2%}").highlight_max(axis=0))
            
            st.subheader("Confusion Matrix")
            cm = metrics['Confusion Matrix']
            fig, ax = plt.subplots(figsize=(5,5))
            ax.matshow(cm, cmap='Blues', alpha=0.5)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticks([0,1])
            ax.set_yticks([0,1])
            ax.set_xticklabels(['No Tumor', 'Tumor'])
            ax.set_yticklabels(['No Tumor', 'Tumor'])
            st.pyplot(fig)
                        st.session_state.model = model

elif page == "Prediction":
    st.header("Enhanced Tumor Prediction")
    
    uploaded_file = st.file_uploader("Upload an MRI image for tumor detection", 
                                   type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load model if not already loaded
        if 'model' not in st.session_state:
            if os.path.exists("models/best_model.h5"):
                st.session_state.model = load_model("models/best_model.h5")
            else:
                st.error("Please train the model first from the Model Training page")
                st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"MRI Scan\nSize: {image.size}", 
                    use_column_width=True)
            
        with col2:
            st.subheader("Prediction")
            with st.spinner("Analyzing image with enhanced model..."):
                processed_img = preprocess_image(np.array(image))
                prediction, confidence = predict_tumor(st.session_state.model, processed_img)
                
                if "Tumor" in prediction:
                    st.error(f"**{prediction}** (Confidence: {confidence:.2%})")
                elif "Uncertain" in prediction:
                    st.warning(f"**{prediction}** (Confidence: {confidence:.2%})")
                else:
                    st.success(f"**{prediction}** (Confidence: {confidence:.2%})")
                
                st.image(processed_img, caption="Processed Image", 
                        use_column_width=True, channels="RGB")
                
        st.subheader("Model Information")
        st.write("""
        **Enhanced CNN Architecture:**
        - 3 Convolutional Blocks with Batch Normalization
        - MaxPooling and increasing Dropout (0.2-0.5)
        - 256-unit Dense layer
        - Output with sigmoid activation
        - Trained with early stopping and learning rate reduction
        """)

# Footer
st.markdown("---")
st.markdown("""
**Enhanced Brain MRI Tumor Classification**  
*AIMIL Ltd. Data Scientist Assignment - Rigved Sarougi*  
[GitHub Repository](https://github.com/yourusername/brain-mri-tumor-classification)
""")
