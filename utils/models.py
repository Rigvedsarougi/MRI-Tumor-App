import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, 
                                    Dense, Dropout, Input, 
                                    Add, BatchNormalization, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

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

def create_vgg_like(input_shape=(256, 256, 3)):
    """Create a VGG-inspired model."""
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Classification block
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    """Create a residual block for ResNet."""
    # Shortcut
    shortcut = x
    
    # Main path
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut to main path
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def create_resnet(input_shape=(256, 256, 3)):
    """Create a small ResNet model."""
    inputs = Input(shape=input_shape)
    
    # Initial conv layer
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, strides=(2, 2))
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, strides=(2, 2))
    x = residual_block(x, 256)
    
    # Final layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, x)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_models(data_dir, epochs=15, batch_size=32):
    """Train all three models and return their histories and metrics."""
    # Create data generators
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
    
    # Initialize models
    models = {
        'Basic CNN': create_basic_cnn(),
        'VGG-like': create_vgg_like(),
        'ResNet': create_resnet()
    }
    
    histories = {}
    metrics = {}
    
    # Train each model
    for name, model in models.items():
        print(f"Training {name}...")
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=1
        )
        histories[name] = history
        
        # Evaluate
        val_pred = (model.predict(val_gen) > 0.5).astype(int)
        val_true = val_gen.classes
        
        metrics[name] = {
            'Accuracy': accuracy_score(val_true, val_pred),
            'Precision': precision_score(val_true, val_pred),
            'Recall': recall_score(val_true, val_pred),
            'F1 Score': f1_score(val_true, val_pred)
        }
    
    return histories, metrics

def load_model(model_name):
    """Load a pre-trained model by name."""
    if model_name == "Basic CNN":
        model = create_basic_cnn()
    elif model_name == "VGG-like":
        model = create_vgg_like()
    elif model_name == "ResNet":
        model = create_resnet()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # In a real app, you would load pre-trained weights here
    # model.load_weights(f"models/{model_name.lower().replace(' ', '_')}.h5")
    
    return model

def predict_tumor(model_name, image):
    """Predict tumor presence using the specified model."""
    model = load_model(model_name)
    
    # Preprocess and predict
    img_array = np.expand_dims(image, axis=0)
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        return "Tumor Detected", prediction
    else:
        return "No Tumor Detected", 1 - prediction
