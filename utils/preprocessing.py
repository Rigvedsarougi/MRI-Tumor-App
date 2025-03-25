import os
import numpy as np
from PIL import Image
import cv2
import random

def explore_dataset(data_path):
    """Load and return sample images from the dataset."""
    samples = []
    
    # Get random samples from both classes
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
    """Preprocess a single MRI image."""
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image / 255.0
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply((image * 255).astype(np.uint8)) / 255.0
    
    # Denoising
    image = cv2.medianBlur((image * 255).astype(np.uint8), 3) / 255.0
    
    # Edge enhancement
    kernel = np.array([[-1, -1, -1], 
                       [-1, 9, -1], 
                       [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)
    
    # Convert back to 3 channels for model input
    image = np.stack((image,)*3, axis=-1)
    
    return image
