import matplotlib.pyplot as plt
import numpy as np

def plot_sample_images(samples, figsize=(12, 8)):
    """Plot sample images from the dataset."""
    plt.figure(figsize=figsize)
    
    for i, (image, label) in enumerate(samples):
        plt.subplot(2, 4, i+1)
        plt.imshow(image, cmap='gray' if len(np.array(image).shape) == 2 else None)
        plt.title(f"Class: {label}")
        plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def plot_metrics(histories, figsize=(12, 8)):
    """Plot training metrics for all models."""
    plt.figure(figsize=figsize)
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} Train')
        plt.plot(history.history['val_accuracy'], '--', label=f'{name} Val')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} Train')
        plt.plot(history.history['val_loss'], '--', label=f'{name} Val')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf()import matplotlib.pyplot as plt
import numpy as np

def plot_sample_images(samples, figsize=(12, 8)):
    """Plot sample images from the dataset."""
    plt.figure(figsize=figsize)
    
    for i, (image, label) in enumerate(samples):
        plt.subplot(2, 4, i+1)
        plt.imshow(image, cmap='gray' if len(np.array(image).shape) == 2 else None)
        plt.title(f"Class: {label}")
        plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def plot_metrics(histories, figsize=(12, 8)):
    """Plot training metrics for all models."""
    plt.figure(figsize=figsize)
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} Train')
        plt.plot(history.history['val_accuracy'], '--', label=f'{name} Val')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} Train')
        plt.plot(history.history['val_loss'], '--', label=f'{name} Val')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf()
