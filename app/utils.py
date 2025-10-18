"""
Utility functions for the Flask application.

This module contains helper functions for model loading, image preprocessing,
and prediction generation.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from typing import Tuple, List
import os


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained Keras model from file.
    
    Args:
        model_path: Path to the saved model (.h5 file)
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def preprocess_image(image: Image.Image, 
                    target_size: Tuple[int, int] = (32, 32),
                    normalize: bool = True) -> np.ndarray:
    """
    Preprocess an image for model prediction.
    
    Args:
        image: PIL Image object
        target_size: Target size (width, height) for resizing
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image array ready for prediction
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize if requested
    if normalize:
        image_array = image_array.astype('float32') / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def get_prediction(model: tf.keras.Model,
                  image: Image.Image,
                  class_names: List[str],
                  input_size: Tuple[int, int]) -> Tuple[str, float, np.ndarray]:
    """
    Get prediction for a single image.
    
    Args:
        model: Trained Keras model
        image: PIL Image object
        class_names: List of class names
        input_size: Expected input size for the model
        
    Returns:
        Tuple of (predicted_class, confidence, all_probabilities)
    """
    # Preprocess the image
    processed_image = preprocess_image(image, input_size)
    
    # Get prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]


def batch_predict(model: tf.keras.Model,
                 images: List[Image.Image],
                 class_names: List[str],
                 input_size: Tuple[int, int]) -> List[Tuple[str, float, np.ndarray]]:
    """
    Get predictions for multiple images in batch.
    
    Args:
        model: Trained Keras model
        images: List of PIL Image objects
        class_names: List of class names
        input_size: Expected input size for the model
        
    Returns:
        List of tuples (predicted_class, confidence, all_probabilities)
    """
    # Preprocess all images
    processed_images = np.vstack([
        preprocess_image(image, input_size) for image in images
    ])
    
    # Get batch predictions
    predictions = model.predict(processed_images, verbose=0)
    
    results = []
    for pred in predictions:
        predicted_class_idx = np.argmax(pred)
        confidence = pred[predicted_class_idx]
        predicted_class = class_names[predicted_class_idx]
        results.append((predicted_class, confidence, pred))
    
    return results


def validate_image(image: Image.Image) -> bool:
    """
    Validate that the image is suitable for processing.
    
    Args:
        image: PIL Image object
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        # Check if image can be converted to RGB
        image.convert('RGB')
        
        # Check minimum size
        if image.size[0] < 32 or image.size[1] < 32:
            return False
        
        # Check maximum size (reasonable limit)
        if image.size[0] > 4096 or image.size[1] > 4096:
            return False
        
        return True
    except Exception:
        return False


def get_top_predictions(probabilities: np.ndarray,
                       class_names: List[str],
                       top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Get top-k predictions with their probabilities.
    
    Args:
        probabilities: Array of class probabilities
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        List of (class_name, probability) tuples sorted by probability
    """
    # Get indices of top-k predictions
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    # Create list of (class_name, probability) tuples
    top_predictions = [
        (class_names[idx], probabilities[idx]) 
        for idx in top_indices
    ]
    
    return top_predictions


def format_confidence(confidence: float) -> str:
    """
    Format confidence as percentage string.
    
    Args:
        confidence: Confidence value between 0 and 1
        
    Returns:
        Formatted percentage string
    """
    return f"{confidence * 100:.1f}%"


def log_prediction(image_filename: str,
                  predicted_class: str,
                  confidence: float,
                  log_file: str = 'prediction_log.txt') -> None:
    """
    Log prediction results to a file.
    
    Args:
        image_filename: Name of the processed image file
        predicted_class: Predicted class
        confidence: Prediction confidence
        log_file: Path to log file
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} | {image_filename} | {predicted_class} | {confidence:.4f}\\n"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)