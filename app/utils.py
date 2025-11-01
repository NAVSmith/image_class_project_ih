"""
Utility functions for the Flask application.

This module contains helper functions for model loading, image preprocessing,
and prediction generation using preprocess_v1 approach.
"""

import numpy as np
import os
import logging

# Force TensorFlow to use CPU only for prediction
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
from PIL import Image
from typing import Tuple, List, Dict

# Configure TensorFlow to use CPU only
# tf.config.set_visible_devices([], 'GPU')

# Set TensorFlow to use single thread for better consistency on CPU
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnimalClassifier:
    """
    Animal image classifier that loads model and handles predictions.
    
    This class encapsulates the model loading, preprocessing, and prediction
    logic for the animal classification system using EfficientNet with 
    preprocess_v1 approach (nearest neighbor + EfficientNet preprocessing).
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the classifier with a trained model.
        
        Args:
            model_path: Path to the saved model (.h5 file)
        """
        self.model_path = model_path
        self.model = None
        
        # Italian to English class name translation
        self.translate = {
            "cane": "dog", 
            "cavallo": "horse", 
            "elefante": "elephant", 
            "farfalla": "butterfly", 
            "gallina": "chicken", 
            "gatto": "cat", 
            "mucca": "cow", 
            "pecora": "sheep", 
            "scoiattolo": "squirrel", 
            "ragno": "spider"
        }
        
        # Original Italian class names (as they appear in the model)
        self.original_class_names = [
            'cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
            'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
        ]
        
        # Translated English class names
        self.class_names = [self.translate[name] for name in self.original_class_names]
        
        self.input_size = (224, 224)  # EfficientNet input size
        self.num_classes = len(self.class_names)
        
        # Load the model
        self._load_model()
        logger.info(f"AnimalClassifier initialized with {self.num_classes} classes")
        logger.info(f"Class names: {self.class_names}")
    
    def _load_model(self) -> None:
        """Load the trained model from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Ensure CPU-only execution during model loading and inference
            # with tf.device('/CPU:0'):
            self.model = tf.keras.models.load_model(self.model_path)
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            logger.info("TensorFlow model loaded")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_image_from_path(self, image_path: str) -> tf.Tensor:
        """
        Preprocess an image from file path using preprocess_v1 approach.
        
        preprocess_v1: nearest neighbor resizing + EfficientNet preprocessing
        - Read image from path
        - Resize with nearest neighbor method
        - Expand dims for batch
        - Cast to float32
        - Apply EfficientNet preprocessing
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor ready for prediction
        """
        try:
            # Read and decode image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)  # yields uint8
            
            # Resize using nearest neighbor method (preprocess_v1 approach)
            image = tf.image.resize(image, self.input_size, method='nearest')
            
            # Add batch dimension
            image = tf.expand_dims(image, 0)
            
            # Cast to float32 after resize and expand_dims
            image = tf.cast(image, tf.float32)
            
            # Apply EfficientNet preprocessing
            image = tf.keras.applications.efficientnet.preprocess_input(image)
            
            logger.info(f"Image preprocessed from path: {image.shape}, range: [{tf.reduce_min(image):.3f}, {tf.reduce_max(image):.3f}]")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image from path: {e}")
            raise ValueError(f"Failed to preprocess image from path: {e}")
    
    def preprocess_pil_image(self, pil_image: Image.Image) -> tf.Tensor:
        """
        Preprocess a PIL Image using preprocess_v1 approach.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Preprocessed image tensor ready for prediction
        """
        try:
            # Convert PIL image to RGB if not already
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL to numpy array
            image_array = np.array(pil_image, dtype=np.uint8)
            
            # Convert to TensorFlow tensor
            image = tf.constant(image_array)
            
            # Resize using nearest neighbor method (preprocess_v1 approach)
            image = tf.image.resize(image, self.input_size, method='nearest')
            
            # Add batch dimension
            image = tf.expand_dims(image, 0)
            
            # Cast to float32 after resize and expand_dims
            image = tf.cast(image, tf.float32)
            
            # Apply EfficientNet preprocessing
            image = tf.keras.applications.efficientnet.preprocess_input(image)
            
            logger.info(f"PIL image preprocessed: {image.shape}, range: [{tf.reduce_min(image):.3f}, {tf.reduce_max(image):.3f}]")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing PIL image: {e}")
            raise ValueError(f"Failed to preprocess PIL image: {e}")
    
    def predict(self, image_input, top_k: int = 3) -> List[Dict]:
        """
        Predict the class of an image.
        
        Args:
            image_input: Either string (file path), PIL Image, or preprocessed tensor
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with class predictions and confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Determine input type and preprocess accordingly
            if isinstance(image_input, str):
                # File path
                processed_image = self.preprocess_image_from_path(image_input)
                logger.info("Preprocessed image from file path")
            elif isinstance(image_input, Image.Image):
                # PIL Image
                processed_image = self.preprocess_pil_image(image_input)
                logger.info("Preprocessed PIL image")
            elif isinstance(image_input, (tf.Tensor, np.ndarray)):
                # Already preprocessed
                processed_image = image_input
                logger.info("Using already preprocessed image")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Make prediction using CPU only
            # with tf.device('/CPU:0'):
            predictions = self.model.predict(processed_image, verbose=0)
            
            logger.info(f"Predictions shape: {predictions.shape}")
            logger.info(f"Top prediction index: {np.argmax(predictions, axis=1)[0]}")
            
            # Get top-k predictions
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                confidence = float(predictions[0][idx])
                class_name = self.class_names[idx]
                
                results.append({
                    'class': class_name,
                    'confidence': confidence,
                    'percentage': f"{confidence * 100:.2f}%"
                })
            
            logger.info(f"Prediction completed. Top class: {results[0]['class']} ({results[0]['percentage']})")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Failed to make prediction: {e}")
    
    def get_best_prediction(self, image_input) -> str:
        """
        Get the best (highest confidence) prediction for an image.
        
        Args:
            image_input: String (file path), PIL Image, or preprocessed tensor
            
        Returns:
            String name of the predicted class
        """
        predictions = self.predict(image_input, top_k=1)
        return predictions[0]['class']


# Legacy functions for backward compatibility
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
        # with tf.device('/CPU:0'):
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def preprocess_image_v1(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> tf.Tensor:
    """
    Standalone function for preprocess_v1 approach.
    
    Args:
        image_path: Path to image file
        target_size: Target size (width, height) for resizing
        
    Returns:
        Preprocessed image tensor ready for prediction
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, target_size, method='nearest')
    image = tf.expand_dims(image, 0)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image


if __name__ == "__main__":
    # Initialize classifier
    classifier = AnimalClassifier("models/efficientnet_transfer_best_09725_val_09612_trian.h5")

    # Predict from file path
    results = classifier.predict("/Users/smithn5/ironhack/image_class_project_ih/data/test/cane/OIP-TpmRCG2TNjevsaBGU8LWcwHaE7.jpeg", top_k=3)
    for res in results:
        print(f"Class: {res['class']}, Confidence: {res['percentage']}")
    
    # Get just the best prediction
    best_class = classifier.get_best_prediction("/Users/smithn5/ironhack/image_class_project_ih/data/test/cane/OIP-TpmRCG2TNjevsaBGU8LWcwHaE7.jpeg")

