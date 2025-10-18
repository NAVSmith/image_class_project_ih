"""
Data loading and preprocessing utilities for image classification.

This module provides functions for loading datasets, preprocessing images,
and creating data augmentation pipelines.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, Optional


def load_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load CIFAR-10 dataset.
    
    Returns:
        Tuple containing ((x_train, y_train), (x_test, y_test))
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return (x_train, y_train), (x_test, y_test), class_names


def load_animals10(data_dir: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load Animals10 dataset from directory.
    
    Args:
        data_dir: Path to the Animals10 dataset directory
        
    Returns:
        Tuple containing ((x_train, y_train), (x_test, y_test))
    """
    # This would need to be implemented based on the Animals10 dataset structure
    # Using tf.keras.preprocessing.image_dataset_from_directory or custom loading
    
    class_names = ['dog', 'cat', 'horse', 'spider', 'butterfly', 
                   'chicken', 'sheep', 'cow', 'squirrel', 'elephant']
    
    # Placeholder implementation - replace with actual loading logic
    raise NotImplementedError("Animals10 loading not implemented yet")


def normalize_images(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        x_train: Training images
        x_test: Test images
        
    Returns:
        Normalized training and test images
    """
    x_train_norm = x_train.astype('float32') / 255.0
    x_test_norm = x_test.astype('float32') / 255.0
    
    return x_train_norm, x_test_norm


def prepare_labels(y_train: np.ndarray, y_test: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert labels to categorical one-hot encoding.
    
    Args:
        y_train: Training labels
        y_test: Test labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded training and test labels
    """
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    return y_train_cat, y_test_cat


def create_train_val_split(x_train: np.ndarray, y_train: np.ndarray, 
                          val_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training and validation splits.
    
    Args:
        x_train: Training images
        y_train: Training labels
        val_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        x_train_split, x_val_split, y_train_split, y_val_split
    """
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        x_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )
    
    return x_train_split, x_val_split, y_train_split, y_val_split


def create_data_augmentation(rotation_range: int = 15,
                           width_shift_range: float = 0.1,
                           height_shift_range: float = 0.1,
                           horizontal_flip: bool = True,
                           zoom_range: float = 0.1) -> ImageDataGenerator:
    """
    Create data augmentation pipeline for training.
    
    Args:
        rotation_range: Range of degrees for random rotations
        width_shift_range: Fraction of total width for horizontal shifts
        height_shift_range: Fraction of total height for vertical shifts
        horizontal_flip: Whether to randomly flip images horizontally
        zoom_range: Range for random zoom
        
    Returns:
        Configured ImageDataGenerator
    """
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        zoom_range=zoom_range,
        fill_mode='nearest'
    )
    
    return datagen


def resize_for_transfer_learning(images: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize images for transfer learning models that expect larger input sizes.
    
    Args:
        images: Input images
        target_size: Target size (height, width)
        
    Returns:
        Resized images
    """
    resized = tf.image.resize(images, target_size)
    return resized.numpy()


def calculate_class_weights(y_train: np.ndarray) -> dict:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train.flatten())
    
    return dict(zip(classes, class_weights))


def get_dataset_statistics(x_train: np.ndarray, y_train: np.ndarray, 
                          x_test: np.ndarray, y_test: np.ndarray, 
                          class_names: list) -> dict:
    """
    Get comprehensive dataset statistics.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        class_names: List of class names
        
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'num_train_samples': len(x_train),
        'num_test_samples': len(x_test),
        'image_shape': x_train.shape[1:],
        'num_classes': len(class_names),
        'class_names': class_names,
        'pixel_range': (x_train.min(), x_train.max()),
        'data_type': str(x_train.dtype),
        'class_distribution_train': {
            class_names[i]: np.sum(y_train == i) for i in range(len(class_names))
        },
        'class_distribution_test': {
            class_names[i]: np.sum(y_test == i) for i in range(len(class_names))
        }
    }
    
    return stats