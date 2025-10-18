"""
CNN model architectures for image classification.

This module provides functions to create custom CNN architectures
and transfer learning models.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from typing import Tuple, Optional


def create_custom_cnn(input_shape: Tuple[int, int, int], 
                     num_classes: int,
                     dropout_rate: float = 0.5) -> tf.keras.Model:
    """
    Create a custom CNN architecture for image classification.
    
    Architecture follows the pattern: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_smaller_cnn(input_shape: Tuple[int, int, int], 
                      num_classes: int) -> tf.keras.Model:
    """
    Create a smaller CNN for faster training/testing.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        
    Returns:
        Keras model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_transfer_learning_model(base_model_name: str, 
                                 input_shape: Tuple[int, int, int], 
                                 num_classes: int,
                                 trainable: bool = False,
                                 dense_units: int = 256) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Create a transfer learning model with specified base architecture.
    
    Args:
        base_model_name: 'vgg16', 'resnet50', or 'efficientnet'
        input_shape: Input image shape (should be >= 224x224 for pretrained models)
        num_classes: Number of output classes
        trainable: Whether to make base model trainable (for fine-tuning)
        dense_units: Number of units in the dense layer
        
    Returns:
        Tuple of (complete_model, base_model)
    """
    
    # Select base model
    if base_model_name.lower() == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name.lower() == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name.lower() in ['efficientnet', 'efficientnetb0']:
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze or unfreeze base model
    base_model.trainable = trainable
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model


def get_model_summary(model: tf.keras.Model) -> dict:
    """
    Get comprehensive model summary information.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': int(trainable_params),
        'non_trainable_parameters': int(non_trainable_params),
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Rough estimate
        'num_layers': len(model.layers)
    }


def compile_model(model: tf.keras.Model, 
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 loss: str = 'categorical_crossentropy',
                 metrics: list = None) -> tf.keras.Model:
    """
    Compile a Keras model with specified parameters.
    
    Args:
        model: Keras model to compile
        optimizer: Optimizer name
        learning_rate: Learning rate
        loss: Loss function
        metrics: List of metrics to track
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy']
    
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = optimizer  # Assume it's already an optimizer instance
    
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    
    return model


def create_callbacks(model_save_path: str,
                    monitor: str = 'val_accuracy',
                    patience: int = 15,
                    reduce_lr_patience: int = 5,
                    min_lr: float = 1e-7) -> list:
    """
    Create standard callbacks for model training.
    
    Args:
        model_save_path: Path to save the best model
        monitor: Metric to monitor for callbacks
        patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        min_lr: Minimum learning rate
        
    Returns:
        List of callbacks
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        )
    ]
    
    return callbacks