#!/usr/bin/env python3
"""
Test script to verify CPU-only TensorFlow configuration.
"""

import os
import sys

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

print("Testing CPU-only configuration...")

# Force CPU before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    
    # Configure TensorFlow to use CPU only
    tf.config.set_visible_devices([], 'GPU')
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Available devices: {[device.name for device in tf.config.list_physical_devices()]}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(f"CPU devices: {tf.config.list_physical_devices('CPU')}")
    
    # Test basic tensor operation on CPU
    with tf.device('/CPU:0'):
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"CPU tensor operation result: {c.numpy()}")
    
    print("✅ CPU-only configuration working correctly!")
    
    # Test model loading (if available)
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'efficientnet_transfer_best_09725_val_09612_trian.h5')
    if os.path.exists(model_path):
        print(f"Testing model loading from: {model_path}")
        try:
            with tf.device('/CPU:0'):
                model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully on CPU!")
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️  Model file not found at: {model_path}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")