#!/usr/bin/env python3
"""
Debug script to test different preprocessing approaches and compare results.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_preprocessing_approaches(image_path, model_path):
    """Test different preprocessing approaches to debug the issue."""
    
    print("🔍 Testing Different Preprocessing Approaches")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Class names (in the order the model expects)
    class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
                   'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
    
    translate = {
        "cane": "dog", "cavallo": "horse", "elefante": "elephant", 
        "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", 
        "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", 
        "ragno": "spider"
    }
    
    def preprocess_v1(path):
        """Original approach with nearest neighbor"""
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (224, 224), method='nearest')
        image = tf.expand_dims(image, 0)
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image
    
    def preprocess_v2(path):
        """Fixed approach with bilinear and proper casting order"""
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224), method='bilinear')
        image = tf.expand_dims(image, 0)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image
    
    def preprocess_v3(path):
        """Without EfficientNet preprocessing (in case model has it built-in)"""
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224), method='bilinear')
        image = tf.expand_dims(image, 0)
        # Normalize to [0, 1] without EfficientNet preprocessing
        image = image / 255.0
        return image
    
    def preprocess_v4(path):
        """Raw float32 without any normalization"""
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224), method='bilinear')
        image = tf.expand_dims(image, 0)
        # Keep [0, 255] range
        return image
    
    approaches = [
        ("V1: Original (nearest + EfficientNet)", preprocess_v1),
        ("V2: Fixed (bilinear + EfficientNet)", preprocess_v2),
        ("V3: Without EfficientNet preprocessing", preprocess_v3),
        ("V4: Raw float32 [0, 255]", preprocess_v4)
    ]
    
    for name, preprocess_fn in approaches:
        print(f"\n🧪 Testing {name}")
        print("-" * 40)
        
        try:
            # Preprocess image
            processed = preprocess_fn(image_path)
            print(f"   Shape: {processed.shape}")
            print(f"   Value range: [{tf.reduce_min(processed):.3f}, {tf.reduce_max(processed):.3f}]")
            print(f"   Mean: {tf.reduce_mean(processed):.3f}")
            print(f"   Std: {tf.math.reduce_std(processed):.3f}")
            
            # Make prediction
            predictions = model.predict(processed, verbose=0)
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            
            print("   Top 3 predictions:")
            for i, idx in enumerate(top_indices, 1):
                confidence = predictions[0][idx]
                italian_name = class_names[idx]
                english_name = translate[italian_name]
                print(f"   {i}. {english_name} ({italian_name}): {confidence:.4f} ({confidence*100:.2f}%)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n💡 Expected result: DOG (from 'cane' folder)")
    print("=" * 60)

if __name__ == "__main__":
    image_path = "data/test/cane/OIP-TpmRCG2TNjevsaBGU8LWcwHaE7.jpeg"
    model_path = "models/efficientnet_transfer_best_09725_val_09612_trian.h5"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    test_preprocessing_approaches(image_path, model_path)