#!/usr/bin/env python3
"""
Test script to verify EfficientNet preprocessing is working correctly.
This script compares the preprocessing used in training vs prediction.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Force CPU before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

print("=" * 70)
print("🔬 EfficientNet Preprocessing Verification Test")
print("=" * 70)

# Test image path - using the original dog image you mentioned
test_image_path = "data/test/cane/OIP-TpmRCG2TNjevsaBGU8LWcwHaE7.jpeg"
model_path = "models/efficientnet_transfer_best_09725_val_09612_trian.h5"

# Check if files exist
if not os.path.exists(test_image_path):
    print(f"❌ Test image not found: {test_image_path}")
    # Try to find any dog image
    cane_dir = "data/test/cane"
    if os.path.exists(cane_dir):
        images = [f for f in os.listdir(cane_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            test_image_path = os.path.join(cane_dir, images[0])
            print(f"✅ Using alternative dog image: {test_image_path}")
        else:
            print("❌ No dog images found")
            sys.exit(1)
    else:
        print("❌ Dog image directory not found")
        sys.exit(1)

if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    sys.exit(1)

print(f"✅ Test image: {test_image_path}")
print(f"✅ Model file: {model_path}")

try:
    # Load image
    image = Image.open(test_image_path)
    print(f"\n📷 Image Information:")
    print(f"   - Original size: {image.size}")
    print(f"   - Mode: {image.mode}")
    print(f"   - Format: {image.format}")
    
    # Test 1: Manual preprocessing (what our app does)
    print(f"\n🔧 Test 1: Manual EfficientNet Preprocessing")
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image_rgb = image.convert('RGB')
    else:
        image_rgb = image
    
    # Resize to 224x224
    image_resized = image_rgb.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image_resized, dtype=np.float32)
    print(f"   - After resize: {img_array.shape}")
    print(f"   - Value range before preprocessing: [{img_array.min():.2f}, {img_array.max():.2f}]")
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply EfficientNet preprocessing
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_array)
    print(f"   - After EfficientNet preprocessing: {img_preprocessed.shape}")
    print(f"   - Value range after preprocessing: [{img_preprocessed.numpy().min():.2f}, {img_preprocessed.numpy().max():.2f}]")
    
    
    )
    
    # Create a temporary directory with our image for the generator
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a class directory
        class_dir = os.path.join(temp_dir, 'test_class')
        os.makedirs(class_dir)
        
        # Copy image to temp directory
        temp_image_path = os.path.join(class_dir, 'test_image.jpg')
        image_resized.save(temp_image_path)
        
        # Create generator
        test_generator = train_datagen.flow_from_directory(
            temp_dir,
            target_size=(224, 224),
            batch_size=1,
            class_mode=None,
            shuffle=False
        )
        
        # Get preprocessed image from generator
        generator_batch = next(test_generator)
        
        print(f"   - Generator output shape: {generator_batch.shape}")
        print(f"   - Value range: [{generator_batch.min():.2f}, {generator_batch.max():.2f}]")
        print(f"   - Mean: {generator_batch.mean():.4f}")
        print(f"   - Std: {generator_batch.std():.4f}")
    
    # Test 3: Compare the two methods
    print(f"\n🔍 Test 3: Comparison")
    
    # Check if they're the same
    diff = np.abs(img_preprocessed.numpy() - generator_batch)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"   - Maximum difference: {max_diff:.8f}")
    print(f"   - Mean difference: {mean_diff:.8f}")
    
    if max_diff < 1e-5:
        print("   - ✅ Preprocessing methods are identical!")
    else:
        print("   - ⚠️  Preprocessing methods differ!")
    
    # Test 4: Test with our AnimalClassifier
    print(f"\n🤖 Test 4: AnimalClassifier Prediction")
    
    from utils import AnimalClassifier
    
    # Initialize the classifier
    classifier = AnimalClassifier(model_path)
    print("   - ✅ AnimalClassifier initialized")
    
    # Make prediction
    results = classifier.predict(image, top_k=5)
    
    print(f"\n🎯 Prediction Results:")
    print("-" * 40)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['class'].upper()}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Percentage: {result['percentage']}")
    
    # Expected result check (since this is from cane folder = dog)
    top_prediction = results[0]['class'].lower()
    expected = 'dog'
    
    if top_prediction == expected:
        print(f"\n✅ SUCCESS! Correctly predicted '{top_prediction}' (expected: '{expected}')")
    else:
        print(f"\n⚠️  Top prediction is '{top_prediction}' (expected: '{expected}')")
        print(f"   This might indicate a preprocessing or model issue.")
    
    print(f"\n🏆 Top prediction: {results[0]['class'].upper()} ({results[0]['percentage']})")
    
    # Test 5: Direct model prediction with manual preprocessing
    print(f"\n🔧 Test 5: Direct Model Prediction")
    
    model = tf.keras.models.load_model(model_path)
    direct_prediction = model.predict(img_preprocessed, verbose=0)
    
    print(f"   - Direct model prediction shape: {direct_prediction.shape}")
    print(f"   - Prediction sum: {direct_prediction.sum():.6f} (should be ~1.0)")
    
    # Get class names from the classifier
    direct_predicted_idx = np.argmax(direct_prediction)
    direct_confidence = direct_prediction[0][direct_predicted_idx]
    direct_class = classifier.class_names[direct_predicted_idx]
    
    print(f"   - Direct prediction: {direct_class} ({direct_confidence:.4f})")
    print(f"   - Matches classifier: {'✅' if direct_class == results[0]['class'] else '❌'}")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure TensorFlow and other dependencies are installed.")
except Exception as e:
    print(f"❌ Error during test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Preprocessing verification completed!")
print("=" * 70)