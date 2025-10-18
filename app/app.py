"""
Flask web application for image classification model deployment.

This application provides a web interface for uploading images and getting
predictions from the trained CNN model.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import json
from typing import List, Dict, Tuple
from utils import load_model, preprocess_image, get_prediction

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and class names
model = None
class_names = None
input_size = None


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and return predictions."""
    if 'files' not in request.files:
        flash('No files selected')
        return redirect(request.url)
    
    files = request.files.getlist('files')
    
    if not files or all(file.filename == '' for file in files):
        flash('No files selected')
        return redirect(request.url)
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Secure the filename
                filename = secure_filename(file.filename)
                
                # Read image directly from memory
                image = Image.open(file.stream)
                
                # Get prediction
                prediction, confidence, all_probabilities = get_prediction(
                    model, image, class_names, input_size
                )
                
                results.append({
                    'filename': filename,
                    'prediction': prediction,
                    'confidence': float(confidence),
                    'all_probabilities': {
                        class_names[i]: float(prob) 
                        for i, prob in enumerate(all_probabilities)
                    }
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        else:
            results.append({
                'filename': file.filename,
                'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.'
            })
    
    return render_template('results.html', results=results)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON response)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Read and process image
        image = Image.open(file.stream)
        prediction, confidence, all_probabilities = get_prediction(
            model, image, class_names, input_size
        )
        
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'all_probabilities': {
                class_names[i]: float(prob) 
                for i, prob in enumerate(all_probabilities)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(class_names) if class_names else 0
    })


@app.route('/model_info')
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({'error': 'No model loaded'}), 500
    
    return jsonify({
        'model_type': 'CNN Image Classifier',
        'input_shape': input_size,
        'num_classes': len(class_names),
        'class_names': class_names,
        'model_parameters': model.count_params() if hasattr(model, 'count_params') else 'Unknown'
    })


def initialize_app(model_path: str, dataset_type: str = 'cifar10'):
    """
    Initialize the Flask app with the trained model.
    
    Args:
        model_path: Path to the saved model
        dataset_type: Type of dataset ('cifar10' or 'animals10')
    """
    global model, class_names, input_size
    
    try:
        # Load the model
        model = load_model(model_path)
        
        # Set class names based on dataset
        if dataset_type.lower() == 'cifar10':
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
            input_size = (32, 32)  # Original CIFAR-10 size
        elif dataset_type.lower() == 'animals10':
            class_names = ['dog', 'cat', 'horse', 'spider', 'butterfly', 
                          'chicken', 'sheep', 'cow', 'squirrel', 'elephant']
            input_size = (224, 224)  # Typical transfer learning size
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Try to infer input size from model if possible
        if hasattr(model, 'input_shape') and model.input_shape:
            input_size = model.input_shape[1:3]  # (height, width)
        
        print(f"Model loaded successfully!")
        print(f"Dataset: {dataset_type}")
        print(f"Classes: {len(class_names)}")
        print(f"Input size: {input_size}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the image classification web app')
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'animals10'],
                       help='Dataset type (default: cifar10)')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the app on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the app on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Initialize the app with the model
    initialize_app(args.model, args.dataset)
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug)