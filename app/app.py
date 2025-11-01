"""
Flask web application for animal image classification.

This application provides a web interface for uploading animal images and getting
predictions from the trained EfficientNet model.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
import json
import logging
from typing import List, Dict, Tuple
from utils import AnimalClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the animal classifier
MODEL_PATH = '../models/efficientnet_transfer_best_09725_val_09612_trian.h5'
classifier = None

def initialize_classifier():
    """Initialize the animal classifier with the trained model."""
    global classifier
    try:
        classifier = AnimalClassifier(MODEL_PATH)
        logger.info("Animal classifier initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")

# Initialize classifier on startup
initialize_classifier()


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
        return redirect(url_for('index'))
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(url_for('index'))
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Load and process the image
                image = Image.open(file.stream)
                
                # Get prediction from the classifier
                predictions = classifier.predict(image, top_k=3)
                
                # Get the best prediction
                best_prediction = predictions[0]['class']
                confidence = predictions[0]['confidence']
                
                results.append({
                    'filename': secure_filename(file.filename),
                    'prediction': best_prediction,
                    'confidence': f"{confidence * 100:.1f}%",
                    'all_predictions': predictions
                })
                
                logger.info(f"Processed {file.filename}: {best_prediction} ({confidence * 100:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                flash(f'Error processing {file.filename}: {str(e)}')
                results.append({
                    'filename': secure_filename(file.filename),
                    'error': str(e)
                })
        else:
            flash(f'Invalid file type: {file.filename}')
    
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
        # Load and process the image
        image = Image.open(file.stream)
        
        # Get prediction from the classifier
        predictions = classifier.predict(image, top_k=3)
        
        return jsonify({
            'success': True,
            'prediction': predictions[0]['class'],
            'confidence': predictions[0]['confidence'],
            'all_predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'classifier_loaded': classifier is not None,
        'classes': len(classifier.class_names) if classifier else 0,
        'model_path': MODEL_PATH
    })


@app.route('/model_info')
def model_info():
    """Get model information."""
    if not classifier:
        return jsonify({'error': 'Classifier not initialized'}), 500
    
    return jsonify({
        'model_type': 'EfficientNet Transfer Learning',
        'input_size': classifier.input_size,
        'num_classes': classifier.num_classes,
        'class_names': classifier.class_names,
        'model_path': MODEL_PATH
    })


if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)