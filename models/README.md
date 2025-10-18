# Models Directory

This directory contains saved trained models and related files.

## File Structure

- `*.h5` - Keras model files (best weights and final models)
- `*.pkl` - Pickled training histories and results
- `*_config.pkl` - Training configuration files

## Naming Convention

### Custom CNN Models

- `custom_cnn_cifar10_best.h5` - Best weights during training
- `custom_cnn_cifar10_final.h5` - Final model after training
- `custom_cnn_cifar10_history.pkl` - Training history

### Transfer Learning Models

- `vgg16_transfer_cifar10_best.h5` - Best VGG16 transfer learning weights
- `resnet50_transfer_cifar10_best.h5` - Best ResNet50 transfer learning weights
- `efficientnet_transfer_cifar10_best.h5` - Best EfficientNet transfer learning weights

### Fine-tuned Models

- `vgg16_finetuned_best.h5` - Fine-tuned model weights
- `best_transfer_model_cifar10.h5` - Best overall transfer learning model

## Usage

Load models using TensorFlow/Keras:

```python
import tensorflow as tf

# Load a saved model
model = tf.keras.models.load_model('custom_cnn_cifar10_final.h5')

# Load training history
import pickle
with open('custom_cnn_cifar10_history.pkl', 'rb') as f:
    history = pickle.load(f)
```

## Model Performance Tracking

Keep track of model performance in this directory:

| Model             | Dataset  | Validation Acc | Test Acc | Parameters | Training Time |
| ----------------- | -------- | -------------- | -------- | ---------- | ------------- |
| Custom CNN        | CIFAR-10 | 0.XX           | 0.XX     | XXX,XXX    | XX min        |
| VGG16 Transfer    | CIFAR-10 | 0.XX           | 0.XX     | XX,XXX,XXX | XX min        |
| ResNet50 Transfer | CIFAR-10 | 0.XX           | 0.XX     | XX,XXX,XXX | XX min        |

Note: This directory is excluded from version control to avoid large file commits.
