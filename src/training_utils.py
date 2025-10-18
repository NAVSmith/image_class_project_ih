"""
Training utilities and model evaluation functions.

This module provides functions for training models, evaluating performance,
and generating visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import pickle
import os


def train_model(model: tf.keras.Model,
               train_data,
               validation_data,
               epochs: int = 50,
               batch_size: int = 32,
               callbacks: Optional[List] = None,
               class_weights: Optional[Dict] = None,
               verbose: int = 1) -> tf.keras.callbacks.History:
    """
    Train a Keras model with the specified parameters.
    
    Args:
        model: Compiled Keras model
        train_data: Training data (generator or tuple)
        validation_data: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        callbacks: List of callbacks
        class_weights: Dictionary of class weights
        verbose: Verbosity level
        
    Returns:
        Training history
    """
    if callbacks is None:
        callbacks = []
    
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=verbose
    )
    
    return history


def evaluate_model(model: tf.keras.Model,
                  x_test: np.ndarray,
                  y_test: np.ndarray,
                  class_names: List[str],
                  verbose: int = 0) -> Dict:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels (one-hot encoded)
        class_names: List of class names
        verbose: Verbosity level
        
    Returns:
        Dictionary containing evaluation results
    """
    # Get model predictions
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    y_pred = model.predict(x_test, verbose=verbose)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    class_report = classification_report(
        y_true_classes, y_pred_classes, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    class_accuracy_dict = {
        class_names[i]: class_accuracy[i] for i in range(len(class_names))
    }
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'class_accuracy': class_accuracy_dict,
        'predictions': y_pred,
        'predicted_classes': y_pred_classes,
        'true_classes': y_true_classes
    }
    
    return results


def plot_training_history(history: tf.keras.callbacks.History,
                         save_path: Optional[str] = None) -> None:
    """
    Plot training and validation metrics over time.
    
    Args:
        history: Training history from model.fit()
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    epochs = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print best metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_val_acc_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print(f"\\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_val_acc_epoch}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str],
                         title: str = 'Confusion Matrix',
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sample_predictions(model: tf.keras.Model,
                           x_test: np.ndarray,
                           y_test: np.ndarray,
                           class_names: List[str],
                           num_samples: int = 12,
                           save_path: Optional[str] = None) -> None:
    """
    Plot sample predictions with true and predicted labels.
    
    Args:
        model: Trained model
        x_test: Test images
        y_test: Test labels (one-hot)
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Optional path to save the plot
    """
    # Get predictions
    y_pred = model.predict(x_test[:num_samples])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test[:num_samples], axis=1)
    
    # Create subplot grid
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Display image
        axes[i].imshow(x_test[i])
        
        # Get labels and confidence
        true_label = class_names[y_true_classes[i]]
        pred_label = class_names[y_pred_classes[i]]
        confidence = y_pred[i][y_pred_classes[i]]
        
        # Set title with color coding
        color = 'green' if y_true_classes[i] == y_pred_classes[i] else 'red'
        axes[i].set_title(f'True: {true_label}\\nPred: {pred_label}\\nConf: {confidence:.2f}', 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_results(results: Dict, history: tf.keras.callbacks.History,
                model_name: str, dataset_name: str, save_dir: str = '../models') -> None:
    """
    Save training results and history to files.
    
    Args:
        results: Evaluation results dictionary
        history: Training history
        model_name: Name of the model
        dataset_name: Name of the dataset
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save history
    history_path = os.path.join(save_dir, f'{model_name}_{dataset_name}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    # Save results (excluding large arrays)
    results_to_save = {k: v for k, v in results.items() 
                      if k not in ['predictions', 'confusion_matrix']}
    
    results_path = os.path.join(save_dir, f'{model_name}_{dataset_name}_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_to_save, f)
    
    print(f"Results saved to {save_dir}")


def compare_models(results_dict: Dict[str, Dict], 
                  metric: str = 'test_accuracy') -> None:
    """
    Compare multiple models based on a specific metric.
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
        metric: Metric to compare (e.g., 'test_accuracy')
    """
    print(f"\\nModel Comparison - {metric.title()}:")
    print("=" * 50)
    
    # Sort models by metric
    sorted_models = sorted(results_dict.items(), 
                          key=lambda x: x[1][metric], reverse=True)
    
    for i, (model_name, results) in enumerate(sorted_models, 1):
        print(f"{i}. {model_name}: {results[metric]:.4f}")
    
    print("=" * 50)
    
    # Create bar plot
    model_names = [name for name, _ in sorted_models]
    metric_values = [results[metric] for _, results in sorted_models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values, 
                  color=['gold' if i == 0 else 'skyblue' for i in range(len(model_names))])
    plt.title(f'Model Comparison - {metric.title()}')
    plt.ylabel(metric.title())
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def generate_model_report(model: tf.keras.Model,
                         results: Dict,
                         model_name: str,
                         dataset_name: str) -> str:
    """
    Generate a comprehensive text report for a model.
    
    Args:
        model: Trained model
        results: Evaluation results
        model_name: Name of the model
        dataset_name: Name of the dataset
        
    Returns:
        Report as a string
    """
    from src.model_architectures import get_model_summary
    
    model_info = get_model_summary(model)
    
    report = f"""
{'='*60}
MODEL PERFORMANCE REPORT
{'='*60}

Model: {model_name}
Dataset: {dataset_name}

ARCHITECTURE SUMMARY:
- Total Parameters: {model_info['total_parameters']:,}
- Trainable Parameters: {model_info['trainable_parameters']:,}
- Non-trainable Parameters: {model_info['non_trainable_parameters']:,}
- Model Size: {model_info['model_size_mb']:.2f} MB
- Number of Layers: {model_info['num_layers']}

PERFORMANCE METRICS:
- Test Accuracy: {results['test_accuracy']:.4f}
- Test Loss: {results['test_loss']:.4f}

CLASSIFICATION REPORT:
{results['classification_report']['macro avg']['precision']:.4f} (Macro Avg Precision)
{results['classification_report']['macro avg']['recall']:.4f} (Macro Avg Recall)
{results['classification_report']['macro avg']['f1-score']:.4f} (Macro Avg F1-Score)

PER-CLASS ACCURACY:
"""
    
    for class_name, accuracy in results['class_accuracy'].items():
        report += f"- {class_name}: {accuracy:.4f}\\n"
    
    report += f"\\n{'='*60}\\n"
    
    return report