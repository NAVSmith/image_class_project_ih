"""
Training script for image classification models.

This script allows you to train either a custom CNN or transfer learning model
from the command line.
"""

import argparse
import os
import sys
import pickle
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import (
    load_cifar10, normalize_images, prepare_labels, 
    create_train_val_split, create_data_augmentation, resize_for_transfer_learning
)
from model_architectures import (
    create_custom_cnn, create_transfer_learning_model, 
    compile_model, create_callbacks
)
from training_utils import train_model, evaluate_model, plot_training_history, save_results


def main():
    parser = argparse.ArgumentParser(description='Train image classification models')
    parser.add_argument('--model-type', choices=['custom', 'transfer'], required=True,
                       help='Type of model to train')
    parser.add_argument('--dataset', choices=['cifar10', 'animals10'], default='cifar10',
                       help='Dataset to use')
    parser.add_argument('--base-model', choices=['vgg16', 'resnet50', 'efficientnet'],
                       help='Base model for transfer learning')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    parser.add_argument('--save-dir', default='../models',
                       help='Directory to save trained models')
    parser.add_argument('--augmentation', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Starting training with parameters:")
    print(f"  Model type: {args.model_type}")
    print(f"  Dataset: {args.dataset}")
    if args.model_type == 'transfer':
        print(f"  Base model: {args.base_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Validation split: {args.validation_split}")
    print(f"  Data augmentation: {args.augmentation}")
    
    # Load and preprocess data
    print("\\nLoading dataset...")
    if args.dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test), class_names = load_cifar10()
    else:
        raise NotImplementedError("Animals10 dataset not implemented yet")
    
    print(f"Dataset loaded: {len(x_train)} training, {len(x_test)} test samples")
    
    # Normalize images
    x_train, x_test = normalize_images(x_train, x_test)
    
    # Prepare labels
    num_classes = len(class_names)
    y_train_cat, y_test_cat = prepare_labels(y_train, y_test, num_classes)
    
    # Create train/validation split
    x_train_split, x_val_split, y_train_split, y_val_split = create_train_val_split(
        x_train, y_train_cat, val_size=args.validation_split
    )
    
    print(f"Data split: {len(x_train_split)} train, {len(x_val_split)} validation")
    
    # Resize for transfer learning if needed
    if args.model_type == 'transfer':
        print("Resizing images for transfer learning...")
        x_train_split = resize_for_transfer_learning(x_train_split)
        x_val_split = resize_for_transfer_learning(x_val_split)
        x_test = resize_for_transfer_learning(x_test)
        input_shape = x_train_split.shape[1:]
    else:
        input_shape = x_train.shape[1:]
    
    print(f"Input shape: {input_shape}")
    
    # Create model
    print("\\nCreating model...")
    if args.model_type == 'custom':
        model = create_custom_cnn(input_shape, num_classes)
        model_name = f"custom_cnn_{args.dataset}"
    else:
        if not args.base_model:
            raise ValueError("Base model must be specified for transfer learning")
        model, base_model = create_transfer_learning_model(
            args.base_model, input_shape, num_classes
        )
        model_name = f"{args.base_model}_transfer_{args.dataset}"
    
    # Compile model
    model = compile_model(model, learning_rate=args.learning_rate)
    
    print(f"Model created with {model.count_params():,} parameters")
    
    # Create callbacks
    model_save_path = os.path.join(args.save_dir, f"{model_name}_best.h5")
    callbacks = create_callbacks(
        model_save_path, 
        patience=args.early_stopping_patience
    )
    
    # Prepare training data
    if args.augmentation:
        print("Setting up data augmentation...")
        datagen = create_data_augmentation()
        datagen.fit(x_train_split)
        train_data = datagen.flow(x_train_split, y_train_split, batch_size=args.batch_size)
    else:
        train_data = (x_train_split, y_train_split)
    
    validation_data = (x_val_split, y_val_split)
    
    # Train model
    print(f"\\nStarting training for {args.epochs} epochs...")
    start_time = datetime.now()
    
    history = train_model(
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training completed in {training_time}")
    
    # Plot training history
    print("\\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate on test set
    print("\\nEvaluating on test set...")
    results = evaluate_model(model, x_test, y_test_cat, class_names)
    
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Test loss: {results['test_loss']:.4f}")
    
    # Save results
    print("\\nSaving results...")
    save_results(results, history, model_name, args.dataset, args.save_dir)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Save training configuration
    config = {
        'model_type': args.model_type,
        'dataset': args.dataset,
        'base_model': args.base_model,
        'epochs': len(history.history['accuracy']),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split,
        'augmentation': args.augmentation,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'class_names': class_names,
        'training_time': str(training_time),
        'final_test_accuracy': results['test_accuracy'],
        'final_test_loss': results['test_loss']
    }
    
    config_path = os.path.join(args.save_dir, f"{model_name}_config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"\\nTraining completed successfully!")
    print(f"Best model: {model_save_path}")
    print(f"Final model: {final_model_path}")
    print(f"Configuration: {config_path}")


if __name__ == '__main__':
    main()