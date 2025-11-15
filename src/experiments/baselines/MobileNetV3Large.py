#!/usr/bin/env python3
"""
Fake vs Real Face Classification - MobileNetV3Large Training Script
Extracted and refactored from Kaggle notebook source code
Adapted to project structure with path_config
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV3Large
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import csv

# Import project path configuration (assumes script is run from project root)
from src.config.path_config import *

# Suppress Python and TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Ensure output directories exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Model folder: {MODEL_FOLDER}")


def create_dataset_csv(data_dirs, output_csv):
    """
    Create CSV file containing image paths and labels from directories.
    Expected directory structure: data_dir/[class_name]/[images...]
    """
    print(f"Creating dataset CSV from directories: {data_dirs}")

    # Remove existing CSV if present
    output_csv = Path(output_csv)
    if output_csv.exists():
        output_csv.unlink()
        print(f"Removed existing {output_csv}")

    # Label mapping to normalize folder names
    class_mapping = {
        'fake': 'Fake',
        'real': 'Real',
        'Fake': 'Fake',
        'Real': 'Real',
    }

    # Write CSV file
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'label'])  # Header

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                print(f"Warning: Directory {data_dir} does not exist, skipping...")
                continue

            # Walk through the directory
            for root, dirs, files in os.walk(data_dir):
                root_path = Path(root)
                for file_name in files:
                    if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        image_path = root_path / file_name
                        # Get label from parent folder name (fake/real)
                        label = root_path.name
                        mapped_label = class_mapping.get(label, label)
                        writer.writerow([str(image_path), mapped_label])

    print(f"Dataset CSV created: {output_csv}")
    return output_csv


def load_and_preprocess_data(csv_path, height=224, width=224):
    """
    Load images from CSV and preprocess for training.
    Returns: X (images), y (one-hot labels), label_dict (name->id mapping)
    """
    print(f"Loading images from {csv_path}...")
    data = pd.read_csv(csv_path)

    # Create label mapping
    label_names = data['label'].unique()
    label_dict = {name: idx for idx, name in enumerate(label_names)}
    print(f"Classes found: {label_dict}")

    # Convert labels to categorical
    y = data['label'].map(label_dict).values
    y = to_categorical(y, num_classes=len(label_dict))

    # Load and preprocess images
    X = np.empty((data.shape[0], height, width, 3), dtype=np.float32)
    for i, img_path in enumerate(data['image'].values):
        if i % 100 == 0:
            print(f"Loading image {i + 1}/{data.shape[0]}")
        img = load_img(img_path, target_size=(height, width))
        img_array = img_to_array(img)  # Converts to float32 and scales to [0, 1]
        X[i] = img_array

    print(f"Data loaded: X.shape={X.shape}, y.shape={y.shape}")
    return X, y, label_dict


def create_mobilenet_model(num_classes, input_shape=(224, 224, 3)):
    """
    Create MobileNetV3Large model with custom classification head.
    Architecture matches the original source code exactly.
    """
    print("Creating MobileNetV3Large model...")
    base_model = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Custom layers (learn the features generated from previous CNN layers)
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.45)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Compile with Adam optimizer (exactly as in original)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def train_baseline_model(X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
    """
    Train baseline model without data augmentation.
    """
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODEL (No Augmentation)")
    print("=" * 60)

    model = create_mobilenet_model(y_train.shape[1])

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    return model, history


def train_augmented_model(X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
    """
    Train model with data augmentation and early stopping.
    Matches the exact augmentation parameters from original code.
    """
    print("\n" + "=" * 60)
    print("TRAINING MODEL WITH DATA AUGMENTATION")
    print("=" * 60)

    # Data augmentation (exact parameters from original)
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        vertical_flip=True
    )
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

    model = create_mobilenet_model(y_train.shape[1])

    # Early stopping (exact parameters from original)
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    return model, history


def evaluate_and_plot(model, X_test, y_test, label_dict, experiment_name="experiment"):
    """
    Evaluate model, plot confusion matrix, and print metrics.
    """
    print(f"\nEvaluating {experiment_name}...")

    # Predictions
    y_pred = model.predict(X_test, verbose=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Metrics
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
    accuracy = accuracy_score(y_true_labels, y_pred_labels)

    print("\n" + "=" * 40)
    print(f"METRICS FOR {experiment_name.upper()}")
    print("=" * 40)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("=" * 40 + "\n")

    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='g',
        xticklabels=label_dict.keys(),
        yticklabels=label_dict.keys(),
        cmap='Blues'
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f'MobileNetV3Large_{experiment_name}_confusion_matrix.png', dpi=300)
    plt.show()

    return accuracy, f1


def plot_training_history(history, experiment_name="experiment"):
    """
    Plot training curves (accuracy and loss).
    """
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training', color='red', linestyle='-')
    plt.plot(history.history['val_accuracy'], label='Validation', color='blue', linestyle='-')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training', color='red', linestyle='-')
    plt.plot(history.history['val_loss'], label='Validation', color='blue', linestyle='-')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f'MobileNetV3Large_{experiment_name}_training_curves.png', dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Train MobileNetV3Large for Fake vs Real Face Classification'
    )
    parser.add_argument('--data_dirs', nargs='+',
                        default=[str(TRAIN_ROOT), str(VALID_ROOT)],
                        help='List of directories containing class subfolders')
    parser.add_argument('--csv_path',
                        default=str(OUTPUT_FOLDER / "MobileNetV3Large_dataset.csv"),
                        help='Path to dataset CSV file')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size (height, width)')
    parser.add_argument('--run_baseline', action='store_true', default=True,
                        help='Run baseline experiment')
    parser.add_argument('--run_augmented', action='store_true', default=True,
                        help='Run augmented experiment')
    parser.add_argument('--no_baseline', dest='run_baseline', action='store_false',
                        help='Skip baseline experiment')
    parser.add_argument('--no_augmented', dest='run_augmented', action='store_false',
                        help='Skip augmented experiment')
    parser.add_argument('--test_split', type=float, default=0.3,
                        help='Test set ratio')
    parser.add_argument('--val_split', type=float, default=0.5,
                        help='Validation ratio from test set')

    args = parser.parse_args()

    # Create or load dataset CSV
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        create_dataset_csv(args.data_dirs, csv_path)
    else:
        print(f"Using existing CSV file: {csv_path}")

    # Load data
    X, y, label_dict = load_and_preprocess_data(csv_path, args.image_size, args.image_size)

    # Split data (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=args.test_split, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=args.val_split, random_state=42
    )

    print(f"Train set: {X_train.shape}")
    print(f"Val set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Run baseline experiment
    if args.run_baseline:
        model_baseline, history_baseline = train_baseline_model(
            X_train, y_train, X_val, y_val, args.epochs, args.batch_size
        )

        plot_training_history(history_baseline, "baseline")
        acc_baseline, f1_baseline = evaluate_and_plot(
            model_baseline, X_test, y_test, label_dict, "baseline"
        )

        # Save model to MODEL_FOLDER with MobileNetV3Large naming
        baseline_model_path = MODEL_FOLDER / 'MobileNetV3Large_baseline_model.h5'
        model_baseline.save(baseline_model_path)
        print(f"Baseline model saved: {baseline_model_path}")

    # Run augmented experiment
    if args.run_augmented:
        model_augmented, history_augmented = train_augmented_model(
            X_train, y_train, X_val, y_val, args.epochs, args.batch_size
        )

        plot_training_history(history_augmented, "augmented")
        acc_augmented, f1_augmented = evaluate_and_plot(
            model_augmented, X_test, y_test, label_dict, "augmented"
        )

        # Save model to MODEL_FOLDER with MobileNetV3Large naming
        augmented_model_path = MODEL_FOLDER / 'MobileNetV3Large_augmented_model.h5'
        model_augmented.save(augmented_model_path)
        print(f"Augmented model saved: {augmented_model_path}")

    # Summary comparison
    if args.run_baseline and args.run_augmented:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Baseline Test Accuracy: {acc_baseline:.4f}")
        print(f"Augmented Test Accuracy: {acc_augmented:.4f}")
        print(f"Improvement: {acc_augmented - acc_baseline:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Run main
    main()