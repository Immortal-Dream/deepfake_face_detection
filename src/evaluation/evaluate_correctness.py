"""
Evaluation functions for deepfake detection model
Provides metrics calculation and single prediction verification
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.config.path_config import VALID_CSV, OUTPUT_FOLDER


def evaluate_predictions(predict_csv_filename, ground_truth_csv=None):
    """
    Evaluate prediction results against ground truth.

    Args:
        predict_csv_filename: Name of the prediction CSV file (e.g., "model_output.csv")
        ground_truth_csv: Path to ground truth CSV file (default: VALID_CSV)

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Set default ground truth path
    if ground_truth_csv is None:
        ground_truth_csv = VALID_CSV

    # Load prediction and ground truth CSV files
    pred_path = OUTPUT_FOLDER / predict_csv_filename

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    if not Path(ground_truth_csv).exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_csv}")

    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(ground_truth_csv)

    # Merge on filename to align predictions with ground truth
    merged_df = pd.merge(
        gt_df,
        pred_df,
        on='filename',
        suffixes=('_gt', '_pred')
    )

    if len(merged_df) == 0:
        raise ValueError("No matching filenames found between prediction and ground truth!")

    # Extract labels
    y_true = merged_df['label_gt'].values
    y_pred = merged_df['label_pred'].values

    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate class-specific accuracies
    # Real (label=1) accuracy
    real_mask = y_true == 1
    real_correct = (y_true[real_mask] == y_pred[real_mask]).sum()
    real_total = real_mask.sum()
    real_accuracy = real_correct / real_total if real_total > 0 else 0

    # Fake (label=0) accuracy
    fake_mask = y_true == 0
    fake_correct = (y_true[fake_mask] == y_pred[fake_mask]).sum()
    fake_total = fake_mask.sum()
    fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0

    # Compile results
    results = {
        'total_samples': len(merged_df),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'real_samples': int(real_total),
        'fake_samples': int(fake_total),
        'real_correct': int(real_correct),
        'fake_correct': int(fake_correct)
    }

    # Print detailed results
    print("=" * 70)
    print(f"EVALUATION RESULTS: {predict_csv_filename}")
    print("=" * 70)
    print(f"Total Samples: {results['total_samples']}")
    print(f"  - Real samples: {results['real_samples']}")
    print(f"  - Fake samples: {results['fake_samples']}")
    print()
    print("Overall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f} ({results['precision'] * 100:.2f}%)")
    print(f"  Recall:    {results['recall']:.4f} ({results['recall'] * 100:.2f}%)")
    print(f"  F1 Score:  {results['f1_score']:.4f} ({results['f1_score'] * 100:.2f}%)")
    print()
    print("Class-Specific Accuracy:")
    print(
        f"  Real (label=1): {results['real_accuracy']:.4f} ({results['real_accuracy'] * 100:.2f}%) - {results['real_correct']}/{results['real_samples']}")
    print(
        f"  Fake (label=0): {results['fake_accuracy']:.4f} ({results['fake_accuracy'] * 100:.2f}%) - {results['fake_correct']}/{results['fake_samples']}")
    print()
    print("Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                Fake (0)  Real (1)")
    print(f"  Actual Fake     {tn:6d}    {fp:6d}")
    print(f"  Actual Real     {fn:6d}    {tp:6d}")
    print("=" * 70)

    return results


def check_single_prediction(filename, predicted_label, ground_truth_csv=None):
    """
    Check if a single prediction is correct.

    Args:
        filename: Image filename (e.g., "24731.jpg")
        predicted_label: Predicted label (0 for fake, 1 for real)
        ground_truth_csv: Path to ground truth CSV file (default: VALID_CSV)

    Returns:
        bool: True if prediction is correct, False otherwise
    """
    # Set default ground truth path
    if ground_truth_csv is None:
        ground_truth_csv = VALID_CSV

    if not Path(ground_truth_csv).exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_csv}")

    # Load ground truth
    gt_df = pd.read_csv(ground_truth_csv)

    # Find the row with matching filename
    match = gt_df[gt_df['filename'] == filename]

    if len(match) == 0:
        raise ValueError(f"Filename '{filename}' not found in ground truth CSV")

    if len(match) > 1:
        raise ValueError(f"Multiple entries found for filename '{filename}'")

    # Get true label
    true_label = match.iloc[0]['label']

    # Compare prediction with ground truth
    is_correct = (predicted_label == true_label)

    # Print result
    true_label_str = match.iloc[0]['label_str']
    pred_label_str = "real" if predicted_label == 1 else "fake"

    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    print(
        f"{status} | File: {filename} | True: {true_label_str} ({true_label}) | Predicted: {pred_label_str} ({predicted_label})")

    return is_correct


def batch_check_predictions(predictions, ground_truth_csv=None):
    """
    Check multiple predictions at once.

    Args:
        predictions: List of tuples (filename, predicted_label)
        ground_truth_csv: Path to ground truth CSV file (default: VALID_CSV)

    Returns:
        dict: Summary of batch prediction results
    """
    results = []

    print("\nBatch Prediction Check")
    print("-" * 70)

    for filename, predicted_label in predictions:
        is_correct = check_single_prediction(filename, predicted_label, ground_truth_csv)
        results.append(is_correct)

    # Summary
    correct_count = sum(results)
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print("-" * 70)
    print(f"Summary: {correct_count}/{total_count} correct ({accuracy * 100:.2f}%)")

    return {
        'total': total_count,
        'correct': correct_count,
        'wrong': total_count - correct_count,
        'accuracy': accuracy
    }

