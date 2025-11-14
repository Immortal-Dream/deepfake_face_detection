"""
Simplify CSV file format for deepfake detection dataset
Extracts only filename, label, and label_str columns
"""

import pandas as pd
from pathlib import Path
import argparse


def simplify_csv(input_csv_path, output_csv_path=None):
    """
    Simplify one CSV file by keeping only filename, label, and label_str columns.

    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file (default: overwrites input)
    """
    # Read the original CSV
    df = pd.read_csv(input_csv_path)

    print(f"Original CSV shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    # Extract filename from path column
    df['filename'] = df['path'].apply(lambda x: Path(x).name)

    # Create simplified dataframe with only required columns
    simplified_df = df[['filename', 'label', 'label_str']].copy()

    # Use input path as output if not specified
    if output_csv_path is None:
        output_csv_path = input_csv_path

    # Save simplified CSV
    simplified_df.to_csv(output_csv_path, index=False)

    print(f"\nSimplified CSV shape: {simplified_df.shape}")
    print(f"Simplified columns: {simplified_df.columns.tolist()}")
    print(f"Output saved to: {output_csv_path}")

    # Show preview
    print("\nPreview of simplified CSV:")
    print(simplified_df.head(10))

    return simplified_df


def process_dataset_csvs(data_root):
    """
    Process both train.csv and valid.csv in the dataset.

    Args:
        data_root: Path to the data root directory (e.g., data/rvf10k)
    """
    data_path = Path(data_root)

    csv_files = ['train.csv', 'valid.csv']

    for csv_file in csv_files:
        csv_path = data_path / csv_file

        if csv_path.exists():
            print(f"\n{'=' * 60}")
            print(f"Processing: {csv_file}")
            print('=' * 60)
            simplify_csv(csv_path)
        else:
            print(f"Warning: {csv_path} not found, skipping...")


def main():
    parser = argparse.ArgumentParser(
        description='Simplify deepfake detection CSV files'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (optional, defaults to overwriting input)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/rvf10k',
        help='Data root directory to process all CSVs (default: data/rvf10k)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all CSV files in data root directory'
    )

    args = parser.parse_args()

    if args.all:
        # Process all CSVs in the data root
        process_dataset_csvs(args.data_root)
    elif args.input:
        # Process single CSV file
        simplify_csv(args.input, args.output)
    else:
        # Default: process all CSVs in data root
        print("No specific file provided, processing all CSVs in data root...")
        process_dataset_csvs(args.data_root)


