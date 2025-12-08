from src.config.path_config import *
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.config.path_config import *


def model_sensitivity_analysis(experiment_name: str, dataset_name: str):
    """
    Analyze the model's dependency on specific facial regions for a given dataset (Sensitivity Analysis).
    This calculates how often specific regions (eyes, nose, etc.) are activated in the CAM heatmaps.
    """
    # 1. Define paths
    input_csv_path = OUTPUT_FOLDER / experiment_name / dataset_name / 'layercam_analysis.csv'
    output_dir = OUTPUT_FOLDER / experiment_name / dataset_name / 'data_analysis'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load data
    if not input_csv_path.exists():
        print(f"[Error] CSV file not found: {input_csv_path}")
        return

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return

    print(f"\n{'=' * 20} Sensitivity Analysis: {experiment_name} on {dataset_name} {'=' * 20}")
    print(f"Total rows loaded: {len(df)}")

    # 3. Data Cleaning: Filter out images where face detection failed (is_failed == 1)
    # Rationale: If no face is detected, all 'attention_*' columns are 0 by default.
    # Keeping them would artificially lower the activation rates (expanding the denominator).
    if 'is_failed' in df.columns:
        valid_df = df[df['is_failed'] == 0]
        failed_count = len(df) - len(valid_df)
        print(f"Skipping {failed_count} images where face detection failed.")
    else:
        valid_df = df
        print("Warning: 'is_failed' column not found. Using all rows.")

    if len(valid_df) == 0:
        print("[Error] No valid data points after filtering.")
        return

    total_valid_images = len(valid_df)
    print(f"Valid images for analysis: {total_valid_images}")

    # 4. Define columns to analyze
    region_columns = [
        'attention_jaw',
        'attention_eyebrows',
        'attention_nose',
        'attention_eyes',
        'attention_mouth',
        'attention_forehead'
    ]

    # Check which columns actually exist in the CSV
    existing_cols = [col for col in region_columns if col in valid_df.columns]

    # 5. Statistical Analysis
    stats_data = []
    for col in existing_cols:
        # Sum the column to get the count of activated images (values are 0 or 1)
        activated_count = valid_df[col].sum()
        # Calculate percentage based on valid images
        percentage = (activated_count / total_valid_images) * 100

        # Clean up name for display (e.g., "attention_nose" -> "Nose")
        region_name = col.replace('attention_', '').capitalize()

        stats_data.append({
            'Region': region_name,
            'Activated_Count': int(activated_count),
            'Total_Count': total_valid_images,
            'Dependency_Rate (%)': round(percentage, 2)
        })

    # Create DataFrame and Sort
    result_df = pd.DataFrame(stats_data)
    # Sort by dependency rate in descending order (High dependency -> Low dependency)
    result_df = result_df.sort_values(by='Dependency_Rate (%)', ascending=False).reset_index(drop=True)

    # 6. Print Ranking Results
    print(f"\n[{experiment_name}] [{dataset_name}] Facial Region Dependency (High to Low):")
    print("-" * 60)
    print(f"{'Rank':<5} | {'Region':<12} | {'Rate (%)':<10} | {'Count'}")
    print("-" * 60)
    for idx, row in result_df.iterrows():
        print(
            f"{idx + 1:<5} | {row['Region']:<12} | {row['Dependency_Rate (%)']:<10} | {row['Activated_Count']}/{row['Total_Count']}")
    print("-" * 60)

    # 7. Save Analysis to CSV
    output_csv_path = output_dir / 'sensitivity_ranking.csv'
    result_df.to_csv(output_csv_path, index=False)
    print(f"\nAnalysis saved to: {output_csv_path}")

    # 8. (Optional) Generate and Save Bar Chart
    try:
        plt.figure(figsize=(10, 6))
        # Use a color palette for better visualization
        sns.barplot(x='Region', y='Dependency_Rate (%)', data=result_df, palette='viridis')

        plt.title(f'Model Attention Sensitivity by Facial Region\n({experiment_name} - {dataset_name})')
        plt.ylabel('Activation Rate (%)')
        plt.ylim(0, 100)  # Percentage scale

        # Annotate bars with specific percentage values
        for index, row in result_df.iterrows():
            plt.text(index, row['Dependency_Rate (%)'] + 1, f"{row['Dependency_Rate (%)']}%",
                     color='black', ha="center")

        plot_path = output_dir / 'sensitivity_chart.png'
        plt.savefig(plot_path)
        print(f"Chart saved to: {plot_path}")
        plt.close()  # Close plot to free memory
    except Exception as e:
        print(f"Skipped plotting due to error: {e}")


# Usage Example
if __name__ == "__main__":
    for dataset in DATASET_LIST:
        model_sensitivity_analysis(Xception_BSL_baseline, dataset)
