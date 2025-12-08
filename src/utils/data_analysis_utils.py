import numpy as np

from src.config.path_config import *
import pandas as pd
from scipy import stats
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


def _get_valid_data(experiment_name, dataset_name):
    """
    Internal helper function: read and clean data,
    return valid samples and their total count.
    """
    csv_path = OUTPUT_FOLDER / experiment_name / dataset_name / 'layercam_analysis.csv'

    if not csv_path.exists():
        print(f"[Error] CSV not found: {csv_path}")
        return None, 0

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Reading CSV {csv_path}: {e}")
        return None, 0

    # Data cleaning: remove samples where face detection failed
    if 'is_failed' in df.columns:
        valid_df = df[df['is_failed'] == 0]
    else:
        valid_df = df  # backward compatibility if 'is_failed' column does not exist

    return valid_df, len(valid_df)


def compare_models_difference(experiment_name1, experiment_name2, dataset_name):
    """
    Compare the dependency of two models on facial regions
    within the same dataset.

    Metrics calculated:
        - Absolute difference (Delta)
        - Relative change rate
        - Odds Ratio (OR)
        - Z-test p-value

    Args:
        experiment_name1 (str): Model A (baseline/control)
        experiment_name2 (str): Model B (comparison target)
        dataset_name (str): Dataset name
    """

    print(f"\n{'=' * 20} Comparing Models: {experiment_name1} vs {experiment_name2} {'=' * 20}")

    # 1. Load data
    df_A, N_A = _get_valid_data(experiment_name1, dataset_name)
    df_B, N_B = _get_valid_data(experiment_name2, dataset_name)

    if df_A is None or df_B is None or N_A == 0 or N_B == 0:
        print("[Error] Invalid data source or empty data. Comparison aborted.")
        return

    print(f"Sample Size Model A ({experiment_name1}): {N_A}")
    print(f"Sample Size Model B ({experiment_name2}): {N_B}")

    # 2. Define facial regions to analyze
    region_columns = [
        'attention_jaw', 'attention_eyebrows', 'attention_nose',
        'attention_eyes', 'attention_mouth', 'attention_forehead'
    ]

    # Ensure columns exist in both datasets
    valid_cols = [c for c in region_columns if c in df_A.columns and c in df_B.columns]

    comparison_results = []

    # 3. Statistical comparison for each region
    for col in valid_cols:
        region_name = col.replace('attention_', '').capitalize()

        # Activation counts
        C_A = df_A[col].sum()
        C_B = df_B[col].sum()

        # Proportions
        p_A = C_A / N_A
        p_B = C_B / N_B

        # --- 3.1 Descriptive comparison ---

        # 1) Absolute difference (Delta)
        delta = p_A - p_B

        # 2) Relative change rate
        if p_B == 0:
            rel_change = np.inf if p_A > 0 else 0
        else:
            rel_change = (p_A - p_B) / p_B

        # 3) Odds Ratio (OR)
        epsilon = 1e-9  # small constant to avoid division by zero
        odds_A = (p_A + epsilon) / (1 - p_A + epsilon)
        odds_B = (p_B + epsilon) / (1 - p_B + epsilon)
        odds_ratio = odds_A / odds_B

        # --- 3.2 Hypothesis testing (two-sample proportion Z-test) ---

        # Pooled proportion
        p_hat = (C_A + C_B) / (N_A + N_B)

        # Standard error
        se_term = p_hat * (1 - p_hat) * ((1 / N_A) + (1 / N_B))
        se = np.sqrt(se_term)

        # Z-score and p-value
        if se == 0:
            z_score = 0.0
            p_value = 1.0
        else:
            z_score = (p_A - p_B) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # two-tailed test

        # Collect results
        comparison_results.append({
            "Region": region_name,
            f"Rate_A ({experiment_name1})": round(p_A, 4),
            f"Rate_B ({experiment_name2})": round(p_B, 4),
            "Abs_Diff (Delta)": round(delta, 4),
            "Rel_Change (%)": round(rel_change * 100, 2),
            "Odds_Ratio": round(odds_ratio, 4),
            "Z_Score": round(z_score, 4),
            "P_Value": p_value,
            "Significant": "*" if p_value < 0.05 else ""  # significance marker
        })

    # 5. Convert results to DataFrame
    res_df = pd.DataFrame(comparison_results)

    # Print formatted table
    print("\n[Comparison Result] (Significant difference marked with *)")
    print("-" * 120)
    header = f"{'Region':<12} | {'Rate A':<8} | {'Rate B':<8} | {'Diff':<8} | {'RelChg%':<8} | {'OR':<6} | {'Z-score':<8} | {'P-Value':<8}"
    print(header)
    print("-" * 120)

    for _, row in res_df.iterrows():
        p_val_str = "< 0.001" if row['P_Value'] < 0.001 else f"{row['P_Value']:.4f}"
        sig_mark = row['Significant']

        line = (f"{row['Region']:<12} | {row[f'Rate_A ({experiment_name1})']:<8} | "
                f"{row[f'Rate_B ({experiment_name2})']:<8} | {row['Abs_Diff (Delta)']:<8} | "
                f"{row['Rel_Change (%)']:<8} | {row['Odds_Ratio']:<6} | "
                f"{row['Z_Score']:<8} | {p_val_str:<8} {sig_mark}")
        print(line)
    print("-" * 120)

    # 6. Save results to CSV
    output_dir = OUTPUT_FOLDER / "model_comparison" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"compare_{experiment_name1}_vs_{experiment_name2}.csv"
    output_path = output_dir / output_filename

    res_df.to_csv(output_path, index=False)
    print(f"\nComparison detailed report saved to: {output_path}")

