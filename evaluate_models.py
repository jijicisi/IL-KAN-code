import pandas as pd
import numpy as np
import os


# --- 1. Metric Definitions ---
def calc_metrics(obs, sim):
    """Calculate all hydrological performance indicators."""
    # Nash-Sutcliffe Efficiency
    nse = 1 - np.sum((sim - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((sim - obs) ** 2))

    # Mean Absolute Error
    mae = np.mean(np.abs(sim - obs))

    # Kling-Gupta Efficiency
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-8)
    beta = np.mean(sim) / (np.mean(obs) + 1e-8)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    # Percent Peak Error
    peak_obs, peak_sim = np.max(obs), np.max(sim)
    pe = ((peak_sim - peak_obs) / peak_obs * 100) if peak_obs != 0 else np.nan

    # Time to Peak Error (Index difference)
    te = np.argmax(sim) - np.argmax(obs)

    return {
        'NSE': round(nse, 4),
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'KGE': round(kge, 4),
        'PE(%)': round(pe, 2),
        'TE': te
    }

# --- 2. Configuration ---
# Set paths to English-standard naming for GitHub
DATA_CONFIG = {
    "input_file": 'RESULT/2016-3-zl.xlsx',
    "output_file": 'RESULT/Model_Evaluation_Metrics.xlsx',
    "obs_column": 'tunxi flow',
}

def main():
    if not os.path.exists(DATA_CONFIG["input_file"]):
        print(f"Error: File {DATA_CONFIG['input_file']} not found.")
        return

    # Load Data
    df = pd.read_excel(DATA_CONFIG["input_file"])
    obs_col = DATA_CONFIG["obs_column"]

    # Filter columns that exist in the dataframe
    available_models = [m for m in DATA_CONFIG["target_models"] if m in df.columns]
    evaluation_results = []
    obs_data = df[obs_col].values

    for model_name in available_models:
        sim_data = df[model_name].values

        # Data validation
        if np.isnan(sim_data).any() or np.isnan(obs_data).any():
            print(f"Warning: NaNs detected in {model_name}. Skipping.")
            continue

        # Calculate metrics
        metrics = calc_metrics(obs_data, sim_data)
        metrics['Model_Name'] = model_name
        evaluation_results.append(metrics)

    # Convert to DataFrame and reorganize columns
    results_df = pd.DataFrame(evaluation_results)

    # Reorder columns to put Model_Name first
    cols = ['Model_Name'] + [c for c in results_df.columns if c != 'Model_Name']
    results_df = results_df[cols]

    # Display and Save
    print("\nEvaluation Results Summary:")
    print(results_df.to_string(index=False))

    os.makedirs(os.path.dirname(DATA_CONFIG["output_file"]), exist_ok=True)
    results_df.to_excel(DATA_CONFIG["output_file"], index=False)

if __name__ == "__main__":
    main()