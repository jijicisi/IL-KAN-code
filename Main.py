import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from IL_KAN_model import RealKANModel
import time
import jobpy  # For saving scalers

# --- Configuration ---
CONFIG = {
    "SKEWED_PARAMS": ['CS', 'LAG', 'X', 'WM', 'MP', 'CG', 'KG'],
    "MODEL_PATH": "RESULT/modeltest/tunxi/final_IL_KAN.ckpt",
    "SCALER_PATH": "RESULT/modeltest/tunxi/scalers.pt"
}


def preprocess_hydrology_data(df):
    """Clean and transform features."""
    if 'Total rainfall' in df.columns and 'total flood volume' in df.columns:
        df['Q_P_ratio'] = df['total flood volume'] / (df['Total rainfall'] + 1e-6)
    log_targets = ['Measured peak flow', 'Total rainfall', 'total flood volume', 'Area rainfall of rainstorm center']
    for col in log_targets:
        if col in df.columns: df[f'log_{col}'] = np.log1p(df[col])
    exclude = ['Interval name', 'Measured peak time', 'Measured flood rise time', 'Measured flood recession time', 'Time of rainstorm center']
    return df.select_dtypes(include=[np.number]).drop(columns=exclude, errors='ignore')


class HydrologyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.x)

    def __getitem__(self, idx): return self.x[idx], self.y[idx]


def train_and_save():
    """Incremental Learning process."""
    feat_df = preprocess_hydrology_data(pd.read_csv('RESULT/FEATURE/tunxi/Feature-all-fd.csv'))
    param_df = pd.read_csv('RESULT/Parameter Data/tunxi/Parameter-all-fd.csv')
    param_df.columns = CONFIG["ALL_PARAMS"]

    for p in CONFIG["SKEWED_PARAMS"]: param_df[p] = np.log1p(param_df[p])

    x_scaler, y_scaler = RobustScaler(), RobustScaler()
    x_norm = x_scaler.fit_transform(feat_df.values)
    y_norm = y_scaler.fit_transform(param_df.values)

    skewed_indices = [CONFIG["ALL_PARAMS"].index(p) for p in CONFIG["SKEWED_PARAMS"]]
    model = RealKANModel(x_norm.shape[1], y_norm.shape[1], skewed_indices,
                         {'width': [256, 256, 256], 'grid': 30, 'k': 4, 'lr': 1e-5})

    i, window_size, smoothed_window = 0, 10, 10
    last_val_loss = None

    while i + int(smoothed_window) < len(x_norm):
        curr_w = int(smoothed_window)
        train_loader = DataLoader(HydrologyDataset(x_norm[i:i + curr_w], y_norm[i:i + curr_w]), batch_size=32)
        val_loader = DataLoader(HydrologyDataset(x_norm[i + curr_w:i + curr_w + 5], y_norm[i + curr_w:i + curr_w + 5]),
                                batch_size=32)

        trainer = pl.Trainer(max_epochs=5, accelerator="auto", enable_checkpointing=False, logger=False)
        trainer.fit(model, train_loader)

        val_metrics = trainer.validate(model, val_loader, verbose=False)[0]
        curr_loss = val_metrics["val_focused_weighted_MSE"]

        if last_val_loss is not None:
            delta_L = (curr_loss - last_val_loss) / (last_val_loss + 1e-6)
            window_size = np.clip(window_size * (1 - CONFIG["LAMBDA"] * delta_L), CONFIG["WINDOW_MIN"],
                                  CONFIG["WINDOW_MAX"])
            smoothed_window = CONFIG["ALPHA"] * smoothed_window + (1 - CONFIG["ALPHA"]) * window_size
        last_val_loss, i = curr_loss, i + 1

    # Save Model and Scalers
    torch.save(model.state_dict(), CONFIG["MODEL_PATH"])
    torch.save({'x_scaler': x_scaler, 'y_scaler': y_scaler}, CONFIG["SCALER_PATH"])
    print("Training finished and model saved.")
    return model, x_scaler, y_scaler


def load_and_predict():
    """Load existing model and run inference."""
    # 1. Load Scalers
    scalers = torch.load(CONFIG["SCALER_PATH"])
    x_scaler, y_scaler = scalers['x_scaler'], scalers['y_scaler']

    # 2. Re-initialize and load model weights
    # Note: Input size must match original training data (e.g., 18 or 20 features)
    dummy_input_size = x_scaler.n_features_in_
    skewed_indices = [CONFIG["ALL_PARAMS"].index(p) for p in CONFIG["SKEWED_PARAMS"]]

    model = RealKANModel(dummy_input_size, len(CONFIG["ALL_PARAMS"]), skewed_indices,
                         {'width': [256, 256, 256], 'grid': 30, 'k': 4, 'lr': 1e-5})
    model.load_state_dict(torch.load(CONFIG["MODEL_PATH"]))
    model.eval()

    # 3. Predict
    test_df = pd.read_csv('RESULT/FEATURE/tunxi/2016-3-Feature.csv')
    test_proc = preprocess_hydrology_data(test_df)
    x_test = torch.tensor(x_scaler.transform(test_proc.values), dtype=torch.float32)

    with torch.no_grad():
        y_pred_norm, _ = model(x_test)
        y_pred = y_scaler.inverse_transform(y_pred_norm.numpy())

    # 4. Reverse log and Save
    for p in CONFIG["SKEWED_PARAMS"]:
        idx = CONFIG["ALL_PARAMS"].index(p)
        y_pred[:, idx] = np.expm1(y_pred[:, idx])

    res_df = pd.DataFrame(y_pred, columns=CONFIG["ALL_PARAMS"])
    res_df.to_csv("RESULT/modeltest/tunxi/Final_Predictions.csv", index=False)
    print("Predictions saved successfully.")


if __name__ == "__main__":
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        train_and_save()
    load_and_predict()



