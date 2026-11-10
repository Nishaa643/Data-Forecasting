import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """Sliding windows returning multivariate inputs and targets."""

    def __init__(self, values, starts, lookback_len, pred_len):
        self.values = values
        self.starts = list(starts)
        self.lookback_len = lookback_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        middle = start + self.lookback_len
        end = middle + self.pred_len
        x = self.values[start:middle]
        y = self.values[middle:end]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def prepare_data(
    df,
    target_col,
    lookback_len,
    pred_len,
    time_col=None,
    batch_size=32,
    train_ratio=0.70,
    val_ratio=0.15,
):
    """Chronological 70/15/15 split with train-only scaler fitting."""
    if target_col not in df.columns:
        raise ValueError("Selected target column does not exist.")
    if lookback_len < 1 or pred_len < 1:
        raise ValueError("Lookback and prediction horizon must be positive.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid train/validation split ratios.")

    data_df = df.copy()
    if time_col is not None:
        if time_col not in data_df.columns:
            raise ValueError("Selected time column does not exist.")
        parsed = pd.to_datetime(data_df[time_col], errors="coerce")
        if parsed.isna().any():
            raise ValueError("The selected time column contains invalid dates.")
        data_df = data_df.assign(_sort_time=parsed).sort_values("_sort_time").drop(columns="_sort_time")

    feature_cols = [
        c for c in data_df.select_dtypes(include=[np.number]).columns
        if c != time_col
    ]
    if target_col not in feature_cols:
        raise ValueError("Target column must contain numerical values.")
    if not feature_cols:
        raise ValueError("No numerical features found in the dataset.")

    numeric = data_df[feature_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    n = len(numeric)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= lookback_len + pred_len:
        raise ValueError("Not enough training rows. Reduce lookback/prediction horizon or use more data.")
    if val_end - train_end < pred_len or n - val_end < pred_len:
        raise ValueError("Validation or test partition is too small for the selected prediction horizon.")

    # Impute using training information only. Validation/test missing values use
    # training medians, avoiding backward interpolation from future observations.
    train_frame = numeric.iloc[:train_end].copy()
    train_frame = train_frame.interpolate(limit_direction="both")
    medians = train_frame.median().fillna(0.0)
    train_frame = train_frame.fillna(medians)
    val_frame = numeric.iloc[train_end:val_end].fillna(medians)
    test_frame = numeric.iloc[val_end:].fillna(medians)
    clean = pd.concat([train_frame, val_frame, test_frame], ignore_index=True)

    if clean.isna().any().any():
        raise ValueError("Could not impute all numerical missing values.")

    values = clean.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(values[:train_end])
    scaled_values = scaler.transform(values).astype(np.float32)

    # A window's target segment must lie wholly inside its assigned partition.
    train_starts = range(0, train_end - lookback_len - pred_len + 1)
    val_starts = range(max(0, train_end - lookback_len), val_end - lookback_len - pred_len + 1)
    test_starts = range(max(0, val_end - lookback_len), n - lookback_len - pred_len + 1)

    train_starts = list(train_starts)
    val_starts = list(val_starts)
    test_starts = list(test_starts)
    if not train_starts or not val_starts or not test_starts:
        raise ValueError("Not enough windows in one of the data partitions. Reduce lookback/prediction horizon.")

    train_ds = TimeSeriesDataset(scaled_values, train_starts, lookback_len, pred_len)
    val_ds = TimeSeriesDataset(scaled_values, val_starts, lookback_len, pred_len)
    test_ds = TimeSeriesDataset(scaled_values, test_starts, lookback_len, pred_len)

    loaders = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )
    target_idx = feature_cols.index(target_col)
    split_info = {"train_end": train_end, "val_end": val_end, "total_rows": n}
    return (*loaders, scaled_values, scaler, feature_cols, target_idx, split_info)
