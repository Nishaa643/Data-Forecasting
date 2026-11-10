import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.models import FEDformer, PatchTST, Informer, ETSformer, Crossformer
from backend.data import prepare_data

MODEL_NAMES = ["FEDformer", "PatchTST", "Informer", "ETSformer", "Crossformer"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(model_type, lookback_len, pred_len, num_features):
    label_len = min(48, lookback_len)
    if model_type == "FEDformer":
        return FEDformer(seq_len=lookback_len, label_len=label_len, pred_len=pred_len,
                         enc_in=num_features, dec_in=num_features, c_out=num_features,
                         d_model=64, modes=32)
    if model_type == "PatchTST":
        patch_len = min(16, lookback_len)
        stride = min(8, patch_len)
        return PatchTST(seq_len=lookback_len, pred_len=pred_len, num_features=num_features,
                        patch_len=patch_len, stride=stride, d_model=64,
                        n_heads=4, n_layers=2)
    if model_type == "Informer":
        return Informer(seq_len=lookback_len, label_len=label_len, pred_len=pred_len,
                        enc_in=num_features, dec_in=num_features, c_out=num_features,
                        d_model=64, n_heads=4)
    if model_type == "ETSformer":
        return ETSformer(seq_len=lookback_len, pred_len=pred_len, enc_in=num_features,
                         c_out=num_features, d_model=64, K=8, e_layers=2)
    if model_type == "Crossformer":
        return Crossformer(seq_len=lookback_len, pred_len=pred_len, in_dim=num_features,
                           seg_len=16, d_model=64, n_heads=4, depth=2)
    raise ValueError(f"Unknown model: {model_type}")


def forward_model(model, model_type, x, pred_len):
    if model_type in ("FEDformer", "Informer"):
        label_len = min(48, x.shape[1])
        decoder_zeros = x.new_zeros(x.shape[0], pred_len, x.shape[2])
        x_dec = torch.cat([x[:, -label_len:, :], decoder_zeros], dim=1)
        output = model(x, x_dec)
    else:
        output = model(x)
    if output.ndim != 3 or output.shape[1] != pred_len:
        raise RuntimeError(f"{model_type} returned shape {tuple(output.shape)}; expected [batch, {pred_len}, features].")
    return output


def _inverse_target(values_scaled, scaler, target_idx):
    return values_scaled * float(scaler.scale_[target_idx]) + float(scaler.mean_[target_idx])


def _calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nonzero = np.abs(actual) > 1e-8
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def _train_evaluate(
    df, target_col, model_type, lookback_len, pred_len,
    time_col=None, epochs=3, batch_size=32, learning_rate=0.001, seed=42
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (train_loader, val_loader, test_loader, scaled_values, scaler,
     feature_cols, target_idx, split_info) = prepare_data(
        df, target_col, lookback_len, pred_len, time_col=time_col, batch_size=batch_size
    )

    model = create_model(model_type, lookback_len, pred_len, len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_state = None
    training_losses, validation_losses = [], []
    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_model(model, model_type, x, pred_len)
            if prediction.shape != y.shape:
                raise RuntimeError(f"{model_type}: prediction shape {tuple(prediction.shape)} does not match target {tuple(y.shape)}")
            loss = criterion(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_total += loss.item()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_total += criterion(forward_model(model, model_type, x, pred_len), y).item()
        avg_train = train_total / max(1, len(train_loader))
        avg_val = val_total / max(1, len(val_loader))
        training_losses.append(float(avg_train))
        validation_losses.append(float(avg_val))
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())

    training_time = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on held-out test windows. Each window predicts a horizon;
    # flattening evaluates all forecasted points (overlapping windows included).
    actual_scaled, predicted_scaled, naive_scaled = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x_device = x.to(device)
            pred = forward_model(model, model_type, x_device, pred_len).cpu().numpy()
            actual_scaled.append(y.numpy())
            predicted_scaled.append(pred)
            last_target = x[:, -1, target_idx].numpy()[:, None]
            naive_scaled.append(np.repeat(last_target, pred_len, axis=1))

    actual_scaled = np.concatenate(actual_scaled, axis=0)[:, :, target_idx]
    predicted_scaled = np.concatenate(predicted_scaled, axis=0)[:, :, target_idx]
    naive_scaled = np.concatenate(naive_scaled, axis=0)
    actual_test = _inverse_target(actual_scaled, scaler, target_idx).reshape(-1)
    predicted_test = _inverse_target(predicted_scaled, scaler, target_idx).reshape(-1)
    naive_test = _inverse_target(naive_scaled, scaler, target_idx).reshape(-1)
    metrics = _calculate_metrics(actual_test, predicted_test)
    naive_metrics = _calculate_metrics(actual_test, naive_test)

    # Forecast the next horizon using the latest observed lookback window.
    latest = torch.tensor(scaled_values[-lookback_len:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        future_scaled = forward_model(model, model_type, latest, pred_len).squeeze(0).cpu().numpy()
    future_forecast = _inverse_target(future_scaled[:, target_idx], scaler, target_idx)
    history = _inverse_target(scaled_values[-lookback_len:, target_idx], scaler, target_idx)
    future_naive = np.repeat(float(history[-1]), pred_len)

    # Empirical symmetric error band based on the 90th percentile absolute test residual.
    residual_radius = float(np.quantile(np.abs(actual_test - predicted_test), 0.90)) if len(actual_test) else 0.0
    lower = future_forecast - residual_radius
    upper = future_forecast + residual_radius

    return {
        "model": model_type,
        "history": history.tolist(),
        "future_forecast": future_forecast.tolist(),
        "future_naive": future_naive.tolist(),
        "future_lower": lower.tolist(),
        "future_upper": upper.tolist(),
        "actual_test": actual_test.tolist(),
        "predicted_test": predicted_test.tolist(),
        "naive_test": naive_test.tolist(),
        "metrics": metrics,
        "naive_metrics": naive_metrics,
        "training_loss": training_losses,
        "validation_loss": validation_losses,
        "training_time": float(training_time),
        "device": str(device),
        "split_info": split_info,
        "feature_cols": feature_cols,
        "target_idx": target_idx,
    }


def run_transformer_forecast(
    df, target_col, model_type, lookback_len, pred_len,
    time_col=None, epochs=3, batch_size=32, learning_rate=0.001, seed=42
):
    return _train_evaluate(df, target_col, model_type, lookback_len, pred_len,
                           time_col, epochs, batch_size, learning_rate, seed)


def compare_all_models(
    df, target_col, lookback_len, pred_len, time_col=None,
    epochs=3, batch_size=32, learning_rate=0.001, seed=42
):
    rows = []
    detailed = {}
    for model_name in MODEL_NAMES:
        result = _train_evaluate(df, target_col, model_name, lookback_len, pred_len,
                                 time_col, epochs, batch_size, learning_rate, seed)
        detailed[model_name] = result
        rows.append({
            "Model": model_name,
            **result["metrics"],
            "Training Time (s)": result["training_time"],
            "Naive RMSE": result["naive_metrics"]["RMSE"],
        })
    rows.sort(key=lambda row: row["RMSE"])
    return rows, detailed
