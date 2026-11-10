import numpy as np
import pandas as pd


def analyze_dataset_and_recommend(df, target_col):
    if target_col not in df.columns:
        raise ValueError("Target column does not exist.")

    series = pd.to_numeric(df[target_col], errors="coerce")
    clean = series.dropna()
    mean_val = float(clean.mean()) if len(clean) else 0.0
    std_val = float(clean.std()) if len(clean) > 1 else 0.0
    variability = round(std_val / abs(mean_val), 3) if abs(mean_val) > 1e-12 else float("inf")

    numeric = df.select_dtypes(include=[np.number])
    missing = df.isna().sum().astype(int).to_dict()
    duplicate_rows = int(df.duplicated().sum())
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = float(q3 - q1) if len(clean) else 0.0
    outlier_count = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()) if len(clean) else 0

    trend = "Insufficient data"
    if len(clean) >= 3:
        slope = np.polyfit(np.arange(len(clean)), clean.to_numpy(), 1)[0]
        trend = "Increasing" if slope > 1e-8 else "Decreasing" if slope < -1e-8 else "Approximately stable"

    recommendations = []
    if len(clean) >= 1000:
        recommendations.append({"model": "Consider Informer", "reason": f"The target contains {len(clean):,} valid observations, so a long-sequence architecture may be worth benchmarking."})
    else:
        recommendations.append({"model": "Consider PatchTST", "reason": f"The target contains {len(clean):,} valid observations. Patch-based modelling is a candidate, but should be judged on held-out performance."})
    if np.isfinite(variability) and variability > 0.5:
        recommendations.append({"model": "Consider FEDformer", "reason": f"The coefficient of variation is {variability}; frequency/decomposition-based modelling may be worth testing."})
    else:
        recommendations.append({"model": "Consider ETSformer", "reason": "The measured relative variability is not high; compare exponential-smoothing and other architectures on the same test split."})
    if outlier_count:
        recommendations.append({"model": "Inspect outliers", "reason": f"The IQR rule flagged {outlier_count} target observations. Inspect them before deciding whether to remove or transform them."})
    if any(missing.values()):
        recommendations.append({"model": "Handle missing values", "reason": "Missing values were found. Review their pattern and use a time-aware imputation strategy before training."})

    return {
        "dataset_summary": {
            "total_samples": int(len(clean)),
            "mean": mean_val,
            "median": float(clean.median()) if len(clean) else 0.0,
            "std": std_val,
            "variability_score": variability,
            "missing_values": missing,
            "duplicate_rows": duplicate_rows,
            "outlier_count": outlier_count,
            "trend": trend,
            "numeric_columns": list(numeric.columns),
        },
        "recommendations": recommendations,
    }
