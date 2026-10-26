import pandas as pd
import numpy as np

def analyze_dataset_and_recommend(df: pd.DataFrame, target_col: str = None):
    """
    Analyzes the uploaded time-series dataset and returns structured insights
    and model recommendations for the Streamlit dashboard.
    """
    # Clean and extract numeric series for analysis
    series = pd.to_numeric(df[target_col], errors='coerce').dropna()
    total_samples = len(series)
    
    # Calculate a simple variability score (Coefficient of Variation)
    mean_val = series.mean()
    std_val = series.std()
    variability_score = round(float(std_val / mean_val), 3) if mean_val != 0 else 0.0

    # Dataset summary dictionary
    dataset_summary = {
        "total_samples": total_samples,
        "variability_score": variability_score
    }

    # Recommendations based on data length and behavior
    recommendations = [
        {
            "model": "PatchTST",
            "reason": "Highly recommended for long-horizon forecasting with large datasets due to channel-independence and transformer patching."
        },
        {
            "model": "Informer",
            "reason": "Efficient for capturing long-range dependencies with ProbSparse self-attention, reducing memory overhead."
        },
        {
            "model": "Autoformer",
            "reason": "Great for datasets with strong seasonal patterns and trend decompositions using series-connection architecture."
        }
    ]

    # Return strictly as a dictionary
    return {
        "dataset_summary": dataset_summary,
        "recommendations": recommendations
    }