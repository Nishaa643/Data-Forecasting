# Time Series Forecasting Using Transformer Models

A Streamlit-based time-series forecasting application for exploring
datasets, training Transformer-based forecasting models, visualizing
predictions, and comparing model performance.

## Overview

This project brings five Transformer architectures into one forecasting
workflow:

-   **FEDformer**
-   **PatchTST**
-   **Informer**
-   **ETSformer**
-   **Crossformer**

Users can upload a time-series dataset in CSV format, select a timestamp
column and target variable, configure the input lookback window and
prediction horizon, and run a selected model. The application also
provides dataset analysis and model-evaluation visualizations.

> **Note:** Model performance depends on the dataset, preprocessing,
> train/validation/test split, target variable, and hyperparameters.
> Results should be treated as experimental results, not as a guarantee
> that one architecture will always outperform the others.

## Features

-   CSV dataset upload through a Streamlit interface.
-   Selection of an optional date/time column and a numerical target
    variable.
-   Configurable lookback window and prediction horizon.
-   Five Transformer forecasting architectures in a common application.
-   Dataset analysis, including basic statistical information and
    data-quality indicators.
-   Forecast visualization alongside historical observations.
-   Model evaluation using:
    -   Mean Absolute Error (MAE)
    -   Root Mean Squared Error (RMSE)
    -   Mean Absolute Percentage Error (MAPE)
    -   Coefficient of Determination (R²)
-   Comparison against a naive forecasting baseline.
-   Training and validation loss information, where available from the
    training run.
-   Model comparison and recommendation views.

## Project Structure

``` text
Data_Forecasting/
│
├── app.py
├── extract_models.py
├── LONG_TERM_SERIES_FORECASTING_ETT.ipynb
├── requirements.txt
│
└── backend/
    ├── __init__.py
    ├── analyzer.py
    ├── data.py
    ├── models.py
    └── trainer.py
```

### Main files

  ------------------------------------------------------------------------------
  File                                       Purpose
  ------------------------------------------ -----------------------------------
  `app.py`                                   Streamlit user interface and
                                             dashboard

  `backend/analyzer.py`                      Dataset analysis and model
                                             recommendations

  `backend/data.py`                          Data preparation and creation of
                                             forecasting sequences

  `backend/models.py`                        Implementations of the five
                                             Transformer architectures

  `backend/trainer.py`                       Model initialization, training,
                                             forecasting, and evaluation

  `extract_models.py`                        Utility for extracting model
                                             definitions from the research
                                             notebook

  `LONG_TERM_SERIES_FORECASTING_ETT.ipynb`   Research and experimentation
                                             notebook

  `requirements.txt`                         Python dependencies
  ------------------------------------------------------------------------------

## Requirements

-   Python 3.10 or a compatible Python version supported by the
    dependencies.
-   pip
-   A supported operating system (Windows, Linux, or macOS).
-   CPU for model execution; a compatible CUDA-enabled GPU can be used
    if the installed PyTorch build and code environment support it.

## Installation

1.  Clone the repository:

    ``` bash
    git clone <YOUR_REPOSITORY_URL>
    cd Data_Forecasting
    ```

    Replace `<YOUR_REPOSITORY_URL>` with the URL of your Git repository.

2.  (Recommended) Create and activate a virtual environment:

    **Windows PowerShell**

    ``` powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

    **Linux/macOS**

    ``` bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the dependencies:

    ``` bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Run the application

From the project root directory, execute:

``` bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open
that address in a browser.

## Using the application

1.  Start the application.
2.  Upload a time-series dataset in `.csv` format.
3.  Inspect the uploaded data and verify that the columns were read
    correctly.
4.  Select the date/time column if the dataset contains one.
5.  Select a numerical target variable to forecast.
6.  Choose one of the available Transformer architectures.
7.  Set the lookback window (number of past observations supplied to the
    model).
8.  Set the prediction horizon (number of future observations to
    predict).
9.  Run the forecasting workflow.
10. Review the forecast, evaluation metrics, baseline comparison, and
    dataset/model analysis.

### Dataset expectations

The application is designed for tabular time-series data:

-   Each row should represent an observation at a point in time.
-   The dataset should contain at least one numerical column to use as
    the target.
-   If a timestamp column is available, select it as the time column.
-   Keep observations in chronological order. Check for missing
    timestamps, duplicate rows, missing values, and irregular sampling
    intervals before interpreting results.
-   Use a sufficiently long dataset. Very small datasets may not provide
    enough training and test windows for meaningful Transformer
    comparisons.

For a multivariate dataset, numerical columns can provide additional
information to the forecasting model, depending on the data-preparation
configuration.

## Models

### 1. Informer
Uses ProbSparse attention to reduce the computational cost of attention for long input sequences.

### 2. FEDformer
Uses frequency-enhanced representations and decomposition to model trend and seasonal patterns.

### 3. PatchTST
Divides time series into patches and applies Transformer-based modelling to the patch representations.

### 4. ETSformer
Incorporates exponential-smoothing-inspired components into a Transformer architecture.

### 5. Crossformer
Uses cross-dimension and segment-level modelling for multivariate time series.
  -----------------------------------------------------------------------

## Evaluation metrics

The application reports forecasting performance using the following
metrics:

-   **MAE:** Average absolute difference between actual and predicted
    values. Lower is better.
-   **RMSE:** Square root of the mean squared prediction error. Lower is
    better and large errors receive a stronger penalty.
-   **MAPE:** Average absolute percentage error. Lower is better, but it
    can be unreliable when actual target values are zero or close to
    zero.
-   **R²:** Coefficient of determination. Values closer to 1 generally
    indicate a better fit; a negative value means the predictions
    perform worse than predicting the test-set mean.

Metrics should be calculated on the original target scale after inverse
transformation. When comparing architectures, use the same chronological
data split, target, input window, prediction horizon, and evaluation
procedure for every model.

## Dataset suggestions

The application can be used with different time-series datasets,
provided they are prepared in a compatible CSV format. Examples include:

-   **Weather:** Weather measurements such as temperature, humidity,
    pressure, and wind variables.
-   **Electricity:** Electricity consumption or load measurements.
-   **Solar Energy:** Solar power generation measurements.
-   **Traffic:** Traffic speed or flow measurements.
-   **Exchange Rate:** Historical currency exchange-rate observations.

Choose the timestamp and target columns according to the dataset's
schema. For example, for the commonly used Weather dataset, `date` is
the timestamp column and `T (degC)` can be selected as the temperature
target.

## Important experimental considerations

-   Use chronological splitting rather than randomly shuffling
    time-series observations.
-   Fit preprocessing scalers on the training data only, then apply the
    fitted transformations to validation and test data.
-   Avoid using future information when constructing input windows or
    features.
-   Compare all models on the same test period and with the same
    forecasting horizon.
-   Check whether reported metrics are calculated over overlapping
    forecast windows or unique timestamps; the evaluation method affects
    how results should be interpreted.
-   MAPE may be unsuitable for targets that can be zero or negative.
-   A naive baseline is useful for checking whether a more complex model
    provides an actual improvement.
-   Record model settings, dataset details, and random seeds when
    conducting repeatable experiments.

## Troubleshooting

### `ModuleNotFoundError` or missing dependency

Activate the intended Python environment and install the dependencies
again:

``` bash
python -m pip install -r requirements.txt
```

### Import errors from the backend

Run Streamlit from the project root directory (the directory containing
`app.py` and the `backend` folder). Confirm that `backend/__init__.py`
exists.

### No numerical target variable appears

Check that the target column contains numeric values and that it was not
imported as text. Clean invalid values and reload the CSV.

### Poor forecasting metrics

Check the timestamp order, missing values, target selection, scaling,
lookback window, prediction horizon, and train/test split. Compare the
model with the naive baseline before drawing conclusions.

## Future improvements

Potential extensions include:

-   Support for additional dataset formats.
-   More configurable model hyperparameters.
-   Unique-timestamp evaluation in addition to window-level evaluation.
-   Prediction interval calibration and uncertainty analysis.
-   Experiment logging and downloadable prediction results.
-   Reproducible multi-dataset benchmarking.

## Acknowledgements

This project explores Transformer architectures and ideas developed in
the time-series forecasting research community. Refer to the original
papers and official implementations when using or extending individual
model architectures.

## License

Add the license applicable to this repository before distributing or
reusing the project.
