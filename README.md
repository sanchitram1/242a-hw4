# HW 4: Bootstrap Analysis of Random Forest vs OLS

**Course:** IND ENG 242A (UC Berkeley)

## Overview

This assignment compares the performance of Random Forest and Ordinary Least Squares (OLS) regression models using bootstrap resampling on out-of-sample R² (OSR²) estimates.

## Problem 1: Bootstrap Comparison of Random Forest vs OLS

### Objective

Evaluate and compare the bias, variance, and confidence intervals of out-of-sample R² estimates for two regression models:
- **OLS**: Linear regression
- **RF**: Random Forest regressor

### Methodology

1. **Data Generation**: Synthetic data following the model:
   - X ~ N(0, 1)
   - ε ~ N(0, 1)
   - y = 1 + 2.5X + ε

2. **Bootstrap Analysis**: 
   - Generate B=2000 bootstrap samples from the test set
   - Compute OSR² on each bootstrap sample
   - Calculate bias, variance, and 95% confidence intervals

3. **Comparisons**:
   - Empirical OSR² on original test set
   - Bootstrap statistics for each model
   - Difference (OLS - RF) with 95% CI

### Key Results

#### Part (a): N=300
- **OLS OSR²**: 86.52% (empirical)
  - Bias: -0.0068
  - Variance: 0.0011
  - 95% CI: (0.7865, 0.9143)

- **RF OSR²**: 80.38% (empirical)
  - Bias: -0.0116
  - Variance: 0.0026
  - 95% CI: (0.6809, 0.8736)

- **Difference (OLS - RF)**: 
  - Mean: 0.0662
  - 95% CI: (0.0049, 0.1441)
  - **Interpretation**: OLS performs significantly better than RF (95% confidence that difference is non-zero and positive)

#### Part (b): N=2000
Results available in notebook output.

## Code Structure

The notebook is organized with modularized functions:

### Data Generation
- `generate_data(N, rng)`: Creates synthetic data
- `train_test_split_data(x, y, test_size, seed)`: Splits into train/test

### Bootstrapping
- `create_test_bootstraps(X_test, y_test, B, seed)`: Generates B bootstrap samples

### Model Fitting
- `fit_ols_model(X_train, y_train)`: Fits OLS model
- `fit_rf_model(X_train, y_train, random_state)`: Fits Random Forest model

### Evaluation
- `compute_empirical_osr2(model, X_test, y_test)`: Out-of-sample R² on original test set
- `compute_bootstrapped_osr2(model, X_test_bootstraps, y_test_bootstraps)`: OSR² on bootstrap samples
- `compute_bias_variance_ci(bootstrapped_r2, empirical_r2)`: Calculates bias, variance, and CI
- `compute_difference_ci(ols_r2, rf_r2)`: Compares two models

### Reporting & Visualization
- `report_empirical_scores()`: Prints empirical R² scores
- `report_all_statistics()`: Prints bootstrap statistics
- `report_difference_ci()`: Prints difference statistics
- `plot_bootstrap_distributions()`: Histograms of OSR² distributions
- `plot_difference_distribution()`: Histogram of difference

### Orchestration
- `run_analysis(N, B, seed, verbose)`: Runs complete analysis pipeline

## Installation

Requires Python 3.8+ with dependencies specified in `pyproject.toml`.

### Using uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. uv is a fast Python package manager written in Rust.

**Install uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Setup the project environment:**

```bash
uv sync
```

This will create a virtual environment and install all dependencies from the lockfile (`uv.lock`), ensuring consistent dependency versions across all environments.

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

Or call the analysis function directly:
```python
# Run analysis with sample size N=300
results = run_analysis(N=300, B=2000)

# Run analysis with sample size N=2000
results = run_analysis(N=2000, B=2000)
```

## Key Insights

- Bootstrap resampling allows estimation of estimator variability without theoretical assumptions
- Random seed management is critical when using `sklearn.utils.resample` to ensure independent samples
- OLS shows lower variance (more stable) compared to RF on this synthetic regression problem
- The 95% confidence interval for the difference excludes 0, providing strong evidence that OLS outperforms RF for this particular dataset

## Files

- `main.ipynb`: Complete analysis notebook
- `main.html`: HTML export of notebook
- `README.md`: This file

