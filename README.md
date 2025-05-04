# Causal Penalization

A Python project implementing causal inference with penalization methods for estimating invariant parameters across different environments.

## Table of Contents
- [Overview](#overview)
- [Theoretical Background](#theoretical-background)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Simulation Framework](#simulation-framework)
- [Perturbation Methods](#perturbation-methods)
- [Usage Examples](#usage-examples)
- [Experimental Results](#experimental-results)
- [Parameter Tuning](#parameter-tuning)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

This project explores the use of penalization methods in causal inference, particularly for identifying invariant causal relationships across different environments or domains. The implementation is based on Bühlmann's invariance principle, which posits that causal mechanisms remain stable across environments while non-causal associations may vary.

The project focuses on two main applications:

1. **Linear Causal Models**: Estimating invariant linear coefficients across environments.
2. **Marked Spatio-Temporal Hawkes Processes**: Estimating invariant parameters in point process models across environments.

These methods are particularly valuable for domains with heterogeneous data sources or when dealing with distribution shifts in causal modeling.

## Theoretical Background

### The Invariance Principle

The core idea behind this implementation is the invariance principle in causal inference: causal relationships remain invariant across different environments, while non-causal relationships may change across environments. This principle provides a powerful approach for distinguishing causal from non-causal relationships when data from multiple environments is available.

### Causal Penalized Estimator

For each environment e, we estimate environment-specific parameters θ^e by minimizing:

```
L(θ^e) + λ * ||θ^e - θ||^2
```

where:
- L(θ^e) is the loss function (e.g., negative log-likelihood) for environment e
- θ is the global parameter that is invariant across environments
- λ is the penalization strength controlling how strongly we enforce invariance

This approach iteratively:
1. Updates environment-specific parameters given the current global parameter
2. Updates the global parameter as a robust estimate (e.g., median) of the environment-specific parameters

### Mathematical Formulation

#### Linear Models

For linear models, the iterative algorithm alternates between:

1. Computing environment-specific parameters:
   ```
   β^e = (X^e'X^e + λI)^(-1)(X^e'Y^e + λβ)
   ```

2. Computing the global parameter:
   ```
   β = median({β^e})
   ```

where X^e and Y^e are the predictors and response variables for environment e.

#### Hawkes Processes

For marked spatio-temporal Hawkes processes, we minimize:

```
-logL(θ^e) + 0.5 * κ * ||θ^e - θ||^2
```

where logL is the log-likelihood function for the Hawkes process with parameters:
- μ: Background intensity
- α: Branching ratio
- β: Temporal decay rate
- σ: Spatial standard deviation
- γ: Covariate effect parameters

## Installation

This project uses Poetry for dependency management:

```bash
# Clone the repository
git clone https://github.com/yourusername/causal-penalization.git
cd causal-penalization

# Install dependencies using Poetry
poetry install

# Alternatively, if you don't use Poetry
pip install -r requirements.txt  # Note: you may need to create this from pyproject.toml
```

### Dependencies

- Python 3.11+
- NumPy
- pandas
- scikit-learn
- matplotlib
- tqdm
- openpyxl

## Project Structure

```
causal-penalization/
├── estimator/                    # Core implementation of estimators
│   ├── __init__.py
│   ├── estimator.py              # Linear causal model estimator
│   └── hawkes/                   # Hawkes process estimators
│       └── hawk_estimator.py
├── simulate/                     # Data generation modules
│   └── simulate_marked_temporal.py
├── main_simulation/              # Main simulation workflows
│   ├── __init__.py
│   ├── main.py                   # Main simulation runner
│   ├── README.md
│   ├── plots/                    # Generated visualizations
│   ├── visualization/            # Visualization utilities
│   ├── estimation/               # Estimation wrappers
│   └── simulation/               # Simulation utilities
│       ├── __init__.py
│       ├── data_generation.py
│       └── data_perturbation.py
├── performance_simulation/       # Performance benchmarking
├── plot_beta_convergence/        # Visualization for convergence analysis
├── penalty_term_simulation/      # Simulations for different penalty strengths
├── pyproject.toml                # Project configuration and dependencies
├── poetry.lock                   # Locked dependencies
└── README.md                     # This file
```

## Core Components

### Linear Causal Model Estimator

The linear causal model estimator (`estimator.py`) implements the penalized regression approach for linear models. Key features:

- Iterative algorithm alternating between environment-specific and global parameter updates
- Robust to outlier environments using median-based global updates
- Convergence tracking for parameter history analysis
- Tolerance-based stopping criteria for optimization

```python
from estimator import causal_penalized_estimator_iterative

# X_envs: list of predictor matrices for each environment
# Y_envs: list of response vectors for each environment
# lambda_reg: regularization strength
beta, beta_envs = causal_penalized_estimator_iterative(X_envs, Y_envs, lambda_reg)
```

### Hawkes Process Estimator

The Hawkes process estimator (`hawk_estimator.py`) extends the penalization approach to marked spatio-temporal point processes:

- Maximum likelihood estimation with penalty term
- Supports marked processes with covariate effects
- L-BFGS-B optimization with bounds for parameter constraints
- Both penalized and non-penalized estimation for comparison

## Simulation Framework

### Data Generation

The simulation framework generates synthetic data that follows the assumed model:

1. **Linear Models**: Regression data with environment-specific perturbations
2. **Hawkes Processes**: Self-exciting point processes with spatial and temporal components

### Perturbation Methods

To test the robustness of the estimators, the framework applies various perturbations to create environment heterogeneity:

1. **Covariate Shift**: Adding constant shifts to covariate values in selected environments
   ```python
   events_copy[:, 3:] += shift_value * np.random.randn()
   ```

2. **Covariate Removal**: Setting certain covariates to zero in selected environments
   ```python
   events_copy[:, 3 + idx] = 0
   ```

3. **Time Offset**: Adding time delays to events in selected environments
   ```python
   events_copy[:, 0] += time_offset
   ```

4. **Combined Perturbations**: Applying multiple types of perturbations simultaneously

## Usage Examples

### Running Main Simulations

```bash
# Navigate to the main simulation directory
cd main_simulation

# Run the main simulation
python main.py
```

This will:
1. Generate data for multiple environments
2. Apply various perturbations to create environment shifts
3. Estimate parameters using both penalized and non-penalized approaches
4. Compare results and generate visualizations

### Linear Model Example

```python
from estimator import causal_penalized_estimator_iterative
import numpy as np

# Example setup with 3 environments
n_env = 3
n_obs = 500
p = 2
true_beta = np.array([1.0, -1.0])

# Generate synthetic data for multiple environments
X_envs = [np.random.randn(n_obs, p) for _ in range(n_env)]
Y_envs = [
    X @ true_beta + np.random.normal(0, 1, size=n_obs)
    for X in X_envs
]

# Apply penalized estimation
lambda_reg = 10.0
beta_hat, beta_envs = causal_penalized_estimator_iterative(X_envs, Y_envs, lambda_reg)

print("True beta:", true_beta)
print("Estimated beta:", beta_hat)
```

### Hawkes Process Example

```python
from estimator.hawkes.hawk_estimator import estimate_penalized
from simulate.simulate_marked_temporal import generate_data

# Simulation parameters
mu_true = 0.0005
alpha_true = 0.5
beta_true = 0.8
sigma_true = 0.5
true_gamma = np.array([0.8, 0.8, 0.8])
T = 100.0
region = (0, 10, 0, 10)

# Generate data for multiple environments
num_env = 5
events_list = generate_data(
    num_env, mu_true, alpha_true, beta_true, sigma_true, 
    T, region, num_cov=3, true_gamma=true_gamma
)

# Run penalized estimation
kappa = 10.0
initial_guess = np.concatenate([np.array([0.001, 0.4, 1.0, 1.0]), np.zeros(3)])
theta_list, theta_global = estimate_penalized(
    events_list, T, region[1] * region[3], kappa, initial_guess
)

print("True parameters:", np.concatenate([np.array([mu_true, alpha_true, beta_true, sigma_true]), true_gamma]))
print("Estimated parameters:", theta_global)
```

## Experimental Results

The simulations test the performance of the estimators under different scenarios:

### No Perturbation Scenario

When all environments follow the same data generation process, both penalized and non-penalized estimators perform well. However, the penalized estimator typically shows reduced variance in estimates.

### Covariate Shift Scenario

When environments experience covariate shifts, the penalized estimator demonstrates robustness by identifying the invariant components, while the non-penalized estimator may be significantly biased.

### Covariate Removal Scenario

When some environments have removed covariates, the penalized estimator can effectively recover the true parameters using information from other environments, while the non-penalized estimator struggles.

### Combined Perturbations

Under multiple types of perturbations, the advantage of the penalized estimator becomes most pronounced, demonstrating its ability to identify invariant parameters despite diverse environmental heterogeneity.

## Parameter Tuning

The penalization strength (λ or κ) is a critical hyperparameter that controls the trade-off between fitting individual environments and enforcing parameter invariance:

- **Low values**: Allow environment-specific parameters to vary significantly
- **High values**: Force environment-specific parameters to be very close to the global parameter

Optimal values depend on:
1. The degree of heterogeneity between environments
2. The sample size per environment
3. The true causal strength of parameters

The `penalty_term_simulation` module provides tools to analyze the impact of different penalty values on estimation performance.

## Performance Considerations

### Computational Complexity

- The iterative algorithm's computational cost scales linearly with the number of environments
- For Hawkes processes, likelihood computation can be expensive with large event counts
- Optimization convergence may require more iterations with complex perturbations

### Statistical Efficiency

- Sample efficiency improves with the number of environments
- The penalized estimator shows better finite-sample properties than methods that treat environments independently
- Performance degradation is graceful with increasing environment heterogeneity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Guidelines for contributing:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Citation

If you use this code in your research, please cite:

```
Zimmermann, S. Causal Regularization for Marked Spatial Point Processes (2025).
``` 