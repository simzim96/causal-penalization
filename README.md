# Causal Penalization

A Python project implementing causal inference with penalization methods for estimating invariant parameters across different environments.

## Overview

This project explores the use of penalization methods in causal inference, particularly for identifying invariant causal relationships across different environments or domains. The core idea is based on Bühlmann's invariance principle, which suggests that causal mechanisms remain stable across different environments while non-causal associations may vary.

The implementation focuses on two main applications:

1. **Linear Causal Models**: Estimating invariant linear coefficients across environments.
2. **Marked Spatio-Temporal Hawkes Processes**: Estimating invariant parameters in point process models across environments.

## Theoretical Background

The causal penalized estimator is designed to estimate parameters that remain invariant across different environments. For each environment e, we estimate environment-specific parameters θ^e by minimizing:

```
L(θ^e) + λ * ||θ^e - θ||^2
```

where:
- L(θ^e) is the loss function (e.g., negative log-likelihood) for environment e
- θ is the global parameter that is invariant across environments
- λ is the penalization strength controlling how strongly we enforce invariance

This approach is inspired by the invariant causal prediction framework (Peters, Bühlmann, and Meinshausen, 2016), which posits that causal mechanisms are invariant across environments, while non-causal associations may change.

## Installation

This project uses Poetry for dependency management. To install the dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/causal-penalization.git
cd causal-penalization

# Install dependencies using Poetry
poetry install
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

- `estimator/`: Core implementation of causal penalization methods
  - `estimator.py`: Linear causal model estimator
  - `hawkes/`: Hawkes process estimators
- `simulate/`: Data generation modules
- `main_simulation/`: Main simulation workflows
- `performance_simulation/`: Performance benchmarking
- `plot_beta_convergence/`: Visualization tools for convergence analysis
- `penalty_term_simulation/`: Simulations for different penalty term strengths

## Usage

### Running Simulations

To run the main simulations comparing penalized and non-penalized estimators:

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

### Using the Estimator for Custom Data

To use the causal penalized estimator with your own data:

```python
from estimator import causal_penalized_estimator_iterative

# Prepare your data from different environments
X_envs = [X_env1, X_env2, ...]  # List of predictor matrices
Y_envs = [y_env1, y_env2, ...]  # List of response vectors

# Run the estimator
lambda_reg = 1.0  # Regularization strength
beta, beta_envs = causal_penalized_estimator_iterative(X_envs, Y_envs, lambda_reg)

# beta now contains the estimated invariant coefficients
print("Invariant coefficients:", beta)
```

For Hawkes process modeling:

```python
from estimator.hawkes.hawk_estimator import estimate_penalized

# Prepare your point process data from different environments
events_list = [events_env1, events_env2, ...]

# Run the penalized estimator
kappa = 10.0  # Penalization strength
initial_guess = [0.001, 0.4, 1.0, 1.0, 0.0, 0.0]  # Initial parameter guess
theta_list, theta_global = estimate_penalized(events_list, T, area, kappa, initial_guess)

# theta_global now contains the estimated invariant parameters
print("Invariant parameters:", theta_global)
```

## Examples and Experiments

The project includes several simulation scenarios:

1. **No Perturbation**: All environments follow the same data generation process
2. **Covariate Shift**: Some environments have shifted covariate values
3. **Covariate Removal**: Some environments have removed covariates
4. **Time Offset**: Some environments have temporal offsets
5. **Combined Perturbations**: Multiple types of environmental heterogeneity

These experiments demonstrate how the penalized estimator can recover invariant parameters even when environments differ substantially.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```
Zimmermann, S. (2023). Causal Penalization: A Framework for Identifying Invariant Parameters Across Environments.
``` 