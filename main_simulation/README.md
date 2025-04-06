# Main Simulation

This module contains the main simulation code for the causal penalization project.

## Structure

The code is organized into the following modules:

- `simulation/`: Contains functions for data generation and perturbation
  - `data_generation.py`: Functions for simulating Hawkes processes
  - `data_perturbation.py`: Functions for perturbing simulated data

- `estimation/`: Contains functions for parameter estimation
  - `likelihood.py`: Functions for computing likelihoods
  - `parameter_estimation.py`: Functions for estimating parameters

- `visualization/`: Contains functions for visualizing results
  - `visualization.py`: Functions for plotting events and comparing estimates

- `main.py`: Main script that ties everything together

## Usage

To run the simulation:

```bash
python main.py
```

This will:
1. Generate simulated data for multiple environments
2. Perturb the data using various methods
3. Estimate parameters using both non-penalized and penalized methods
4. Visualize the results

## Dependencies

- numpy
- matplotlib
- scipy
- tqdm 