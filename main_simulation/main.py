import numpy as np
import os
import matplotlib.pyplot as plt

from simulation import generate_data, perturb_environments_flags
from estimation import get_bounds, estimate_non_penalized, estimate_penalized
from visualization import plot_spatial_events, plot_beta_comparison

def setup_parameters():
    """
    Set up simulation parameters for the marked spatio-temporal Hawkes process.
    
    This function defines all the parameters needed for simulating and estimating
    the marked spatio-temporal Hawkes process, including:
    
    1. True model parameters (μ, α, β, σ, γ)
    2. Simulation parameters (time horizon, spatial region)
    3. Estimation parameters (initial guess, bounds, penalty strength)
    
    Returns:
        dict: Dictionary containing all simulation parameters:
            - mu_true: Background intensity
            - alpha_true: Branching ratio
            - beta_true: Temporal decay rate
            - sigma_true: Spatial standard deviation
            - num_cov: Number of covariates
            - true_gamma: Covariate effect parameters
            - true_params: Combined parameter vector
            - T: End time of observation period
            - region: Spatial region as (xmin, xmax, ymin, ymax)
            - area: Area of the spatial region
            - initial_guess: Initial parameter values for optimization
            - bounds: Parameter bounds for optimization
            - kappa: Penalty strength for penalized estimation
    """
    # True simulation parameters
    mu_true = 0.0005        
    alpha_true = 0.5        
    beta_true = 1.5         
    sigma_true = 0.5        
    
    # Set true covariate effects
    num_cov = 3
    true_gamma = np.array([0.8] * num_cov)
    true_params = np.concatenate([np.array([mu_true, alpha_true, beta_true, sigma_true]), true_gamma])
    
    # Time and region parameters
    T = 100.0               
    region = (0, 10, 0, 10) 
    area = (region[1]-region[0]) * (region[3]-region[2])
    
    # Estimation parameters
    initial_guess = np.concatenate([np.array([0.001, 0.4, 1.0, 1.0]), np.zeros(num_cov)])
    bounds = get_bounds(num_cov)
    kappa = 10.0  # Penalty strength
    
    return {
        'mu_true': mu_true,
        'alpha_true': alpha_true,
        'beta_true': beta_true,
        'sigma_true': sigma_true,
        'num_cov': num_cov,
        'true_gamma': true_gamma,
        'true_params': true_params,
        'T': T,
        'region': region,
        'area': area,
        'initial_guess': initial_guess,
        'bounds': bounds,
        'kappa': kappa
    }

def run_simulation(params, scenario_name, perturb_cov_shift=False, perturb_cov_remove=False, 
                  perturb_time_offset=False, num_env=7, nsim=20):
    """
    Run a simulation experiment with specified perturbation scenario.
    
    This function conducts a complete simulation experiment, including:
    1. Generating data for multiple environments
    2. Applying specified perturbations to create environment shifts
    3. Performing non-penalized and penalized parameter estimation
    4. Comparing estimation results with true parameters
    5. Generating visualizations
    6. Running multiple simulations to assess parameter convergence
    
    Parameters:
        params (dict): Simulation parameters from setup_parameters()
        scenario_name (str): Name of the perturbation scenario for file naming
        perturb_cov_shift (bool, optional): Whether to apply covariate shifts to some
                                           environments. Defaults to False.
        perturb_cov_remove (bool, optional): Whether to remove covariates from some
                                            environments. Defaults to False.
        perturb_time_offset (bool, optional): Whether to apply time offsets to some
                                             environments. Defaults to False.
        num_env (int, optional): Number of environments to simulate. Defaults to 7.
        nsim (int, optional): Number of simulations for convergence analysis. Defaults to 20.
    
    Returns:
        dict: Results dictionary containing:
            - theta_global_np: Global non-penalized parameter estimates
            - theta_global_p: Global penalized parameter estimates
            - error_b1_np: Error in first covariate parameter (non-penalized)
            - error_b1_p: Error in first covariate parameter (penalized)
            - beta_np_list: List of β estimates across simulations (non-penalized)
            - beta_p_list: List of β estimates across simulations (penalized)
    
    Notes:
        - Creates visualization plots in the 'plots' directory
        - Prints detailed comparison of estimates with true parameters
    """
    print(f"\n==== Running simulation: {scenario_name} ====")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate data for multiple environments
    print(f"Generating data for {num_env} environments...")
    events_list = generate_data(
        num_env, 
        params['mu_true'], 
        params['alpha_true'], 
        params['beta_true'], 
        params['sigma_true'],
        params['T'], 
        params['region'], 
        num_cov=params['num_cov'], 
        true_gamma=params['true_gamma']
    )
    
    # Apply perturbations if specified
    remove_indices = [0]

    events_list = perturb_environments_flags(
        events_list, 
        perturb_fraction=0.5,
        perturb_cov_shift=perturb_cov_shift, 
        shift_value=0.5,
        perturb_cov_remove=perturb_cov_remove, 
        remove_indices=remove_indices,
        perturb_time_offset=perturb_time_offset, 
        time_offset=10.0
    )
    
    # Run non-penalized estimation
    print("\n==== Non-Penalized Estimation ====")
    theta_list_np, theta_global_np = estimate_non_penalized(
        events_list, 
        params['T'], 
        params['area'], 
        params['initial_guess'], 
        params['bounds']
    )
    
    # Run penalized estimation
    print("\n==== Penalized Estimation ====")
    theta_list_p, theta_global_p = estimate_penalized(
        events_list, 
        params['T'], 
        params['area'], 
        params['kappa'], 
        params['initial_guess'], 
        params['bounds']
    )
    
    # Compare estimates with true parameters
    print("\nTrue parameters:")
    print(params['true_params'])
    print("\nDifference (Non-Penalized Global - True):")
    print(theta_global_np - params['true_params'])
    print("\nDifference (Penalized Global - True):")
    print(theta_global_p - params['true_params'])
    
    # Measure the error for the first covariate effect (b1)
    error_b1_np = theta_global_np[4] - params['true_gamma'][0]
    error_b1_p = theta_global_p[4] - params['true_gamma'][0]
    print("\nError for b1 (Non-Penalized):", error_b1_np)
    print("Error for b1 (Penalized):", error_b1_p)
    
    # Visualize spatial locations for each environment
    for i in range(num_env):    
        spatial_plot_path = os.path.join(plots_dir, f"{scenario_name}_spatial_events_env{i}.png")
        plot_spatial_events(
            events_list[i], 
            title=f"Simulated Marked Spatio-Temporal Hawkes Process ({scenario_name})",
            save_path=spatial_plot_path
        )
        plt.close()  # Close the figure after saving
    
    # Run multiple simulations and compare the β estimates
    beta_np_list = []
    beta_p_list = []
    
    print(f"\nRunning {nsim} simulations for β estimation comparison...")
    for sim in range(nsim):
        events_list = generate_data(
            num_env, 
            params['mu_true'], 
            params['alpha_true'], 
            params['beta_true'], 
            params['sigma_true'],
            params['T'], 
            params['region'], 
            num_cov=params['num_cov'], 
            true_gamma=params['true_gamma']
        )
        
        # Apply perturbations if specified
        # select random indices to remove
        remove_indices = [0]
        events_list = perturb_environments_flags(
            events_list, 
            perturb_fraction=0.5,
            perturb_cov_shift=perturb_cov_shift, 
            shift_value=0.5,
            perturb_cov_remove=perturb_cov_remove, 
            remove_indices=remove_indices,
            perturb_time_offset=perturb_time_offset, 
            time_offset=10.0
        )
        
        _, theta_global_np = estimate_non_penalized(
            events_list, 
            params['T'], 
            params['area'], 
            params['initial_guess'], 
            params['bounds']
        )
        _, theta_global_p = estimate_penalized(
            events_list, 
            params['T'], 
            params['area'], 
            params['kappa'], 
            params['initial_guess'], 
            params['bounds']
        )
        beta_np_list.append(theta_global_np[2])
        beta_p_list.append(theta_global_p[2])
    
    # Save beta comparison plot
    beta_plot_path = os.path.join(plots_dir, f"{scenario_name}_beta_comparison.png")
    plot_beta_comparison(beta_np_list, beta_p_list, params['beta_true'], save_path=beta_plot_path)
    plt.close()  # Close the figure after saving
    
    return {
        'theta_global_np': theta_global_np,
        'theta_global_p': theta_global_p,
        'error_b1_np': error_b1_np,
        'error_b1_p': error_b1_p,
        'beta_np_list': beta_np_list,
        'beta_p_list': beta_p_list
    }

def main():
    """
    Main function to run the simulation experiments.
    
    This function:
    1. Sets up simulation parameters
    2. Defines different perturbation scenarios
    3. Runs simulations for each scenario
    4. Summarizes and prints results
    
    The following scenarios are evaluated:
    - No perturbation: All environments follow the same data generation process
    - Covariate shift: Some environments have shifted covariate values
    - Covariate removal: Some environments have removed covariates
    - All perturbations: Combination of covariate shifts, removals, and time offsets
    
    Each scenario helps evaluate the robustness of penalized vs. non-penalized estimation
    under different types of environmental heterogeneity.
    """
    # Set up parameters
    params = setup_parameters()
    
    # Run different simulation scenarios
    scenarios = [
        {
            'name': 'no_perturbation',
            'cov_shift': False,
            'cov_remove': False,
            'time_offset': False
        },
        {
            'name': 'cov_shift_only',
            'cov_shift': True,
            'cov_remove': False,
            'time_offset': False
        },
        {
            'name': 'cov_remove_only',
            'cov_shift': False,
            'cov_remove': True,
            'time_offset': False
        },
        {
            'name': 'all_perturbations',
            'cov_shift': True,
            'cov_remove': True,
            'time_offset': True
        }
    ]
    
    results = {}
    for scenario in scenarios:
        results[scenario['name']] = run_simulation(
            params,
            scenario['name'],
            perturb_cov_shift=scenario['cov_shift'],
            perturb_cov_remove=scenario['cov_remove'],
            perturb_time_offset=scenario['time_offset'],
            num_env=7,
            nsim=20
        )
    
    # Print summary of results
    print("\n==== SUMMARY OF RESULTS ====")
    for scenario_name, result in results.items():
        print(f"\n{scenario_name}:")
        print(f"  Error for b1 (Non-Penalized): {result['error_b1_np']:.4f}")
        print(f"  Error for b1 (Penalized): {result['error_b1_p']:.4f}")
        print(f"  Mean β (Non-Penalized): {np.mean(result['beta_np_list']):.4f}")
        print(f"  Mean β (Penalized): {np.mean(result['beta_p_list']):.4f}")
        print(f"  True β: {params['beta_true']:.4f}")

if __name__ == "__main__":
    main()
# %%