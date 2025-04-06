import numpy as np

from simulation import generate_data, perturb_environments_flags
from estimation import get_bounds, estimate_non_penalized, estimate_penalized
from visualization import plot_spatial_events, plot_beta_comparison

def main():
    # True simulation parameters
    mu_true = 0.0005        
    alpha_true = 0.5        
    beta_true = 1.5         
    sigma_true = 0.5        
    # Set true covariate effects. For num_cov covariates, true_gamma has length = num_cov.
    num_cov = 40
    true_gamma = np.array([0.8] * num_cov)  # for example, each covariate has effect 0.8
    true_params = np.concatenate([np.array([mu_true, alpha_true, beta_true, sigma_true]), true_gamma])
    
    T = 100.0               
    region = (0, 10, 0, 10) 
    area = (region[1]-region[0]) * (region[3]-region[2])
    
    # Generate data for multiple environments.
    num_env = 3
    print("Generating data for", num_env, "environments...")
    events_list = generate_data(num_env, mu_true, alpha_true, beta_true, sigma_true,
                                T, region, num_cov=num_cov, true_gamma=true_gamma)
    
    # Perturb the data using flags (e.g., 50% of environments perturbed with covariate shift and removal)
    events_list = perturb_environments_flags(events_list, perturb_fraction=0.5, 
                                             perturb_cov_shift=True, shift_value=0.5,
                                             perturb_cov_remove=True, remove_indices=[0])
    
    # Create an initial guess for estimation: [μ, α, β, σ, γ₁, γ₂, ..., γ_num_cov]
    initial_guess = np.concatenate([np.array([0.001, 0.4, 1.0, 1.0]), np.zeros(num_cov)])
    bounds = get_bounds(num_cov)
    
    print("\n==== Non-Penalized Estimation ====")
    theta_list_np, theta_global_np = estimate_non_penalized(events_list, T, area, initial_guess, bounds)
    
    print("\n==== Penalized Estimation ====")
    kappa = 10.0  # Adjust penalty strength as needed
    theta_list_p, theta_global_p = estimate_penalized(events_list, T, area, kappa, initial_guess, bounds)
    
    # Compare estimates with true parameters.
    print("\nTrue parameters:")
    print(true_params)
    print("\nDifference (Non-Penalized Global - True):")
    print(theta_global_np - true_params)
    print("\nDifference (Penalized Global - True):")
    print(theta_global_p - true_params)
    
    # In particular, measure the error for the first covariate effect (b1)
    error_b1_np = theta_global_np[4] - true_gamma[0]
    error_b1_p = theta_global_p[4] - true_gamma[0]
    print("\nError for b1 (Non-Penalized):", error_b1_np)
    print("Error for b1 (Penalized):", error_b1_p)
    
    # Visualize spatial locations for the first environment.
    plot_spatial_events(events_list[0], title="Simulated Marked Spatio-Temporal Hawkes Process (Environment 1)")
    
    # Run multiple simulations and compare the β estimates.
    nsim = 2
    beta_np_list = []
    beta_p_list = []
    
    print("\nRunning multiple simulations for β estimation comparison...")
    for sim in range(nsim):
        events_list = generate_data(num_env, mu_true, alpha_true, beta_true, sigma_true,
                                    T, region, num_cov=num_cov, true_gamma=true_gamma)
        _, theta_global_np = estimate_non_penalized(events_list, T, area, initial_guess, bounds)
        _, theta_global_p = estimate_penalized(events_list, T, area, kappa, initial_guess, bounds)
        beta_np_list.append(theta_global_np[2])
        beta_p_list.append(theta_global_p[2])
    
    plot_beta_comparison(beta_np_list, beta_p_list, beta_true, save_path='beta_comparison.png')

if __name__ == "__main__":
    main()
