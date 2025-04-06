from simulate.simulate_marked_temporal import generate_data
from estimator.hawkes.hawk_estimator import estimate_non_penalized, estimate_penalized
import numpy as np
import matplotlib.pyplot as plt

# True simulation parameters
mu_true = 0.0005        
alpha_true = 0.5        
beta_true = 1.5         
sigma_true = 0.5        
gamma1_true = 0.8       
gamma2_true = -0.5      
T = 100.0               
region = (0, 10, 0, 10) 
area = (region[1]-region[0]) * (region[3]-region[2])

# Generate data for a given number of environments.
num_env = 3
seeds = [42, 43, 44]
events_list = generate_data(num_env, mu_true, alpha_true, beta_true, sigma_true,
                             gamma1_true, gamma2_true, T, region, seeds=seeds)

# Set an initial guess for parameter estimation:
# [μ, α, β, σ, γ₁, γ₂]
initial_guess = np.array([0.001, 0.4, 1.0, 1.0, 0.0, 0.0])

print("\n==== Non-Penalized Estimation ====")
theta_list_np, theta_global_np = estimate_non_penalized(events_list, T, area, initial_guess)

print("\n==== Penalized Estimation ====")
kappa = 10.0  # Penalty strength (adjustable)
theta_list_p, theta_global_p = estimate_penalized(events_list, T, area, kappa, initial_guess, tol=1e-4, max_iter=20)

# Compare estimates with true parameters
true_params = np.array([mu_true, alpha_true, beta_true, sigma_true, gamma1_true, gamma2_true])
print("\nTrue parameters:")
print(true_params)
print("\nDifference (Non-Penalized Global - True):")
print(theta_global_np - true_params)
print("\nDifference (Penalized Global - True):")
print(theta_global_p - true_params)

# Optional: Visualize one environment's spatial locations.
plt.figure(figsize=(8,6))
sc = plt.scatter(events_list[0][:,1], events_list[0][:,2], c=events_list[0][:,0], cmap='viridis', s=10)
plt.colorbar(sc, label='Time')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simulated Marked Spatio-Temporal Hawkes Process (Environment 1)")
plt.show()

