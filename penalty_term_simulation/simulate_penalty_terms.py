import numpy as np
import matplotlib.pyplot as plt
from estimator.estimator import causal_penalized_estimator_iterative

# Simulation settings
np.random.seed(42)
n_env = 10
n_obs = 500
p = 2
true_beta = np.array([1.0, -1.0])

# Tuning parameters for noise levels
beta_noise_scale = 3.0      # Controls noise added to the coefficients per environment
error_noise_scale = 1.0     # Controls noise added to the response variable

# Generate synthetic data for multiple environments with distributional shifts
# Generate base environments with no perturbation
X_envs = [np.random.randn(n_obs, p) for _ in range(n_env)]

# Add large perturbations to a few selected environments
perturbed_envs = np.random.choice(n_env, size=5, replace=False)
large_shift_scale = 50.0  # Much larger shift for selected environments

for env_idx in perturbed_envs:
    X_envs[env_idx] = X_envs[env_idx] + np.random.normal(0, large_shift_scale, size=(1, p))

Y_envs = [
    X @ (true_beta + np.random.normal(0, beta_noise_scale, size=p)) + np.random.normal(0, error_noise_scale, size=n_obs)
    for X in X_envs
]

# Range of lambda values to explore
lambda_values = np.linspace(0, 2000, 500)
estimates = []

for lambda_reg in lambda_values:
    beta_hat, _ = causal_penalized_estimator_iterative(X_envs, Y_envs, lambda_reg, tol=1e-7)
    estimates.append(beta_hat)

estimates = np.array(estimates)
colors = ['blue', 'orange', 'green', 'red']

# Plotting the impact of lambda on estimates
plt.figure(figsize=(10, 6))
for i in range(p):
    plt.plot(lambda_values, estimates[:, i], label=f'Coefficient {i+1}', color=colors[i])
    plt.axhline(y=true_beta[i], linestyle='--', color=colors[i], linewidth=1)

plt.xlabel('Penalization Parameter $\lambda$')
plt.ylabel('Estimated Coefficient Values')
plt.title('Impact of Penalization Parameter on Regression Estimates')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./penalty_term_simulation/plots/penalty_term_simulation.png')
