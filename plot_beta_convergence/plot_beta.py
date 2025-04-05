import numpy as np
from estimator.estimator import causal_penalized_estimator_iterative
import matplotlib.pyplot as plt

def plot_beta_convergence(beta_history, true_beta, max_plot=None, save_path=None):
    """
    Plot the convergence of beta values over iterations.
    
    Parameters:
        beta_history (list of np.array): History of beta values from the estimator.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    beta_history = np.array(beta_history)
    iterations = np.arange(len(beta_history))
    
    plt.figure(figsize=(10, 6))
    
    # Define colors for each coefficient
    
    for i in range(beta_history.shape[1]):
        if max_plot is not None and i > max_plot:
            break
        plt.plot(iterations, beta_history[:, i], label=f'Coefficient {i+1}')
        plt.axhline(y=true_beta[i], linestyle='--', color='black', linewidth=1, alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Beta Value')
    plt.title('Convergence of Beta Values')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


np.random.seed(42)
n_env = 150
n_obs = 50
p = 10
true_beta = np.array([1.0, -.7, .5, -.3, .2, -.1, .1, -.1, .1, -.1])

# Tuning parameters for noise levels
beta_noise_scale = 1.0      # Controls noise added to the coefficients per environment
error_noise_scale = 10.0     # Controls noise added to the response variable

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


beta_hat, _, beta_history = causal_penalized_estimator_iterative(X_envs, Y_envs, 50, tol=1e-7, track_history=True)

plot_beta_convergence(beta_history,true_beta, max_plot=3, save_path='./plot_beta_convergence/plots/beta_convergence.png')