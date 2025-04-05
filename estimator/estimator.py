import numpy as np
import matplotlib.pyplot as plt

def causal_penalized_estimator_iterative(X_envs, Y_envs, lambda_reg, tol=1e-6, max_iter=1000, track_history=False):
    """
    Calculate the causal penalized regression estimator based on Bühlmann's invariance principle.
    
    This method jointly estimates environment-specific coefficients (beta_envs) and a global coefficient (beta)
    by minimizing:
    
        sum_{e in E} ||Y^e - X^e beta^e||^2 + lambda * sum_{e in E} ||beta^e - beta||^2,
    
    where:
      - beta^e are the environment-specific coefficients.
      - beta is the global (invariant) coefficient.
    
    Parameters:
        X_envs (list of np.array): List of predictor matrices, one per environment.
        Y_envs (list of np.array): List of response vectors, one per environment.
        lambda_reg (float): Regularization parameter controlling the penalization intensity.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.
        track_history (bool): If True, track and return the history of beta values.
        
    Returns:
        beta (np.array): The estimated global regression coefficients.
        beta_envs (list of np.array): The estimated environment-specific coefficients.
        beta_history (list of np.array, optional): History of beta values if track_history=True.
    """
    n_env = len(X_envs)
    p = X_envs[0].shape[1]
    
    # Initial OLS estimates for each environment
    beta_envs = [np.linalg.inv(X.T @ X) @ (X.T @ Y) for X, Y in zip(X_envs, Y_envs)]
    # Global estimate is the average of environment-specific estimates
    beta = np.mean(beta_envs, axis=0)
    
    beta_history = [beta.copy()] if track_history else None
    
    stability_counter = 0
    for iteration in range(max_iter):
        print(f"Iteration {iteration}")
        beta_envs_new = []
        # Update each environment's coefficient estimate given the current global beta
        for X, Y in zip(X_envs, Y_envs):
            A = X.T @ X + lambda_reg * np.eye(p)
            b = X.T @ Y + lambda_reg * beta
            beta_e = np.linalg.inv(A) @ b
            beta_envs_new.append(beta_e)
        
        # Update global beta as the average of the updated environment-specific estimates
        beta_new = np.median(beta_envs_new, axis=0)
        
        if track_history:
            beta_history.append(beta_new.copy())
        
        # Check for convergence
        if np.linalg.norm(beta_new - beta) < tol:
            stability_counter += 1
            if stability_counter >= 1:
                beta = beta_new
                beta_envs = beta_envs_new
                break
        
        beta = beta_new
        beta_envs = beta_envs_new
    
    if track_history:
        return beta, beta_envs, beta_history
    else:
        return beta, beta_envs

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
    for i in range(beta_history.shape[1]):
        if max_plot is not None and i > max_plot:
            break
        plt.plot(iterations, beta_history[:, i], label=f'Coefficient {i+1}')

    for i in range(beta_history.shape[1]):
        if max_plot is not None and i > max_plot:
            break
        plt.axhline(y=true_beta[i], linestyle='--', color='black', linewidth=1)
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
