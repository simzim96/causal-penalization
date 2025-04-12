import numpy as np
import matplotlib.pyplot as plt

def causal_penalized_estimator_iterative(X_envs, Y_envs, lambda_reg, tol=1e-6, max_iter=1000, track_history=False):
    """
    Calculate the causal penalized regression estimator based on Bühlmann's invariance principle.
    
    This method implements the causal penalized regression estimator that estimates environment-specific 
    coefficients (beta_envs) and a global coefficient (beta) by minimizing:
    
        sum_{e in E} ||Y^e - X^e beta^e||^2 + lambda * sum_{e in E} ||beta^e - beta||^2,
    
    where:
      - beta^e are the environment-specific coefficients.
      - beta is the global (invariant) coefficient that represents causal parameters.
      - lambda controls the strength of the penalization.
    
    The iterative algorithm alternates between:
    1. Updating environment-specific coefficients (beta_envs) given the current global beta
    2. Updating the global beta as the median of environment-specific coefficients
    
    Parameters:
        X_envs (list of np.array): List of predictor matrices, one per environment.
                                  Each matrix should have shape (n_samples, n_features).
        Y_envs (list of np.array): List of response vectors, one per environment.
                                  Each vector should have shape (n_samples,).
        lambda_reg (float): Regularization parameter controlling the penalization intensity.
                           Higher values enforce more similarity between environment-specific 
                           and global coefficients.
        tol (float, optional): Tolerance for convergence. The algorithm stops when the
                              change in beta is below this threshold. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        track_history (bool, optional): If True, track and return the history of beta values
                                       for convergence analysis. Defaults to False.
        
    Returns:
        beta (np.array): The estimated global regression coefficients that are invariant 
                        across environments, representing causal parameters.
        beta_envs (list of np.array): The estimated environment-specific coefficients.
        beta_history (list of np.array, optional): History of beta values across iterations 
                                                 if track_history=True.
    
    Notes:
        - The median is used for the global beta calculation to provide robustness against
          outlier environments.
        - The method assumes that the true causal mechanism is invariant across environments,
          while non-causal mechanisms may vary.
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

