import numpy as np

def causal_penalized_estimator_iterative(X_envs, Y_envs, lambda_reg, tol=1e-6, max_iter=1000):
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
        
    Returns:
        beta (np.array): The estimated global regression coefficients.
        beta_envs (list of np.array): The estimated environment-specific coefficients.
    """
    n_env = len(X_envs)
    p = X_envs[0].shape[1]
    
    # Initial OLS estimates for each environment
    beta_envs = [np.linalg.inv(X.T @ X) @ (X.T @ Y) for X, Y in zip(X_envs, Y_envs)]
    # Global estimate is the average of environment-specific estimates
    beta = np.mean(beta_envs, axis=0)
    
    for iteration in range(max_iter):
        beta_envs_new = []
        # Update each environment's coefficient estimate given the current global beta
        for X, Y in zip(X_envs, Y_envs):
            A = X.T @ X + lambda_reg * np.eye(p)
            b = X.T @ Y + lambda_reg * beta
            beta_e = np.linalg.inv(A) @ b
            beta_envs_new.append(beta_e)
        
        # Update global beta as the average of the updated environment-specific estimates
        beta_new = np.mean(beta_envs_new, axis=0)
        
        # Check for convergence
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            beta_envs = beta_envs_new
            break
        
        beta = beta_new
        beta_envs = beta_envs_new
        
    return beta, beta_envs
