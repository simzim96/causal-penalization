import numpy as np
from scipy.optimize import minimize
from .likelihood import neg_log_likelihood

def get_bounds(num_cov):
    """
    Create bounds for the parameters.
    For μ, α, β, σ > 0 and for each γ, no bounds.
    """
    base_bounds = [(1e-6, None), (1e-6, 0.99), (1e-6, None), (1e-6, None)]
    gamma_bounds = [(None, None)] * num_cov
    return base_bounds + gamma_bounds

def estimate_non_penalized(events_list, T, area, initial_guess, bounds):
    """
    Estimate parameters for each environment independently (without penalization).
    
    Returns:
        theta_list: List of parameter estimates per environment.
        theta_global: Global estimate (average over environments).
    """
    theta_list = []
    for idx, events in enumerate(events_list):
        res = minimize(neg_log_likelihood, initial_guess, args=(events, T, area),
                       bounds=bounds, method='L-BFGS-B')
        if res.success:
            theta_list.append(res.x)
            print(f"Non-penalized, Environment {idx+1}: {res.x}")
        else:
            print(f"Non-penalized estimation failed for Environment {idx+1}: {res.message}")
    theta_list = np.array(theta_list)
    theta_global = np.mean(theta_list, axis=0)
    print("\nGlobal (non-penalized) estimate (average over environments):")
    print(theta_global)
    return theta_list, theta_global

def estimate_penalized(events_list, T, area, kappa, initial_guess, bounds, tol=1e-4, max_iter=20):
    """
    Perform penalized estimation across multiple environments.
    
    For each environment e, minimize:
       neg_log_likelihood(theta, events_e) + 0.5 * kappa * ||theta - theta_global||^2,
    where theta_global is the current global average.
    
    Returns:
        theta_global: Global (invariant) parameter estimate.
        theta_list: List of environment-specific estimates.
    """
    E = len(events_list)
    theta_list = []
    for events in events_list:
        res = minimize(neg_log_likelihood, initial_guess, args=(events, T, area),
                       bounds=bounds, method='L-BFGS-B')
        theta_list.append(res.x)
    theta_list = np.array(theta_list)
    theta_global = np.mean(theta_list, axis=0)
    
    for iteration in range(max_iter):
        theta_list_new = []
        for e in range(E):
            def penalized_obj(theta):
                return neg_log_likelihood(theta, events_list[e], T, area) + 0.5 * kappa * np.sum((theta - theta_global)**2)
            res = minimize(penalized_obj, theta_list[e], bounds=bounds, method='L-BFGS-B')
            theta_list_new.append(res.x)
        theta_list_new = np.array(theta_list_new)
        theta_global_new = np.mean(theta_list_new, axis=0)
        if np.linalg.norm(theta_global_new - theta_global) < tol:
            theta_global = theta_global_new
            theta_list = theta_list_new
            print(f"Penalized estimation converged in {iteration+1} iterations.")
            break
        theta_global = theta_global_new
        theta_list = theta_list_new
    print("\nGlobal (penalized) estimate:")
    print(theta_global)
    for idx, theta in enumerate(theta_list):
        print(f"Penalized, Environment {idx+1}: {theta}")
    return theta_list, theta_global 