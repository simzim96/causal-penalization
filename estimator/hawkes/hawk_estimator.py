import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simulate.simulate_marked_temporal import generate_data

def spatial_kernel(x_diff, y_diff, sigma):
    """
    Compute the Gaussian spatial kernel for the Hawkes process.
    
    This function calculates the probability density of spatial influence 
    between two events separated by x_diff and y_diff. The kernel follows 
    a bivariate Gaussian distribution centered at the origin.
    
    Parameters:
        x_diff (float): The x-coordinate difference between two spatial points.
        y_diff (float): The y-coordinate difference between two spatial points.
        sigma (float): The standard deviation parameter controlling the spatial spread.
                      Larger values result in wider spatial influence.
    
    Returns:
        float: The value of the kernel at the given spatial difference.
    """
    return (1.0 / (2 * np.pi * sigma**2)) * np.exp(-(x_diff**2 + y_diff**2) / (2 * sigma**2))

def neg_log_likelihood(params, events, T, area):
    """
    Calculate the negative log-likelihood for the marked spatio-temporal Hawkes process.
    
    This function computes the negative log-likelihood of observing a set of events
    under the marked spatio-temporal Hawkes process model with parameters given by
    'params'. The likelihood consists of two components:
    1. The sum of log-intensities at each event time/location
    2. The negative integral of the intensity function over the entire space-time domain
    
    Model:
      λ(t,x,y;z) = μ + ∑_{t_j < t} α * β * exp(-β*(t-t_j)) * exp(γ₁*z1_j + γ₂*z2_j) 
                    * spatial_kernel(x - x_j, y - y_j; σ)
    
    Parameters:
        params (array-like): Model parameters [μ, α, β, σ, γ₁, γ₂]
            μ: Background intensity
            α: Branching ratio (expected number of direct offspring per event)
            β: Temporal decay rate
            σ: Spatial standard deviation
            γ₁, γ₂: Coefficients for the mark covariates
        events (np.array): Array of events with shape (n_events, 5), where each row is
                          [t, x, y, z1, z2] representing time, x-coord, y-coord, and two covariates.
        T (float): End time of the observation period
        area (float): Area of the spatial region
    
    Returns:
        float: Negative log-likelihood value (-log L)
    
    Notes:
        - Returns a large penalty (1e10) for invalid parameter values (μ, α, β, σ ≤ 0)
        - The integrated intensity is approximated, assuming uniform spatial distribution
    """
    mu, alpha, beta, sigma, gamma1, gamma2 = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or sigma <= 0:
        return 1e10
    
    n = events.shape[0]
    logL = 0.0
    for i in range(n):
        t_i, x_i, y_i, _, _ = events[i]
        intensity = mu
        for j in range(i):
            t_j, x_j, y_j, z1_j, z2_j = events[j]
            dt = t_i - t_j
            if dt <= 0:
                continue
            contribution = alpha * beta * np.exp(-beta * dt) * np.exp(gamma1 * z1_j + gamma2 * z2_j)
            contribution *= spatial_kernel(x_i - x_j, y_i - y_j, sigma)
            intensity += contribution
        if intensity <= 0:
            return 1e10
        logL += np.log(intensity)
    integrated_background = mu * T * area
    integrated_offspring = 0.0
    for j in range(n):
        t_j, _, _, z1_j, z2_j = events[j]
        integrated_offspring += alpha * np.exp(gamma1 * z1_j + gamma2 * z2_j) * (1 - np.exp(-beta * (T - t_j)))
    logL -= (integrated_background + integrated_offspring)
    return -logL

# Bounds for parameters: μ, α, β, σ > 0; γ₁ and γ₂ are free.
bounds = [(1e-6, None), (1e-6, 0.99), (1e-6, None), (1e-6, None), (None, None), (None, None)]

def estimate_non_penalized(events_list, T, area, initial_guess):
    """
    Estimate parameters for each environment independently without penalization.
    
    This function performs maximum likelihood estimation for each environment
    separately by minimizing the negative log-likelihood. It then computes a
    global estimate by averaging the environment-specific estimates.
    
    Parameters:
        events_list (list of np.array): List of event arrays, one per environment.
                                       Each array has shape (n_events, 5) with columns
                                       [t, x, y, z1, z2].
        T (float): End time of the observation period
        area (float): Area of the spatial region
        initial_guess (np.array): Initial parameter values [μ, α, β, σ, γ₁, γ₂]
    
    Returns:
        theta_list (np.array): Array of parameter estimates per environment with
                               shape (n_environments, n_parameters)
        theta_global (np.array): Global parameter estimate (average over environments)
                                with shape (n_parameters,)
    
    Notes:
        - Uses L-BFGS-B optimization with bounds to ensure valid parameter values
        - Prints estimation results for each environment and the global estimate
    """
    theta_list = []
    for idx, events in enumerate(events_list):
        res = minimize(neg_log_likelihood, initial_guess, args=(events, T, area),
                       bounds=bounds, method='L-BFGS-B')
        if res.success:
            theta_list.append(res.x)
            print(f"Non-penalized estimation, Environment {idx+1}: {res.x}")
        else:
            print(f"Non-penalized estimation failed for Environment {idx+1}: {res.message}")
    theta_list = np.array(theta_list)
    theta_global = np.mean(theta_list, axis=0)
    print("\nGlobal (non-penalized) estimate (average over environments):")
    print(theta_global)
    return theta_list, theta_global

def estimate_penalized(events_list, T, area, kappa, initial_guess, tol=1e-4, max_iter=20):
    """
    Perform penalized parameter estimation across multiple environments.
    
    This function implements the causal penalized estimator for marked spatio-temporal 
    Hawkes processes. It iteratively estimates environment-specific parameters while
    penalizing their deviation from the global average, encouraging the discovery of
    invariant (causal) parameters.
    
    For each environment e, we minimize:
       neg_log_likelihood(theta, events_e) + 0.5 * kappa * ||theta - theta_global||^2,
    where theta_global is the current global average.
    
    Parameters:
        events_list (list of np.array): List of event arrays, one per environment.
                                       Each array has shape (n_events, 5) with columns
                                       [t, x, y, z1, z2].
        T (float): End time of the observation period
        area (float): Area of the spatial region
        kappa (float): Penalization strength parameter. Higher values enforce more
                       similarity between environment-specific and global parameters.
        initial_guess (np.array): Initial parameter values [μ, α, β, σ, γ₁, γ₂]
        tol (float, optional): Convergence tolerance. Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations. Defaults to 20.
    
    Returns:
        theta_list (np.array): Array of environment-specific parameter estimates after
                              penalization with shape (n_environments, n_parameters)
        theta_global (np.array): Global parameter estimate (average of penalized
                                environment-specific estimates) with shape (n_parameters,)
    
    Notes:
        - The algorithm starts with non-penalized estimates and then iteratively
          applies the penalization until convergence
        - Convergence is determined by the change in the global parameter estimate
        - Prints the estimation results for each environment and the global estimate
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
                """
                Penalized objective function for a specific environment.
                
                Combines the negative log-likelihood with a penalty term that encourages
                the environment-specific parameters to be close to the global parameters.
                
                Parameters:
                    theta (np.array): Parameters to be optimized [μ, α, β, σ, γ₁, γ₂]
                
                Returns:
                    float: Value of the penalized objective function
                """
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
        print(f"Penalized estimate, Environment {idx+1}: {theta}")
    return theta_list, theta_global

