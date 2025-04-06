import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simulate.simulate_marked_temporal import generate_data

def spatial_kernel(x_diff, y_diff, sigma):
    """
    Gaussian spatial kernel:
      f(delta) = 1/(2*pi*sigma^2) * exp(-||delta||^2/(2*sigma^2))
    """
    return (1.0 / (2 * np.pi * sigma**2)) * np.exp(-(x_diff**2 + y_diff**2) / (2 * sigma**2))

def neg_log_likelihood(params, events, T, area):
    """
    Negative log-likelihood for the marked spatio-temporal Hawkes process.
    
    Model:
      λ(t,x,y;z) = μ + ∑_{t_j < t} α * β * exp(-β*(t-t_j)) * exp(γ₁*z1_j + γ₂*z2_j) 
                    * spatial_kernel(x - x_j, y - y_j; σ)
    
    Integrated intensity approximates:
      μ * T * area + ∑_j [α * exp(γ₁*z1_j + γ₂*z2_j) * (1 - exp(-β*(T-t_j)))]
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
    Perform penalized estimation across multiple environments.
    
    For each environment e, we minimize:
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
        print(f"Penalized estimate, Environment {idx+1}: {theta}")
    return theta_list, theta_global

