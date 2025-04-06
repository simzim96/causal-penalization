import numpy as np

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
      λ(t,x,y; z) = μ + ∑_{t_j < t} α * β * exp(-β*(t-t_j)) * exp(⟨γ, z_j⟩)
    where ⟨γ, z_j⟩ is the dot product between the covariate effect vector γ and the covariate vector z_j.
    
    The integrated intensity is approximated as:
      μ * T * area + ∑_j [α * exp(⟨γ, z_j⟩) * (1 - exp(-β*(T-t_j)))]
    
    Parameters in params: [μ, α, β, σ, γ₁, γ₂, ..., γ_m]
    """
    mu = params[0]
    alpha = params[1]
    beta = params[2]
    sigma = params[3]
    gamma = params[4:]
    
    if mu <= 0 or alpha <= 0 or beta <= 0 or sigma <= 0:
        return 1e10
    
    n = events.shape[0]
    logL = 0.0
    for i in range(n):
        t_i, x_i, y_i = events[i, :3]
        intensity = mu
        for j in range(i):
            t_j, x_j, y_j = events[j, :3]
            z_j = events[j, 3:]
            dt = t_i - t_j
            if dt <= 0:
                continue
            contribution = alpha * beta * np.exp(-beta * dt) * np.exp(np.dot(gamma, z_j))
            contribution *= spatial_kernel(x_i - x_j, y_i - y_j, sigma)
            intensity += contribution
        if intensity <= 0:
            return 1e10
        logL += np.log(intensity)
    
    integrated_background = mu * T * area
    integrated_offspring = 0.0
    for j in range(n):
        t_j = events[j, 0]
        z_j = events[j, 3:]
        integrated_offspring += alpha * np.exp(np.dot(gamma, z_j)) * (1 - np.exp(-beta * (T - t_j)))
    
    logL -= (integrated_background + integrated_offspring)
    return -logL 