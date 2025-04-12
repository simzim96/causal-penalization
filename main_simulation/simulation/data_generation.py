import numpy as np
from tqdm import tqdm

def simulate_marked_hawkes(mu, alpha, beta, sigma, T, region, num_cov=2, true_gamma=None, max_events=90):
    """
    Simulate a marked spatio-temporal Hawkes process with an arbitrary number of covariates.
    
    Each event is represented as (t, x, y, z1, z2, ..., z_num_cov) where (x,y) is the spatial location 
    and the z's are covariates. Immigrant events are generated with constant intensity and their 
    covariates are drawn uniformly from [0,1]. Offspring events are generated using an exponential 
    waiting time and Gaussian spatial displacement; their covariates are inherited from the parent 
    with added Gaussian noise.
    
    The expected number of offspring for a parent is given by:
       offspring_mean = α * exp(⟨γ, z_parent⟩)
    where true_gamma is the true covariate effect vector.
    
    A maximum number of events (max_events) is enforced to prevent runaway branching.
    
    Parameters:
        mu (float): Background intensity per unit time and area.
        alpha (float): Baseline mean number of offspring per event.
        beta (float): Temporal decay rate.
        sigma (float): Standard deviation of the Gaussian spatial kernel.
        T (float): Time horizon.
        region (tuple): (xmin, xmax, ymin, ymax) for spatial domain.
        num_cov (int): Number of covariates per event.
        true_gamma (np.ndarray): True covariate effect vector (length=num_cov).
        max_events (int): Maximum number of events to simulate.
        
    Returns:
        events (np.ndarray): Array of events with shape (n_events, 3 + num_cov)
    """
    xmin, xmax, ymin, ymax = region
    area = (xmax - xmin) * (ymax - ymin)
    
    # Generate immigrant events
    N_immigrants = np.random.poisson(mu * T * area)
    immigrants = []
    for _ in range(N_immigrants):
        t = np.random.uniform(0, T)
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        covs = np.random.uniform(0, 1, size=num_cov)
        immigrants.append((t, x, y, *covs))
    immigrants.sort(key=lambda ev: ev[0])
    
    events = immigrants.copy()
    queue = immigrants.copy()
    
    while queue and len(events) < max_events:
        parent = queue.pop(0)
        t_parent, x_parent, y_parent, *cov_parent = parent
        # Compute offspring mean with covariate modulation (if true_gamma is provided)
        if true_gamma is None:
            offspring_mean = alpha
        else:
            offspring_mean = alpha * np.exp(np.dot(cov_parent, true_gamma))
        N_offspring = np.random.poisson(offspring_mean)
        for _ in range(N_offspring):
            dt = np.random.exponential(1.0 / beta)
            t_child = t_parent + dt
            if t_child > T:
                continue
            dx = np.random.normal(0, sigma)
            dy = np.random.normal(0, sigma)
            x_child = x_parent + dx
            y_child = y_parent + dy
            cov_child = np.clip(np.array(cov_parent) + np.random.normal(0, 0.1, size=num_cov), 0, 1)
            child = (t_child, x_child, y_child, *cov_child)
            events.append(child)
            queue.append(child)
            # If we hit the maximum number of events, break early.
            if len(events) >= max_events:
                print("Maximum number of events reached. Stopping simulation.")
                break
    return np.array(events)

def generate_data(num_env, mu, alpha, beta, sigma, T, region, num_cov=2, true_gamma=None):
    """
    Generate simulation data for a given number of environments.
    
    Returns:
        events_list: List of event arrays (one per environment).
    """
    events_list = []
    for i in tqdm(range(num_env), desc="Generating data"):
        events = simulate_marked_hawkes(mu, alpha, beta, sigma, T, region, num_cov=num_cov,
                                        true_gamma=true_gamma)
        events_list.append(events)
        print(f"Environment {i+1}: {events.shape[0]} events simulated.")
    return events_list 