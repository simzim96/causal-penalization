import numpy as np

def simulate_marked_hawkes(mu, alpha, beta, sigma, T, region, seed=None):
    """
    Simulate a marked spatio-temporal Hawkes process with covariates.
    
    This function generates a realization of a marked spatio-temporal Hawkes process,
    which models self-exciting point processes where each event increases the probability
    of future events. The process consists of:
    
    1. Background (immigrant) events generated according to a homogeneous Poisson process 
       with intensity μ across the spatial region.
    2. Offspring events triggered by previous events, where the number of offspring follows 
       a Poisson distribution with mean α (branching ratio).
    
    Each event is represented as (t, x, y, z1, z2), where:
      - t: timestamp
      - (x,y): spatial location 
      - (z1,z2): covariates (marks)
    
    Parameters:
        mu (float): Background intensity (events per unit space-time)
        alpha (float): Branching ratio (expected number of direct offspring per event)
        beta (float): Temporal decay rate for the exponential triggering kernel
        sigma (float): Spatial standard deviation for the Gaussian triggering kernel
        T (float): End time of the observation period
        region (tuple): Spatial region as (xmin, xmax, ymin, ymax)
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    
    Returns:
        np.array: Array of events with shape (n_events, 5), where each row contains
                 [t, x, y, z1, z2] for an event.
    
    Notes:
        - Immigrant events have covariates drawn uniformly from [0,1]
        - Offspring events inherit covariates from their parent with added Gaussian noise
        - Events are sorted chronologically in the output
    """
    if seed is not None:
        np.random.seed(seed)
    xmin, xmax, ymin, ymax = region
    area = (xmax - xmin) * (ymax - ymin)
    
    # Simulate immigrant (background) events.
    N_immigrants = np.random.poisson(mu * T * area)
    immigrants = []
    for _ in range(N_immigrants):
        t = np.random.uniform(0, T)
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        z1 = np.random.uniform(0, 1)
        z2 = np.random.uniform(0, 1)
        immigrants.append((t, x, y, z1, z2))
    immigrants.sort(key=lambda ev: ev[0])
    
    events = immigrants.copy()
    queue = immigrants.copy()
    
    # Offspring generation (branching)
    while queue:
        parent = queue.pop(0)
        t_parent, x_parent, y_parent, z1_parent, z2_parent = parent
        N_offspring = np.random.poisson(alpha)
        for _ in range(N_offspring):
            dt = np.random.exponential(1.0 / beta)
            t_child = t_parent + dt
            if t_child > T:
                continue
            dx = np.random.normal(0, sigma)
            dy = np.random.normal(0, sigma)
            x_child = x_parent + dx
            y_child = y_parent + dy
            # Offspring covariates: parent's covariates plus small noise (clipped to [0,1])
            z1_child = np.clip(z1_parent + np.random.normal(0, 0.1), 0, 1)
            z2_child = np.clip(z2_parent + np.random.normal(0, 0.1), 0, 1)
            child = (t_child, x_child, y_child, z1_child, z2_child)
            events.append(child)
            queue.append(child)
    
    events.sort(key=lambda ev: ev[0])
    return np.array(events)


def generate_data(num_env, mu, alpha, beta, sigma, gamma1, gamma2, T, region, seeds=None):
    """
    Generate simulation data for multiple environments of marked Hawkes processes.
    
    This function creates multiple realizations of the marked spatio-temporal Hawkes process,
    each representing a different environment. This is useful for testing causal inference
    methods across environments with the same underlying parameters but different realizations.
    
    Parameters:
        num_env (int): Number of environments to generate
        mu (float): Background intensity (events per unit space-time)
        alpha (float): Branching ratio (expected number of direct offspring per event)
        beta (float): Temporal decay rate for the exponential triggering kernel
        sigma (float): Spatial standard deviation for the Gaussian triggering kernel
        gamma1 (float): First covariate effect parameter
        gamma2 (float): Second covariate effect parameter
        T (float): End time of the observation period
        region (tuple): Spatial region as (xmin, xmax, ymin, ymax)
        seeds (list of int, optional): Random seeds for each environment. If None,
                                      uses sequential seeds starting from 100.
    
    Returns:
        list: List of np.arrays, where each array contains the events for one environment
              in the format [t, x, y, z1, z2].
    
    Notes:
        - Prints summary statistics for each generated environment
        - Different seeds ensure independent realizations across environments
    """
    if seeds is None:
        # If seeds are not provided, use default different seeds.
        seeds = np.arange(100, 100+num_env)
    events_list = []
    for i in range(num_env):
        events = simulate_marked_hawkes(mu, alpha, beta, sigma, T, region, seed=seeds[i])
        events_list.append(events)
        print(f"Environment {i+1} (seed={seeds[i]}): {events.shape[0]} events simulated.")
    return events_list

