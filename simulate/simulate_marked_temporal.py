import numpy as np

def simulate_marked_hawkes(mu, alpha, beta, sigma, T, region, seed=None):
    """
    Simulate a marked spatio-temporal Hawkes process with covariates.
    
    Each event is represented as (t, x, y, z1, z2), where (x,y) is the spatial location 
    and (z1,z2) are covariates. Immigrant events are generated with constant intensity,
    and their covariates are drawn uniformly from [0,1]. Offspring events are generated
    using an exponential waiting time and Gaussian spatial displacement; their covariates 
    are inherited from the parent with added Gaussian noise.
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
    Generate simulation data for a specified number of environments.
    
    Returns:
        events_list: List of event arrays (one per environment).
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

