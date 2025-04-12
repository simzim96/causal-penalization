import numpy as np

def perturb_environments_flags(events_list, perturb_fraction=0.5,
                               perturb_cov_shift=False, shift_value=0.5,
                               perturb_cov_remove=False, remove_indices=None,
                               perturb_time_offset=False, time_offset=10.0):
    """
    Perturb a fraction of environments (i.e. event arrays) using independent Boolean flags.
    
    Parameters:
        events_list (list of np.ndarray): List of event arrays (one per environment).
            Each event array has shape (n_events, 3 + num_cov) with columns: 
            t, x, y, z1, z2, ..., z_num_cov.
        perturb_fraction (float): Fraction of environments to perturb (between 0 and 1).
        perturb_cov_shift (bool): If True, add a constant shift (shift_value) to all covariate columns.
        shift_value (float): The constant value to add to the covariates (used if perturb_cov_shift is True).
        perturb_cov_remove (bool): If True, remove (set to zero) specified covariate columns.
        remove_indices (list of int): List of indices (within the covariate block) to set to zero.
            If None and perturb_cov_remove is True, defaults to the first covariate.
        perturb_time_offset (bool): If True, add a constant time offset (time_offset) to all event times.
        time_offset (float): The constant value to add to the event times (used if perturb_time_offset is True).
    
    Returns:
        perturbed_events_list (list of np.ndarray): List of event arrays after perturbation.
    """
    num_env = len(events_list)
    num_to_perturb = int(np.ceil(perturb_fraction * num_env))
    # Randomly choose which environments to perturb
    indices_to_perturb = np.random.choice(num_env, num_to_perturb, replace=False)
    
    perturbed_events_list = []
    for i, events in enumerate(events_list):
        events_copy = events.copy()  # Copy to avoid modifying original data.
        if i in indices_to_perturb:
            # Perturb covariate shift: add shift_value to all covariate columns (columns 3 onward)
            if perturb_cov_shift:
                events_copy[:, 3:] += shift_value * np.random.randn()
            
            # Perturb covariate removal: set specified covariate columns to zero.
            if perturb_cov_remove:
                if remove_indices is None:
                    remove_indices = [0]
                for idx in remove_indices:
                    try:
                        events_copy[:, 3 + idx] = 0
                    except:
                        print(f"Error: Id out of bound.")
            
            # Perturb time offset: add a constant to the event times (column 0)
            if perturb_time_offset:
                events_copy[:, 0] += time_offset
        perturbed_events_list.append(events_copy)
    return perturbed_events_list 