import matplotlib.pyplot as plt
import numpy as np

def plot_spatial_events(events, title="Simulated Marked Spatio-Temporal Hawkes Process", save_path=None):
    """
    Visualize spatial locations of events with time as color.
    
    Parameters:
        events (np.ndarray): Array of events with shape (n_events, 3 + num_cov)
        title (str): Title for the plot
    """
    try:
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(events[:, 1], events[:, 2], c=events[:, 0], cmap='viridis', s=10)
        plt.colorbar(sc, label='Time')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.savefig(save_path)
        plt.close()
    except:
        print(f"Error plotting events: {events}")

def plot_beta_comparison(beta_np_list, beta_p_list, beta_true, save_path=None, filter_outliers=True, k=1.5):
    """
    Create a boxplot comparing beta estimates from non-penalized and penalized methods.
    
    Parameters:
        beta_np_list (list): List of beta estimates from non-penalized method
        beta_p_list (list): List of beta estimates from penalized method
        beta_true (float): True beta value
        save_path (str, optional): Path to save the figure
        filter_outliers (bool): Whether to filter outliers before plotting
        k (float): Multiplier for IQR to determine outlier threshold (default: 1.5)
    """
    # Filter outliers if requested
    if filter_outliers:
        # Non-penalized outlier filtering
        np_array = np.array(beta_np_list)
        q1_np, q3_np = np.percentile(np_array, [25, 75])
        iqr_np = q3_np - q1_np
        lower_bound_np = q1_np - k * iqr_np
        upper_bound_np = q3_np + k * iqr_np
        filtered_np_list = [x for x in beta_np_list if lower_bound_np <= x <= upper_bound_np]
        
        # Penalized outlier filtering
        p_array = np.array(beta_p_list)
        q1_p, q3_p = np.percentile(p_array, [25, 75])
        iqr_p = q3_p - q1_p
        lower_bound_p = q1_p - k * iqr_p
        upper_bound_p = q3_p + k * iqr_p
        filtered_p_list = [x for x in beta_p_list if lower_bound_p <= x <= upper_bound_p]
        
        # Use filtered data for plotting
        plot_data = [filtered_np_list, filtered_p_list]
    else:
        plot_data = [beta_np_list, beta_p_list]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(plot_data, positions=[1, 2], widths=0.6, patch_artist=True)
    bp['boxes'][0].set(facecolor='lightblue', alpha=0.7)
    bp['boxes'][1].set(facecolor='lightgreen', alpha=0.7)
    ax.axhline(y=beta_true, color='r', linestyle='--', label='True Value')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Non-Penalized', 'Penalized'])
    ax.set_ylabel('Estimate')
    ax.set_title('Comparison of Estimates: Non-Penalized vs Penalized')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path) 
    plt.close()