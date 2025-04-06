import matplotlib.pyplot as plt
import numpy as np

def plot_spatial_events(events, title="Simulated Marked Spatio-Temporal Hawkes Process"):
    """
    Visualize spatial locations of events with time as color.
    
    Parameters:
        events (np.ndarray): Array of events with shape (n_events, 3 + num_cov)
        title (str): Title for the plot
    """
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(events[:, 1], events[:, 2], c=events[:, 0], cmap='viridis', s=10)
    plt.colorbar(sc, label='Time')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()

def plot_beta_comparison(beta_np_list, beta_p_list, beta_true, save_path=None):
    """
    Create a boxplot comparing beta estimates from non-penalized and penalized methods.
    
    Parameters:
        beta_np_list (list): List of beta estimates from non-penalized method
        beta_p_list (list): List of beta estimates from penalized method
        beta_true (float): True beta value
        save_path (str, optional): Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([beta_np_list, beta_p_list], positions=[1, 2], widths=0.6, patch_artist=True)
    bp['boxes'][0].set(facecolor='lightblue', alpha=0.7)
    bp['boxes'][1].set(facecolor='lightgreen', alpha=0.7)
    ax.axhline(y=beta_true, color='r', linestyle='--', label='True β')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Non-Penalized', 'Penalized'])
    ax.set_ylabel('β Estimate')
    ax.set_title('Comparison of β Estimates: Non-Penalized vs Penalized')
    ax.legend()
    plt.show()
    
    if save_path:
        plt.savefig(save_path) 