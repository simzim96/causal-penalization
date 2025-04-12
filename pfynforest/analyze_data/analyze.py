from pfynforest.load_data.load_data import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the repository root to the Python path to access the estimator
sys.path.append('/Users/simon/Documents/Uni/MASTERARBEIT/causal-penalization')
from estimator.estimator import causal_penalized_estimator_iterative

def prepare_data_by_environment(df):
    """
    Prepare the data for estimation by grouping it into environments based on PLOT NO.
    
    Returns:
        X_envs: List of feature matrices, one per environment (plot)
        Y_envs: List of target vectors, one per environment (plot)
        plot_ids: List of plot numbers corresponding to each environment
    """
    # Normalize coordinates to prevent numerical issues
    X_min, X_max = df['X'].min(), df['X'].max()
    Y_min, Y_max = df['Y'].min(), df['Y'].max()
    
    df['X_norm'] = (df['X'] - X_min) / (X_max - X_min)
    df['Y_norm'] = (df['Y'] - Y_min) / (Y_max - Y_min)
    
    # List of all unique plot numbers
    plot_ids = sorted(df['PLOT NO'].unique())
    
    X_envs = []
    Y_envs = []
    
    for plot_id in plot_ids:
        plot_data = df[df['PLOT NO'] == plot_id]
        # dropna 
        plot_data = plot_data.dropna()
        
        # Extract features: normalized coordinates and social status
        X = plot_data[['X_norm', 'Y_norm', 'SOCIAL STATUS']].values
        
        # Add intercept term
        X = np.column_stack([np.ones(X.shape[0]), X])
                
        # Extract target: total crown defoliation
        Y = plot_data['TOTAL CROWN DEFOLIATION'].values
        
        X_envs.append(X)
        Y_envs.append(Y)
    
    return X_envs, Y_envs, plot_ids

def analyze_penalized_vs_non_penalized(X_envs, Y_envs, plot_ids, lambda_values=None):
    """
    Compare penalized and non-penalized regression approaches.
    
    Parameters:
        X_envs: List of feature matrices, one per environment
        Y_envs: List of target vectors, one per environment
        plot_ids: List of plot identifiers
        lambda_values: List of regularization strengths to try (if None, use default values)
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.1, 1.0, 10.0, 100.0]
    
    # Dictionary to store results
    results = {
        'non_penalized': None,
        'penalized': {}
    }
    
    # Non-penalized estimation (equivalent to lambda=0)
    beta_envs_non_penalized = []
    for X, Y in zip(X_envs, Y_envs):
        # Simple OLS for each environment
        beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
        beta_envs_non_penalized.append(beta)
    
    # Calculate global (averaged) beta
    beta_global_non_penalized = np.mean(beta_envs_non_penalized, axis=0)
    results['non_penalized'] = {
        'beta_global': beta_global_non_penalized,
        'beta_envs': beta_envs_non_penalized
    }
    
    # Penalized estimation for different lambda values
    for lambda_reg in lambda_values:
        if lambda_reg == 0.0:
            continue  # Already calculated as non-penalized
            
        # Apply causal penalized estimator
        beta_global_penalized, beta_envs_penalized = causal_penalized_estimator_iterative(
            X_envs, Y_envs, lambda_reg, tol=1e-6, max_iter=100
        )
        
        results['penalized'][lambda_reg] = {
            'beta_global': beta_global_penalized,
            'beta_envs': beta_envs_penalized
        }
    
    return results

def visualize_results(results, plot_ids):
    """
    Visualize how regression coefficients change with different penalization strengths.
    
    Parameters:
        results: Dictionary containing penalized and non-penalized results
        plot_ids: List of plot identifiers
    """
    # Create output directory for plots
    os.makedirs('pfynforest/analyze_data/plots', exist_ok=True)
    
    # Extract data for non-penalized results (reference point)
    non_penalized_global = results['non_penalized']['beta_global']
    non_penalized_envs = np.array(results['non_penalized']['beta_envs'])
    
    # Get lambda values and ensure they're sorted
    lambda_values = sorted(list(results['penalized'].keys()))
    
    # Determine number of features from the data
    num_features = len(non_penalized_global)
    
    # Define feature names for plots - adjust if necessary
    default_feature_names = ['Intercept', 'X_norm', 'Y_norm', 'Social Status']
    feature_names = default_feature_names[:num_features]  # Truncate to match actual number of features
    
    # If we have fewer names than features, add generic names
    if len(feature_names) < num_features:
        for i in range(len(feature_names), num_features):
            feature_names.append(f'Feature {i}')
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:num_features]
    
    # Only proceed if we have valid lambda values
    if lambda_values:
        # Plot 1: Comparison of global coefficients with different lambda values
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        coeffs_by_lambda = {feature: [] for feature in feature_names}
        
        # Add non-penalized coefficients at lambda=0
        for lambda_val in [0.0] + lambda_values:
            if lambda_val == 0.0:
                # Use non-penalized results for lambda=0
                coefs = non_penalized_global
            else:
                # Use penalized results for other lambda values
                coefs = results['penalized'][lambda_val]['beta_global']
            
            for i, feature in enumerate(feature_names):
                if i < len(coefs):
                    if lambda_val == 0.0:
                        # Starting point at lambda=0 (non-penalized)
                        coeffs_by_lambda[feature].append((lambda_val, coefs[i]))
                    else:
                        # Add penalized coefficient
                        coeffs_by_lambda[feature].append((lambda_val, coefs[i]))
        
        # Plot each coefficient's response to lambda
        for i, feature in enumerate(feature_names):
            if coeffs_by_lambda[feature]:
                x_vals, y_vals = zip(*coeffs_by_lambda[feature])
                plt.plot(x_vals, y_vals, 'o-', label=feature, color=colors[i])
        
        # Only use log scale if all lambda values are positive and we have more than just lambda=0
        if all(lam > 0 for lam in lambda_values) and len(lambda_values) > 1:
            plt.xscale('log')
        
        plt.xlabel('Regularization Strength (λ)')
        plt.ylabel('Coefficient Value')
        plt.title('Effect of Penalization on Regression Coefficients')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pfynforest/analyze_data/plots/coefficient_penalization.png')
        plt.close()
        
        # Plot 2: Coefficient variation across plots (penalized vs non-penalized)
        # Select a specific lambda for demonstration
        lambda_demo = lambda_values[-2] if len(lambda_values) >= 2 else lambda_values[-1]  # Use second last or last
        penalized_envs = np.array(results['penalized'][lambda_demo]['beta_envs'])
        
        # Determine how many subplots we need (max 4 per figure)
        num_rows = (len(feature_names) + 1) // 2  # Ceiling division
        
        # Create a multi-panel plot for each coefficient
        plt.figure(figsize=(16, 4 * num_rows))
        
        for i, feature in enumerate(feature_names):
            if i < len(feature_names):
                plt.subplot(num_rows, 2, i+1)
                
                try:
                    # Plot non-penalized values across plots (with error handling)
                    valid_data = True
                    try:
                        non_pen_values = [env[i] for env in non_penalized_envs if i < len(env)]
                        if len(non_pen_values) != len(plot_ids):
                            valid_data = False
                    except (IndexError, ValueError):
                        valid_data = False
                    
                    if valid_data:
                        plt.plot(plot_ids, non_pen_values, 
                                'o-', label='Non-penalized', color='blue', alpha=0.7)
                    
                    # Plot penalized values across plots (with error handling)
                    valid_data = True
                    try:
                        pen_values = [env[i] for env in penalized_envs if i < len(env)]
                        if len(pen_values) != len(plot_ids):
                            valid_data = False
                    except (IndexError, ValueError):
                        valid_data = False
                    
                    if valid_data:
                        plt.plot(plot_ids, pen_values, 
                                'o-', label=f'Penalized (λ={lambda_demo})', color='red', alpha=0.7)
                    
                    # Add horizontal lines for global estimates if they're valid
                    if i < len(non_penalized_global):
                        plt.axhline(y=non_penalized_global[i], linestyle='--', color='blue', alpha=0.5)
                    
                    if i < len(results['penalized'][lambda_demo]['beta_global']):
                        plt.axhline(y=results['penalized'][lambda_demo]['beta_global'][i], 
                                    linestyle='--', color='red', alpha=0.5)
                except Exception as e:
                    print(f"Error plotting coefficient {feature}: {e}")
                
                plt.xlabel('Plot Number')
                plt.ylabel('Coefficient Value')
                plt.title(f'{feature} Coefficient Variation Across Plots')
                plt.legend()
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('pfynforest/analyze_data/plots/penalized_variation.png')
        plt.close()
        
        # Print summary table
        print("\n=== COEFFICIENT VALUES BY PENALIZATION STRENGTH ===")
        print("Feature\tλ=0 (Non-Penalized)\t" + "\t".join([f"λ={lam}" for lam in lambda_values]))
        print("-" * 80)
        
        for i, feature in enumerate(feature_names):
            if i < len(non_penalized_global):
                row = [feature, f"{non_penalized_global[i]:.4f}"]
                for lam in lambda_values:
                    if i < len(results['penalized'][lam]['beta_global']):
                        row.append(f"{results['penalized'][lam]['beta_global'][i]:.4f}")
                    else:
                        row.append("N/A")
                print("\t".join(row))
        
        # Print variation comparison
        print("\n=== COEFFICIENT VARIATION ACROSS PLOTS ===")
        print(f"Feature\tNon-Penalized Std\tPenalized Std (λ={lambda_demo})\tReduction (%)")
        print("-" * 80)
        
        for i, feature in enumerate(feature_names):
            if i < len(non_penalized_global):
                try:
                    # Calculate standard deviation with error handling
                    non_pen_values = [env[i] for env in non_penalized_envs if i < len(env)]
                    pen_values = [env[i] for env in penalized_envs if i < len(env)]
                    
                    if non_pen_values and pen_values:
                        non_pen_std = np.std(non_pen_values)
                        pen_std = np.std(pen_values)
                        
                        if non_pen_std > 0:
                            reduction = (non_pen_std - pen_std) / non_pen_std * 100
                            print(f"{feature}\t{non_pen_std:.4f}\t{pen_std:.4f}\t{reduction:.2f}%")
                        else:
                            print(f"{feature}\t{non_pen_std:.4f}\t{pen_std:.4f}\tN/A")
                    else:
                        print(f"{feature}\tN/A\tN/A\tN/A")
                except Exception as e:
                    print(f"{feature}\tError: {e}\tError\tN/A")
    else:
        print("No penalized results to plot")

def main():
    """
    Main function to analyze forest defoliation data using causal penalization.
    
    This function:
    1. Loads the forest data from CSV files
    2. Prepares the data by grouping it into environments based on plot numbers
    3. Performs both non-penalized and penalized regression analysis
    4. Visualizes the results of different penalization strengths
    5. Prints a summary of coefficient values
    
    The analysis examines how environmental factors influence tree crown defoliation
    across different forest plots, using a causal penalization approach to identify
    invariant relationships. Features include spatial coordinates and social status
    of trees, with total crown defoliation as the target variable.
    
    Results are saved as plots in the pfynforest/analyze_data/plots directory.
    """
    # Path to the CSV file (update the path if needed)
    
    # Load data
    df = load_data()
    # drop treatment column as stable across environemnts
    df = df.drop(columns=['TREATMENT'])
    print(f"Loaded {len(df)} data points")
    
    # Prepare data by environment (PLOT NO)
    X_envs, Y_envs, plot_ids = prepare_data_by_environment(df)
    print(f"Data prepared for {len(plot_ids)} environments (plots)")
    
    # Lambda values to test
    lambda_values = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    # Analyze with both penalized and non-penalized approaches
    results = analyze_penalized_vs_non_penalized(X_envs, Y_envs, plot_ids, lambda_values)
    
    # Visualize results
    visualize_results(results, plot_ids)
    
    print("Analysis complete. Results saved to pfynforest/analyze_data/plots/")

if __name__ == "__main__":
    main()
