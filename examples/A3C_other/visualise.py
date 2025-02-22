import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

name = "exp"
# both inclusive
nums = [33,34,35]
# num_start, num_end = 1,3

def load_experiment_data(name, nums):
    """
    Load all experiments from a given directory.
    For each CSV file found, attempt to load a matching TXT file (same basename) containing parameters.
    
    Returns:
        A list of dictionaries with keys:
            - 'name': experiment name (from the file basename)
            - 'data': a DataFrame from the CSV file
            - 'parameters': a dict of parameter names/values from the TXT file
    """
    experiments = []
    # Look for all CSV files in the directory
    for i in nums:
        filename = f"{name}_{i}"
        text_file = f"{filename}.txt"
        csv_file = f"{filename}.csv"
        
        # Read the CSV file into a DataFrame.
        df = pd.read_csv(csv_file)
        
        # Initialize an empty dictionary for parameters.
        parameters = {}
        if os.path.exists(text_file):
            with open(text_file, 'r') as file:
                # Assume each line is in the format: "parameter_name: parameter_value"
                for line in file:
                    param, val = line.split("=", 1)
                    parameters[param] = val
                    # if '=' in line:
                    #     key, value = line.split(':', 1)
                    #     parameters[key.strip()] = value.strip()
        
        experiments.append({
            'name': filename,
            'results': df,
            'parameters': parameters
        })
    
    return experiments

def plot_training_curves(experiments, metric_name="Mean Reward Against Episodes"):
    """
    Plot training curves for a given metric (by default "Mean Reward Against Episodes")
    for each experiment.
    """
    plt.figure(figsize=(10, 6))
    params_df = pd.DataFrame(e["parameters"] for e in experiments)
    for exp in experiments:
        df = exp['results']
        # Filter the DataFrame for rows matching the metric of interest.
        df_metric = df[df['metric'] == metric_name]
        df_metric = df_metric.sort_values(by='step')
        x = df_metric['step'].values
        y = df_metric['value'].values

        # cubic_interpolation_model = interp1d(x, y, kind = "cubic")
        # X_=np.linspace(x.min(), x.max(), 500)
        # spline = make_interp_spline(x, y, k=3)
        # Y_= spline(X_)
        sigma = 5 # Adjust for more/less smoothing
        y_smoothed = gaussian_filter1d(y, sigma=sigma)
        X_ = x
        Y_ = y_smoothed
        # X_ = x
        
        # Y_ = y

        # if df_metric.empty:
        #     continue
        # Plot the reward vs. step.
        plt.plot(X_, Y_, 
                 linewidth=2, 
                 label=f"{exp['name']}")
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None

    print(params_df)
    plt.xlabel("Step (Episode)")
    plt.ylabel("Mean Reward")
    plt.title(f"Results for experiments {name}")
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.show()

def aggregate_final_rewards(experiments, metric_name="Mean Reward Against Episodes"):
    """
    For each experiment, get a summary by extracting the final reward for the chosen metric,
    and merge that with the parameter values.
    
    Returns:
        A summary DataFrame where each row corresponds to one experiment.
    """
    summaries = []
    for exp in experiments:
        df = exp['data']
        df_metric = df[df['metric'] == metric_name]
        if not df_metric.empty:
            # Get the final reward (last row) for this metric.
            final_reward = df_metric.iloc[-1]['value']
        else:
            final_reward = None
        
        # Build a summary dict that includes the experiment name, final reward,
        # and all parameter key/values.
        summary = {'name': exp['name'], 'final_reward': final_reward}
        summary.update(exp['parameters'])
        summaries.append(summary)
    return pd.DataFrame(summaries)

def plot_parameter_vs_reward(summary_df, parameter_name):
    """
    Create a scatter plot of final reward vs. a selected parameter.
    This function attempts to convert the parameter column to numeric.
    """
    # Convert the parameter values to numeric if possible.
    summary_df[parameter_name] = pd.to_numeric(summary_df[parameter_name], errors='coerce')
    plt.figure(figsize=(8, 6))
    plt.scatter(summary_df[parameter_name], summary_df['final_reward'], s=100)
    plt.xlabel(parameter_name)
    plt.ylabel("Final Reward")
    plt.title(f"Final Reward vs {parameter_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set the directory containing your experiment CSV and TXT files.
    os.chdir("./results")
    experiments = load_experiment_data(name, nums)
    
    # Plot the training curves (reward vs. step)
    plot_training_curves(experiments, metric_name="Mean Reward Against Episodes")
    
    # Aggregate the final rewards and parameters into a summary DataFrame.
    # summary_df = aggregate_final_rewards(experiments, metric_name="Mean Reward Against Episodes")
    # print("Experiment Summary:")
    # print(summary_df)
    
    # # If you want to compare a specific parameter (e.g., "learning_rate") against the final reward,
    # # ensure that parameter exists in your summary and then call:
    # if 'learning_rate' in summary_df.columns:
    #     plot_parameter_vs_reward(summary_df, "learning_rate")
    # else:
    #     print("Parameter 'learning_rate' not found in the summary. Please check your parameter names.")
