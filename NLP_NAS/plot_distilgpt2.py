import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Ensure the plots directory exists
plot_dir = 'results/distilgpt2/plots'
os.makedirs(plot_dir, exist_ok=True)

# Function to read raw data into a DataFrame and tag with model name
def read_raw_data(file_path, model_name):
    data = {}
    cpu_freqs = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            key_value = line.strip().split(": ")
            if len(key_value) == 2:
                key, value = key_value
                if key.startswith('CPU') and '_current_freq_delta' in key:
                    cpu_core = key.split('CPU')[1].split('_')[0]
                    cpu_freqs[f'CPU{cpu_core}_current_freq_delta'] = float(value)
                else:
                    data[key] = float(value)
    df = pd.DataFrame([data])
    for key, value in cpu_freqs.items():
        df[key] = value
    df['model'] = model_name  # Add the model name as a column
    return df

# Load the raw data for all models
def load_data_for_all_models(model_names, base_dir):
    all_data = []
    for model_name in model_names:
        raw_data_path = os.path.join(base_dir, model_name, f'{model_name}_raw.txt')
        df = read_raw_data(raw_data_path, model_name)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Plot with shaded confidence interval (line plot with variance)
def plot_with_confidence(df, metric, title, ylabel, plot_filename):
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        plt.plot(model_df.index, model_df[metric], label=model, marker='o')
        plt.fill_between(model_df.index, model_df[metric] - model_df[metric].std(), 
                         model_df[metric] + model_df[metric].std(), alpha=0.2)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, plot_filename))  # Save the plot
    plt.show()  # Show the plot

# Box plot to show the distribution of values across models
def plot_boxplot(df, metric, title, ylabel, plot_filename):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model', y=metric, data=df)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, plot_filename))  # Save the plot
    plt.show()  # Show the plot

# Heatmap for model vs metric performance
def plot_heatmap(df, title, plot_filename):
    metrics = ['Power_avg_power_mW_delta',
               'Memory_used_KB_delta', 'GPU_load_delta', 'CPU_avg_freq_delta']
    pivot_df = df.groupby('model')[metrics].mean()
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", linewidths=.5)
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, plot_filename))  # Save the plot
    plt.show()  # Show the plot

# Correlation matrix of metrics
def plot_correlation_matrix(df, plot_filename):
    metrics = ['Power_avg_power_mW_delta',
               'Memory_used_KB_delta', 'GPU_load_delta', 'CPU_avg_freq_delta']
    corr_matrix = df[metrics].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=.5)
    plt.title('Correlation Matrix of Metrics')
    plt.savefig(os.path.join(plot_dir, plot_filename))  # Save the plot
    plt.show()  # Show the plot

# Calculate peak, average, and total metrics
def calculate_performance_metrics(df):
    summary = df.groupby('model').agg({
        'Power_avg_power_mW_delta': ['mean', 'max'],
        'Memory_used_KB_delta': ['mean', 'max'],
        'GPU_load_delta': ['mean', 'max'],
        'CPU_avg_freq_delta': ['mean', 'max']
    })
    print("Performance Metrics Summary:")
    print(summary)

# Main function to load data and plot comparisons
def compare_models(model_names, base_dir):
    # Load data from all models
    df = load_data_for_all_models(model_names, base_dir)
    
    # Compute average CPU frequency delta
    cpu_freq_cols = [col for col in df.columns if 'CPU' in col and '_current_freq_delta' in col]
    df['CPU_avg_freq_delta'] = df[cpu_freq_cols].mean(axis=1)
    
    # Plot line charts with confidence intervals
    plot_with_confidence(df, 'Power_avg_power_mW_delta', 'Average Power Consumption with Variance', 'Power (mW)', 'power_consumption_variance.png')
    plot_with_confidence(df, 'Memory_used_KB_delta', 'Memory Usage with Variance', 'Memory Used (KB)', 'memory_usage_variance.png')
    plot_with_confidence(df, 'GPU_load_delta', 'GPU Load with Variance', 'GPU Load', 'gpu_load_variance.png')
    plot_with_confidence(df, 'CPU_avg_freq_delta', 'Average CPU Frequency Delta with Variance', 'CPU Frequency Delta (kHz)', 'cpu_freq_variance.png')

    # Boxplots to visualize distribution across models
    plot_boxplot(df, 'Power_avg_power_mW_delta', 'Distribution of Average Power by Model', 'Power (mW)', 'boxplot_power.png')
    plot_boxplot(df, 'Memory_used_KB_delta', 'Distribution of Memory Used by Model', 'Memory Used (KB)', 'boxplot_memory.png')
    plot_boxplot(df, 'CPU_avg_freq_delta', 'Distribution of Average CPU Frequency Delta by Model', 'CPU Frequency Delta (kHz)', 'boxplot_cpu.png')
    
    # Heatmap of model vs performance metrics
    plot_heatmap(df, 'Model vs Performance Metrics', 'model_performance_heatmap.png')

    # Correlation matrix for metrics
    plot_correlation_matrix(df, 'correlation_matrix.png')

    # Calculate peak, average, and total performance metrics
    calculate_performance_metrics(df)

# Plot total time comparison across models
def plot_total_time(df, plot_filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='Time_delta', data=df)
    plt.title('Total Time Comparison Across Models')
    plt.ylabel('Total Time (seconds)')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, plot_filename))  # Save the plot
    plt.show()  # Show the plot

# Plot time vs metric (e.g., Power, GPU Load)
def plot_time_vs_metric(df, metric, title, ylabel, plot_filename):
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        plt.plot(model_df['Time_delta'], model_df[metric], label=model, marker='o')
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, plot_filename))  # Save the plot
    plt.show()  # Show the plot

# Plot time vs CPU/GPU load or other relevant metrics
def plot_time_vs_load(df, plot_filename):
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        plt.plot(model_df['Time_delta'], model_df['GPU_load_delta'], label=f'{model} GPU Load', marker='o')
        plt.plot(model_df['Time_delta'], model_df['CPU_avg_freq_delta'], label=f'{model} CPU Freq', marker='x')
    plt.title('Time vs CPU/GPU Load')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Load / Frequency (kHz)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, plot_filename))  # Save the plot
    plt.show()  # Show the plot

# Main function to load data and plot comparisons, including time-related plots
def compare_models_with_time(model_names, base_dir):
    # Load data from all models
    df = load_data_for_all_models(model_names, base_dir)
    
    # Compute average CPU frequency delta
    cpu_freq_cols = [col for col in df.columns if 'CPU' in col and '_current_freq_delta' in col]
    df['CPU_avg_freq_delta'] = df[cpu_freq_cols].mean(axis=1)
    
    # Plot total time comparison across models
    plot_total_time(df, 'total_time_comparison.png')
    
    # Plot time vs power consumption
    plot_time_vs_metric(df, 'Power_avg_power_mW_delta', 'Time vs Power Consumption', 'Power (mW)', 'time_vs_power.png')
    
    # Plot time vs memory usage
    plot_time_vs_metric(df, 'Memory_used_KB_delta', 'Time vs Memory Usage', 'Memory Used (KB)', 'time_vs_memory.png')
    
    # Plot time vs CPU and GPU load
    plot_time_vs_load(df, 'time_vs_cpu_gpu_load.png')


# Define the models and paths
model_names = ['distilgpt2_3epochs', 'distilgpt2_5epochs', 'distilgpt2_10epochs', 'distilgpt2_12epochs', 'distilgpt2_15epochs']
raw_data_dir = 'stat_dumps/distilgpt2'  # Update with your actual base directory

# Run comparison
compare_models(model_names, raw_data_dir)

# Run comparison with time-related plots
compare_models_with_time(model_names, raw_data_dir)
