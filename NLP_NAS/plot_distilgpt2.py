import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

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

# Define the function to normalize and plot all metrics in a single grouped bar plot
def plot_normalized_grouped_bars(df, metrics, plot_filename):
    # Select the necessary columns (model and the metrics of interest)
    df_grouped = df[['model'] + metrics]
    
    # Normalize each metric to a 0-1 range
    scaler = MinMaxScaler()
    df_grouped[metrics] = scaler.fit_transform(df_grouped[metrics])

    # Melt the DataFrame to have a long format suitable for a grouped bar plot
    df_melted = df_grouped.melt(id_vars='model', var_name='Metric', value_name='Normalized Value')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='Normalized Value', hue='Metric', data=df_melted, palette='Set2')
    
    plt.title('Normalized Comparison of Models Across Metrics')
    plt.xlabel('Model')
    plt.ylabel('Normalized Metric Value (0-1)')
    plt.grid(True, axis='y', linestyle='--')
    
    # Save and show the plot
    plt.savefig(os.path.join(plot_dir, plot_filename))
    plt.show()

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
    
    plt.title('Correlation Matrix of Metrics', pad=20)  # Add padding for title
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()  # Ensure nothing is cut off
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

# Radar chart
def plot_radar_chart(df, plot_filename):
    metrics = ['Power_avg_power_mW_delta', 'Memory_used_KB_delta', 'GPU_load_delta', 'CPU_avg_freq_delta']
    df_mean = df.groupby('model')[metrics].mean().reset_index()

    # Normalize data for better visualization
    scaler = MinMaxScaler()
    df_mean[metrics] = scaler.fit_transform(df_mean[metrics])

    # Radar chart setup
    categories = metrics
    num_vars = len(categories)

    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Adjust radar plot
    for i, row in df_mean.iterrows():
        values = row[metrics].tolist()
        values += values[:1]  # Complete the loop
        ax.fill(angles, values, alpha=0.25)
        ax.plot(angles, values, label=row['model'], marker='o')

    # Customize ticks and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Adjust label rotation based on angle
    for label, angle in zip(ax.get_xticklabels(), angles):
        angle_deg = np.degrees(angle)
        if angle_deg < 90 or angle_deg > 270:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

        label.set_rotation(angle_deg - 90 if angle_deg < 180 else angle_deg + 90)

    # Move the legend outside the plot
    legend = plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.05))
    plt.title('Radar Chart: Model Comparison')

    # Save the plot with a tight layout to prevent cropping
    plt.savefig(os.path.join(plot_dir, plot_filename), bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()

# Parallel coordinates plot
def plot_parallel_coordinates(df, plot_filename):
    metrics = ['Power_avg_power_mW_delta', 'Memory_used_KB_delta', 'GPU_load_delta', 'CPU_avg_freq_delta']
    df_mean = df.groupby('model')[metrics].mean().reset_index()

    # Normalize data for better visualization
    scaler = MinMaxScaler()
    df_mean[metrics] = scaler.fit_transform(df_mean[metrics])

    plt.figure(figsize=(10, 6))
    # Create the parallel coordinates plot
    parallel_coordinates(df_mean, 'model', color=sns.color_palette("husl", len(df_mean['model'].unique())))

    # Set the title
    plt.title('Parallel Coordinates: Model Comparison')

    # Move the legend outside the plot and place it at the right
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)

    # Save the plot
    plt.savefig(os.path.join(plot_dir, plot_filename), bbox_inches='tight')
    plt.show()

# Additional plotting functions (time-related)
def plot_total_time(df, plot_filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='Time_delta', hue='model', data=df, palette=palette, dodge=False)
    plt.title('Total Time Comparison Across Models')
    plt.ylabel('Total Time (seconds)')
    plt.grid(True, axis='y', linestyle='--')
    plt.savefig(os.path.join(plot_dir, plot_filename))  # Save the plot
    plt.show()  # Show the plot

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
def compare_models_with_all_plots(model_names, base_dir):
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

    # Define metrics, titles, and ylabels for the combined bar plots
    metrics = [
        'Power_avg_power_mW_delta',
        'Memory_used_KB_delta',
        'GPU_load_delta',
        'CPU_avg_freq_delta'
    ]
    titles = [
        'Average Power Consumption with Variance',
        'Memory Usage with Variance',
        'GPU Load with Variance',
        'Average CPU Frequency Delta with Variance'
    ]
    ylabels = [
        'Power (mW)',
        'Memory Used (KB)',
        'GPU Load',
        'CPU Frequency Delta (kHz)'
    ]

    # Bar plots
    plot_normalized_grouped_bars(df, metrics, 'normalized_grouped_bar_plot.png')
   
    # Heatmap of model vs performance metrics
    plot_heatmap(df, 'Model vs Performance Metrics', 'model_performance_heatmap.png')

    # Correlation matrix for metrics
    plot_correlation_matrix(df, 'correlation_matrix.png')

    # Plot radar chart for model comparison
    plot_radar_chart(df, 'radar_chart_comparison.png')

    # Plot parallel coordinates plot
    plot_parallel_coordinates(df, 'parallel_coordinates_comparison.png')

    # Calculate and print performance metrics
    calculate_performance_metrics(df)

# Define the models and paths
model_names = ['distilgpt2_3epochs', 'distilgpt2_5epochs', 'distilgpt2_10epochs', 'distilgpt2_12epochs', 'distilgpt2_15epochs']
raw_data_dir = 'stat_dumps/distilgpt2'

# Set consistent color palette for models
palette = sns.color_palette("Set2", len(model_names))

# Run comparison with all plots
compare_models_with_all_plots(model_names, raw_data_dir)
