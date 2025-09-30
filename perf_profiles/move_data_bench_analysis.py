#!/usr/bin/env python3
"""
Move Data Benchmark Analysis
Converted from Jupyter notebook to Python script

This script analyzes move data benchmark results from JSON files and generates
visualizations for bandwidth vs ntasks and std/mean ratio plots.
"""

import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_benchmark_data(data_dir, file_pattern, bandwidth_keys):
    """
    Load benchmark data from JSON files matching the given pattern.
    
    Args:
        data_dir (str): Directory containing the benchmark files
        file_pattern (str): Pattern to match filenames (e.g., "PFS_to_SSD_")
        bandwidth_keys (list): List of bandwidth keys to extract from JSON records
        
    Returns:
        dict: Nested dictionary with structure {filesize: {node_count: {bandwidth_key: [values]}}}
    """
    filenames = [f for f in os.listdir(data_dir) if f.startswith(file_pattern) and f.endswith(".out")]
    data = {}
    
    for filename in filenames:
        match = re.search(r'_(\d+)n_', filename)
        if not match:
            continue
        node_count = int(match.group(1))
        
        with open(f"{data_dir}/{filename}", 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('{'):
                    record = json.loads(line)
                    filesize = record.get("filesize")
                    if not filesize:
                        continue
                    if filesize not in data:
                        data[filesize] = {}
                    if node_count not in data[filesize]:
                        data[filesize][node_count] = {"ntasks": []}
                        for key in bandwidth_keys:
                            data[filesize][node_count][key] = []
                    
                    for trial_key, entry in record.items():
                        if trial_key == "filesize":
                            continue
                        data[filesize][node_count]["ntasks"].append(entry["ntasks"])
                        for key in bandwidth_keys:
                            data[filesize][node_count][key].append(entry[key])
    
    return data


def calculate_statistics(data, bandwidth_keys):
    """
    Calculate averages and standard deviations for bandwidth data.
    
    Args:
        data (dict): Data dictionary from load_benchmark_data
        bandwidth_keys (list): List of bandwidth keys to calculate stats for
        
    Returns:
        tuple: (averages, std_devs) dictionaries with same structure as input data
    """
    averages = {}
    std_devs = {}
    
    for filesize, nodes in data.items():
        averages[filesize] = {}
        std_devs[filesize] = {}
        for node_count, metrics in nodes.items():
            averages[filesize][node_count] = {}
            std_devs[filesize][node_count] = {}
            
            for key in bandwidth_keys:
                values = metrics[key]
                averages[filesize][node_count][key] = np.mean(values) if values else 0
                std_devs[filesize][node_count][key] = np.std(values) if values else 0
    
    return averages, std_devs


def plot_bandwidth_with_error_bars(data, filesize, bandwidth_key, title_prefix, ylabel_prefix, plots_dir, show_std_ratio=False, target_nodes=None):
    """
    Create plots for bandwidth vs ntasks with error bars and optional std/mean ratio.
    
    Args:
        data (dict): Data dictionary from load_benchmark_data
        filesize (str): File size to plot (e.g., "1024MB")
        bandwidth_key (str): Key for bandwidth data (e.g., "shared-to-local")
        title_prefix (str): Prefix for plot titles
        ylabel_prefix (str): Prefix for y-axis labels
        plots_dir (str): Directory to save PDF files
        show_std_ratio (bool): Whether to show std/mean ratio plot (default: False)
    """
    if show_std_ratio:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(6, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(4, 3.5))
    
    # Filter nodes if target_nodes is specified
    available_nodes = sorted(data[filesize].keys())
    if target_nodes is not None:
        nodes_to_plot = [node for node in available_nodes if node in target_nodes]
        print(f"Available nodes: {available_nodes}, Plotting nodes: {nodes_to_plot}")
    else:
        nodes_to_plot = available_nodes
    
    # Plot bandwidth vs ntasks with error bars
    for node in nodes_to_plot:
        # Calculate mean and std for each unique ntask value
        unique_ntasks = sorted(set(data[filesize][node]["ntasks"]))
        means = []
        stds = []
        
        for ntask in unique_ntasks:
            indices = [i for i, x in enumerate(data[filesize][node]["ntasks"]) if x == ntask]
            values = [data[filesize][node][bandwidth_key][i] for i in indices]
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        # Plot with error bars
        ax1.errorbar(unique_ntasks, means, yerr=stds, 
                    label=f"{node}n", marker='o', capsize=5, capthick=2)
    # x-axis to log
    ax1.set_xscale('log')
    ax1.set_xlabel("Tasks Per Node (log)", fontsize=15)
    ax1.set_ylabel("Transfer Rate (MB/s)", fontsize=15)
    # Convert filesize from MB to GB for title
    filesize_mb = int(filesize.replace("MB", ""))
    filesize_gb = filesize_mb / 1024
    ax1.set_title(f"{title_prefix}({int(filesize_gb)}GB)", fontsize=15)
    
    # Set more x-axis tick labels for log scale
    import matplotlib.ticker as ticker
    
    # Get the data range to determine appropriate tick locations
    all_ntasks = []
    for node in nodes_to_plot:
        all_ntasks.extend(data[filesize][node]["ntasks"])
    
    if all_ntasks:
        min_val = min(all_ntasks)
        max_val = max(all_ntasks)
        
        # Create custom tick locations to ensure at least 5 ticks
        min_log = np.floor(np.log10(min_val))
        max_log = np.ceil(np.log10(max_val))
        
        # Generate tick locations at powers of 10 and intermediate values
        tick_locations = []
        for i in range(int(min_log), int(max_log) + 1):
            tick_locations.extend([10**i, 2*10**i, 5*10**i])
        
        # Filter to only include ticks within data range
        tick_locations = [t for t in tick_locations if min_val <= t <= max_val]
        
        # Ensure we have at least 5 ticks
        if len(tick_locations) < 5:
            # Add more intermediate values
            for i in range(int(min_log), int(max_log) + 1):
                tick_locations.extend([3*10**i, 4*10**i, 6*10**i, 7*10**i, 8*10**i, 9*10**i])
            tick_locations = [t for t in tick_locations if min_val <= t <= max_val]
            tick_locations = sorted(list(set(tick_locations)))
        
        ax1.set_xticks(tick_locations)
        ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Set tick label font sizes
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.tick_params(axis='both', which='minor', labelsize=12)
    
    # Create custom legend with only markers (no error bars)
    handles, labels = ax1.get_legend_handles_labels()
    # Create new handles with only markers
    new_handles = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, handle in enumerate(handles):
        # Get color from the errorbar container
        if hasattr(handle, 'get_color'):
            color = handle.get_color()
        else:
            # For ErrorbarContainer, get color from the line
            color = handle[0].get_color()
        
        # Create a new line2D object with only the marker
        new_handle = plt.Line2D([0], [0], marker='o', linestyle='None', 
                               color=color, markersize=8)
        new_handles.append(new_handle)
    if 'SSD' in title_prefix:
        ax1.legend(new_handles, labels, fontsize=14, framealpha=0.3, loc='lower left')
    else:
        ax1.legend(new_handles, labels, fontsize=14, framealpha=0.3, loc='upper left')
    
    ax1.grid(True)
    
    # Optional std/mean ratio plot
    if show_std_ratio:
        for node in nodes_to_plot:
            std_ratios = []
            unique_ntasks = sorted(set(data[filesize][node]["ntasks"]))
            for ntask in unique_ntasks:
                indices = [i for i, x in enumerate(data[filesize][node]["ntasks"]) if x == ntask]
                values = [data[filesize][node][bandwidth_key][i] for i in indices]
                std_ratio = (100 * np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0
                std_ratios.append(std_ratio)
            ax3.scatter(unique_ntasks, std_ratios, label=f"{node}n")
        
        ax3.set_xlabel("ntasks", fontsize=15)
        ax3.set_ylabel("Std / Mean Ratio (%)", fontsize=15)
        ax3.set_title(f"{title_prefix}\nBandwidth Std/Mean Ratio ({int(filesize_gb)}GB)", fontsize=15)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        if 'SSD' in title_prefix:
            ax3.legend(fontsize=15, framealpha=0.3, loc='lower left')
        else:
            ax3.legend(fontsize=15, framealpha=0.3, loc='upper left')
        ax3.grid(True)
        ax3.set_ylim(0, 200)
    
    plt.tight_layout()
    
    # Save plot as PDF
    safe_title = title_prefix.replace("-", "_").replace(" ", "_").lower()
    safe_filesize = filesize.replace("MB", "mb")
    safe_bandwidth = bandwidth_key.replace("-", "_").replace("(", "").replace(")", "").replace("/", "_").lower()
    filename = f"{safe_title}_{safe_bandwidth}_{safe_filesize}.pdf"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    
    plt.show()


def create_dataframe_rows(data, filesize, bandwidth_keys, operation, storage_types):
    """
    Create DataFrame rows for a specific filesize and bandwidth configuration.
    
    Args:
        data (dict): Data dictionary from load_benchmark_data
        filesize (str): File size (e.g., "1024MB")
        bandwidth_keys (list): List of bandwidth keys
        operation (str): Operation type (e.g., "cp", "scp")
        storage_types (list): List of storage type strings
        
    Returns:
        list: List of dictionaries representing DataFrame rows
    """
    rows = []
    filesize_mb = int(filesize.replace("MB", ""))
    
    for num_nodes, metrics in data[filesize].items():
        ntasks_list = metrics['ntasks']
        for i, bandwidth_key in enumerate(bandwidth_keys):
            bandwidth_list = metrics[bandwidth_key]
            for j in range(len(ntasks_list)):
                ntasks = ntasks_list[j]
                rows.append({
                    'operation': operation,
                    'randomOffset': 0,
                    'transferSize': 4096,
                    'aggregateFilesizeMB': filesize_mb,
                    'numTasks': ntasks,
                    'totalTime': -1,
                    'numNodes': num_nodes,
                    'tasksPerNode': ntasks // num_nodes if num_nodes > 0 else None,
                    'parallelism': ntasks,
                    'trMiB': bandwidth_list[j],
                    'storageType': storage_types[i]
                })
    
    return rows


def process_storage_configuration(data_dir, file_pattern, bandwidth_keys, 
                                title_prefixes, ylabel_prefixes, 
                                operation, storage_types, plots_dir, data_df=None, target_nodes=None):
    """
    Process a complete storage configuration (e.g., PFS_to_SSD, SSD_to_SSD).
    
    Args:
        data_dir (str): Directory containing benchmark files
        file_pattern (str): Pattern to match filenames
        bandwidth_keys (list): List of bandwidth keys to extract
        title_prefixes (list): List of title prefixes for plots
        ylabel_prefixes (list): List of ylabel prefixes for plots
        operation (str): Operation type for DataFrame
        storage_types (list): List of storage type strings
        plots_dir (str): Directory to save PDF plots
        data_df (pd.DataFrame, optional): Existing DataFrame to append to
        
    Returns:
        pd.DataFrame: Updated DataFrame with new data
    """
    # Load data
    data = load_benchmark_data(data_dir, file_pattern, bandwidth_keys)
    
    # Calculate statistics
    averages, std_devs = calculate_statistics(data, bandwidth_keys)
    
    # Create plots for each filesize
    for filesize in sorted(data.keys()):
        for i, bandwidth_key in enumerate(bandwidth_keys):
            plot_bandwidth_with_error_bars(
                data, filesize, bandwidth_key, 
                title_prefixes[i], ylabel_prefixes[i], plots_dir, show_std_ratio=False, target_nodes=target_nodes
            )
    
    # Create DataFrame rows
    all_rows = []
    for filesize in sorted(data.keys()):
        rows = create_dataframe_rows(data, filesize, bandwidth_keys, operation, storage_types)
        all_rows.extend(rows)
    
    # Create or update DataFrame
    IOR_PARAMS = [
        'operation', 'randomOffset', 'transferSize', 
        'aggregateFilesizeMB', 'numTasks', 'totalTime', 
        'numNodes', 'tasksPerNode', 'parallelism', 'trMiB', "storageType"
    ]
    
    new_df = pd.DataFrame(all_rows, columns=IOR_PARAMS)
    
    if data_df is not None:
        data_df = pd.concat([data_df, new_df], ignore_index=True)
    else:
        data_df = new_df
    
    return data_df


def main():
    """Main function to run the complete benchmark analysis."""
    data_dir = "./cp_data"
    plots_dir = "./move_data_plots"
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")
    
    # Initialize empty DataFrame
    data_df = None
    
    # Define target nodes to plot
    target_nodes = [2, 4, 16, 32]
    print(f"Target nodes for plotting: {target_nodes}")
    
    # Configuration 1: PFS to SSD
    print("Processing PFS to SSD configuration...")
    data_df = process_storage_configuration(
        data_dir=data_dir,
        file_pattern="PFS_to_SSD_",
        bandwidth_keys=["shared-to-local-bw(MB/s)", "local-to-share-bw(MB/s)"],
        title_prefixes=["BeeGFS-to-SSD", "SSD-to-BeeGFS"],
        ylabel_prefixes=["BeeGFS-to-SSD", "SSD-to-BeeGFS"],
        operation="cp",
        storage_types=["beegfs-ssd", "ssd-beegfs"],
        plots_dir=plots_dir,
        data_df=data_df,
        target_nodes=target_nodes
    )
    
    # Configuration 2: PFS to TMPFS
    print("Processing PFS to TMPFS configuration...")
    data_df = process_storage_configuration(
        data_dir=data_dir,
        file_pattern="PFS_to_tmpfs_",
        bandwidth_keys=["shared-to-local-bw(MB/s)", "local-to-share-bw(MB/s)"],
        title_prefixes=["BeeGFS-to-TMPFS", "TMPFS-to-BeeGFS"],
        ylabel_prefixes=["BeeGFS-to-TMPFS", "TMPFS-to-BeeGFS"],
        operation="cp",
        storage_types=["beegfs-tmpfs", "tmpfs-beegfs"],
        plots_dir=plots_dir,
        data_df=data_df,
        target_nodes=target_nodes
    )
    
    # Configuration 3: SSD to SSD
    print("Processing SSD to SSD configuration...")
    data_df = process_storage_configuration(
        data_dir=data_dir,
        file_pattern="SSD_to_SSD_",
        bandwidth_keys=["local-to-local-bw(MB/s)"],
        title_prefixes=["SSD-to-SSD"],
        ylabel_prefixes=["SSD-to-SSD"],
        operation="scp",
        storage_types=["ssd-ssd"],
        plots_dir=plots_dir,
        data_df=data_df,
        target_nodes=target_nodes
    )
    
    # Configuration 4: TMPFS to TMPFS
    print("Processing TMPFS to TMPFS configuration...")
    data_df = process_storage_configuration(
        data_dir=data_dir,
        file_pattern="tmpfs_to_tmpfs_",
        bandwidth_keys=["local-to-local-bw(MB/s)"],
        title_prefixes=["TMPFS-to-TMPFS"],
        ylabel_prefixes=["TMPFS-to-TMPFS"],
        operation="scp",
        storage_types=["tmpfs-tmpfs"],
        plots_dir=plots_dir,
        data_df=data_df,
        target_nodes=target_nodes
    )
    
    # Display results
    print(f"\nFinal DataFrame shape: {data_df.shape}")
    print("\nFirst 5 rows:")
    print(data_df.head(5))
    
    # Save to CSV
    output_file = "master_move_data.csv"
    data_df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    main()
