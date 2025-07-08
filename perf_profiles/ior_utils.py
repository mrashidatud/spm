import os
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Key Parameters for the master DataFrame
IOR_PARAMS = ['operation', 'randomOffset', 'transferSize', 
              'aggregateFilesizeMB', 'numTasks', 'totalTime', 
              'numNodes', 'tasksPerNode', 'parallelism', 'trMiB', 'storageType']

def extract_storage_info_from_path(file_path: str) -> Tuple[str, int]:
    """
    Extract storage type and number of nodes from file path.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (storage_type, num_nodes)
    """
    # Extract directory name from path
    dir_name = os.path.basename(os.path.dirname(file_path))
    
    # Extract storage type (before the first underscore)
    storage_type = dir_name.split('_')[0]
    
    # Extract number of nodes from suffix (e.g., "1n" -> 1)
    node_match = re.search(r'(\d+)n$', dir_name)
    num_nodes = int(node_match.group(1)) if node_match else 1
    
    return storage_type, num_nodes

def parse_ior_json_file(file_path: str) -> List[Dict]:
    """
    Parse a single IOR JSON file and extract relevant data.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing extracted data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    storage_type, num_nodes = extract_storage_info_from_path(file_path)
    
    # Extract file size from filename (e.g., "pior_64m_1gb_n8_1.json" -> 1GB)
    filename = os.path.basename(file_path)
    size_match = re.search(r'_(\d+)gb_', filename)
    aggregate_filesize_mb = int(size_match.group(1)) * 1024 if size_match else 0
    
    # Extract transfer size from filename (e.g., "pior_64m_1gb_n8_1.json" -> 64MB)
    transfer_match = re.search(r'_(\d+)m_', filename)
    transfer_size = int(transfer_match.group(1)) * 1024 * 1024 if transfer_match else 0
    
    results = []
    
    # Process summary data
    if 'summary' in data:
        for summary_entry in data['summary']:
            if summary_entry['operation'] in ['write', 'read']:
                result = {
                    'operation': summary_entry['operation'],
                    'randomOffset': 0,  # All benchmarks are sequential
                    'transferSize': summary_entry['transferSize'],
                    'aggregateFilesizeMB': aggregate_filesize_mb,
                    'numTasks': summary_entry['numTasks'],
                    'totalTime': summary_entry['MeanTime'],
                    'numNodes': num_nodes,
                    'tasksPerNode': summary_entry['tasksPerNode'],
                    'parallelism': summary_entry['numTasks'] * num_nodes,
                    'trMiB': summary_entry['bwMeanMIB'],
                    'storageType': storage_type
                }
                results.append(result)
    
    return results

def collect_ior_data(data_dir: str, storage_types: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Collect all IOR benchmark data from the specified directory.
    
    Args:
        data_dir: Directory containing IOR benchmark data
        storage_types: List of storage types to include (e.g., ['beegfs', 'localssd'])
                      If None, includes all storage types
    
    Returns:
        DataFrame containing all collected IOR benchmark data
    """
    all_data = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_dir):
        # Filter directories based on storage types if specified
        if storage_types:
            dirs[:] = [d for d in dirs if any(storage_type in d for storage_type in storage_types)]
        
        # Process JSON files
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                file_data = parse_ior_json_file(file_path)
                all_data.extend(file_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure all required columns are present
    for col in IOR_PARAMS:
        if col not in df.columns:
            df[col] = 0
    
    return df

def clean_data_by_throughput(df: pd.DataFrame, min_throughput: float = 0.0) -> pd.DataFrame:
    """
    Clean the DataFrame by removing rows where throughput is below a minimum threshold.
    
    Args:
        df: Input DataFrame
        min_throughput: Minimum throughput threshold (default: 0.0 to remove zero throughput)
        
    Returns:
        Cleaned DataFrame
    """
    original_count = len(df)
    cleaned_df = df[df['trMiB'] > min_throughput].copy()
    removed_count = original_count - len(cleaned_df)
    
    print(f"Data cleaning summary:")
    print(f"  Original records: {original_count}")
    print(f"  Cleaned records: {len(cleaned_df)}")
    print(f"  Removed records: {removed_count} (trMiB <= {min_throughput})")
    
    return cleaned_df

def calculate_averaged_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate averaged statistics for each storage type and create new rows with averaged values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with original data plus averaged statistics rows
    """
    print("Calculating averaged statistics for each storage type...")
    
    # List to store averaged rows
    averaged_rows = []
    
    # Group by storage type and calculate averages
    for storage_type in df['storageType'].unique():
        if storage_type.startswith('ave_'):  # Skip already averaged data
            continue
            
        storage_data = df[df['storageType'] == storage_type]
        
        # Group by all relevant parameters except storageType
        group_cols = ['operation', 'randomOffset', 'transferSize', 'aggregateFilesizeMB', 
                     'numTasks', 'numNodes', 'tasksPerNode', 'parallelism']
        
        # Calculate averages for each unique combination
        grouped = storage_data.groupby(group_cols).agg({
            'trMiB': 'mean',
            'totalTime': 'mean'
        }).reset_index()
        
        # Create new rows with averaged values
        for _, row in grouped.iterrows():
            averaged_row = row.copy()
            averaged_row['storageType'] = f'ave_{storage_type}'
            averaged_rows.append(averaged_row)
    
    # Create DataFrame from averaged rows
    if averaged_rows:
        averaged_df = pd.DataFrame(averaged_rows)
        
        # Combine original data with averaged data
        combined_df = pd.concat([df, averaged_df], ignore_index=True)
        
        print(f"Added {len(averaged_rows)} averaged statistics rows")
        print(f"New storage types: {[col for col in combined_df['storageType'].unique() if col.startswith('ave_')]}")
        
        return combined_df
    else:
        print("No averaged statistics to add")
        return df

def save_master_ior_df(df: pd.DataFrame, output_file: str = 'master_ior_df.csv'):
    """
    Save the master IOR DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        output_file: Output CSV filename
    """
    df.to_csv(output_file, index=False)
    print(f"Master IOR DataFrame saved to {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

def load_master_ior_df(input_file: str = 'master_ior_df.csv') -> pd.DataFrame:
    """
    Load the master IOR DataFrame from CSV file.
    
    Args:
        input_file: Input CSV filename
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(input_file)

def export_storage_data(df: pd.DataFrame, clean_data: bool = True, min_throughput: float = 0.0):
    """
    Export data for each storage type to separate CSV files.
    
    Args:
        df: Input DataFrame
        clean_data: Whether to remove rows with zero/low throughput
        min_throughput: Minimum throughput threshold for cleaning
    """
    print("Exporting individual storage type data...")
    
    for storage_type in df['storageType'].unique():
        storage_df = filter_data_by_conditions(df, storage_type=storage_type)
        
        if clean_data:
            storage_df = clean_data_by_throughput(storage_df, min_throughput)
        
        filename = f'ior_data_{storage_type}.csv'
        storage_df.to_csv(filename, index=False)
        print(f"  Exported {storage_type} data to {filename} ({len(storage_df)} records)")

def filter_data_by_conditions(df: pd.DataFrame, 
                            storage_type: Optional[str] = None,
                            transfer_size: Optional[int] = None,
                            num_nodes: Optional[int] = None,
                            operation: Optional[str] = None) -> pd.DataFrame:
    """
    Filter the DataFrame based on specified conditions.
    
    Args:
        df: Input DataFrame
        storage_type: Filter by storage type
        transfer_size: Filter by transfer size (in bytes)
        num_nodes: Filter by number of nodes
        operation: Filter by operation type ('write' or 'read')
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if storage_type:
        filtered_df = filtered_df[filtered_df['storageType'] == storage_type]
    
    if transfer_size is not None:
        filtered_df = filtered_df[filtered_df['transferSize'] == transfer_size]
    
    if num_nodes is not None:
        filtered_df = filtered_df[filtered_df['numNodes'] == num_nodes]
    
    if operation:
        filtered_df = filtered_df[filtered_df['operation'] == operation]
    
    return filtered_df

def plot_storage_comparison(df: pd.DataFrame, 
                          transfer_size: int,
                          num_nodes: int,
                          metric: str = 'trMiB',
                          title: str = None,
                          save_path: Optional[str] = None,
                          include_averaged: bool = True):
    """
    Plot comparison of different storage types for fixed transfer size and number of nodes.
    
    Args:
        df: Input DataFrame
        transfer_size: Transfer size to filter by (in bytes)
        num_nodes: Number of nodes to filter by
        metric: Metric to plot (default: 'trMiB')
        title: Plot title
        save_path: Path to save the plot
        include_averaged: Whether to include averaged statistics
    """
    # Filter data
    filtered_df = filter_data_by_conditions(
        df, transfer_size=transfer_size, num_nodes=num_nodes
    )
    
    if filtered_df.empty:
        print(f"No data found for transfer_size={transfer_size}, num_nodes={num_nodes}")
        return
    
    # Filter out averaged data if not requested
    if not include_averaged:
        filtered_df = filtered_df[~filtered_df['storageType'].str.startswith('ave_')]
    
    if filtered_df.empty:
        print(f"No data found after filtering averaged data")
        return
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Group by storage type and operation
    for storage_type in filtered_df['storageType'].unique():
        storage_data = filtered_df[filtered_df['storageType'] == storage_type]
        
        for operation in ['write', 'read']:
            op_data = storage_data[storage_data['operation'] == operation]
            if not op_data.empty:
                # Use different markers for averaged data
                marker = '^' if storage_type.startswith('ave_') else ('o' if operation == 'write' else 's')
                label = f'{storage_type} {operation}'
                plt.scatter(op_data['numTasks'], op_data[metric], 
                           label=label, 
                           marker=marker,
                           s=100)
    
    plt.xlabel('Number of Tasks')
    plt.ylabel(f'{metric} (MiB/s)')
    plt.title(title or f'Storage Comparison - Transfer Size: {transfer_size//(1024*1024)}MB, Nodes: {num_nodes}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_transfer_size_analysis(df: pd.DataFrame,
                              storage_type: str,
                              num_nodes: int,
                              metric: str = 'trMiB',
                              title: str = None,
                              save_path: Optional[str] = None):
    """
    Plot analysis of different transfer sizes for a specific storage type and number of nodes.
    
    Args:
        df: Input DataFrame
        storage_type: Storage type to filter by
        num_nodes: Number of nodes to filter by
        metric: Metric to plot (default: 'trMiB')
        title: Plot title
        save_path: Path to save the plot
    """
    # Filter data
    filtered_df = filter_data_by_conditions(
        df, storage_type=storage_type, num_nodes=num_nodes
    )
    
    if filtered_df.empty:
        print(f"No data found for storage_type={storage_type}, num_nodes={num_nodes}")
        return
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Group by transfer size and operation
    for operation in ['write', 'read']:
        op_data = filtered_df[filtered_df['operation'] == operation]
        if not op_data.empty:
            # Convert transfer size to MB for x-axis
            transfer_sizes_mb = op_data['transferSize'] / (1024 * 1024)
            plt.scatter(transfer_sizes_mb, op_data[metric], 
                       label=f'{operation}', 
                       marker='o' if operation == 'write' else 's',
                       s=100)
    
    plt.xlabel('Transfer Size (MB)')
    plt.ylabel(f'{metric} (MiB/s)')
    plt.title(title or f'Transfer Size Analysis - {storage_type}, Nodes: {num_nodes}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    if save_path:
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for the IOR benchmark data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with summary statistics
    """
    summary_stats = df.groupby(['storageType', 'operation', 'numNodes']).agg({
        'trMiB': ['mean', 'std', 'min', 'max'],
        'totalTime': ['mean', 'std', 'min', 'max'],
        'transferSize': 'first',
        'numTasks': 'first'
    }).round(2)
    
    return summary_stats

def print_data_overview(df: pd.DataFrame):
    """
    Print an overview of the collected data.
    
    Args:
        df: Input DataFrame
    """
    print("=== IOR Benchmark Data Overview ===")
    print(f"Total records: {len(df)}")
    print(f"Storage types: {sorted(df['storageType'].unique())}")
    print(f"Number of nodes: {sorted(df['numNodes'].unique())}")
    print(f"Transfer sizes (MB): {sorted(df['transferSize'].unique() // (1024*1024))}")
    print(f"Operations: {sorted(df['operation'].unique())}")
    print(f"Number of tasks range: {df['numTasks'].min()} - {df['numTasks'].max()}")
    print(f"Throughput range (MiB/s): {df['trMiB'].min():.2f} - {df['trMiB'].max():.2f}")
    print()

if __name__ == "__main__":
    # Example usage
    data_dir = "ior_data"
    
    # Collect all data
    print("Collecting IOR benchmark data...")
    df = collect_ior_data(data_dir)
    
    # Print overview
    print_data_overview(df)
    
    # Clean data
    df = clean_data_by_throughput(df, min_throughput=0.0)
    
    # Calculate averaged statistics
    df = calculate_averaged_statistics(df)
    
    # Save to CSV
    save_master_ior_df(df)
    
    # Export individual storage data (cleaned)
    export_storage_data(df, clean_data=True)
    
    # Example: Filter for specific storage types
    print("\nFiltering for beegfs and localssd only...")
    filtered_df = collect_ior_data(data_dir, storage_types=['beegfs', 'localssd'])
    save_master_ior_df(filtered_df, 'master_ior_df_filtered.csv')