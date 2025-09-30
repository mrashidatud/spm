#!/usr/bin/env python3
"""
IOR Benchmark Analysis Script

This script provides analysis and visualization of IOR benchmark data collected from different storage systems.

Features:
- Collect and organize IOR benchmark data from JSON files
- Filter data by storage type, transfer size, number of nodes
- Generate comparison plots and analysis
- Export data to CSV format
- All plots are saved to the 'plot' directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import argparse
from typing import List, Optional

# Import our utility functions
from ior_utils import *

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_plotting():
    """Set up plotting style and create plot directory."""
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create plot directory if it doesn't exist
    plot_dir = "ior_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created plot directory: {plot_dir}")
    else:
        print(f"Plot directory already exists: {plot_dir}")
    
    return plot_dir

def collect_and_process_data(data_dir: str = "ior_data", csv_file: str = "updated_master_ior_df.csv") -> pd.DataFrame:
    """Collect and process IOR benchmark data."""
    print("Collecting IOR benchmark data...")
    
    # Check if data directory exists, if not try to load from existing CSV
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Checking for existing CSV files...")
        
        # Try to load from specified CSV file if it exists
        if os.path.exists(csv_file):
            print(f"Loading data from {csv_file}...")
            df = pd.read_csv(csv_file)
            # Clean up the data by removing rows with NaN values in critical columns
            df = df.dropna(subset=['storageType', 'operation', 'trMiB'])
            print(f"Loaded {len(df)} records from {csv_file}")
        else:
            print("No data directory or CSV file found. Creating empty DataFrame.")
            df = pd.DataFrame()
    else:
        df = collect_ior_data(data_dir)
    
    # Check if DataFrame is empty
    if df.empty:
        print("Warning: No data available for analysis.")
        return df
    
    # Display overview
    print_data_overview(df)
    
    # Show first few rows
    print("First few rows of the dataset:")
    print(df.head())
    
    # Remove rows where trMiB is 0
    df = df[df['trMiB'] > 0]
    
    # Save the master DataFrame
    save_master_ior_df(df, 'master_ior_df.csv')
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

def storage_comparison_analysis(df: pd.DataFrame, plot_dir: str):
    """Perform storage comparison analysis."""
    print("\n=== Storage Comparison Analysis ===")
    
    # Check if DataFrame is empty
    if df.empty:
        print("Warning: DataFrame is empty. Skipping storage comparison analysis.")
        return
    
    # Example: Compare different storage types for fixed transfer size, number of nodes, and aggregate file size
    transfer_size_mb = 1  # 1MB transfer size
    transfer_size_bytes = transfer_size_mb * 1024 * 1024
    num_nodes = 8
    aggregate_file_size_mb = 1024  # Set this to the file size you want to filter by
    
    print(f"Comparing storage types for {transfer_size_mb}MB transfer size, {num_nodes} node(s), and {aggregate_file_size_mb}MB aggregate file size")
    
    # Create the comparison plot
    plot_storage_comparison(
        df,
        transfer_size=transfer_size_bytes,
        num_nodes=num_nodes,
        title=f'Storage Performance Comparison - {transfer_size_mb}MB Transfer, {num_nodes} Node(s), {aggregate_file_size_mb}MB File',
        save_path=os.path.join(plot_dir, f'storage_comparison_{transfer_size_mb}mb_{num_nodes}node_{aggregate_file_size_mb}mbfile.pdf'),
        aggregate_file_size=aggregate_file_size_mb
    )
    
    # Compare for different transfer sizes: 4KB 1MB, 4MB, 64MB, 1GB
    transfer_sizes_mb = [0.004, 1, 4, 64, 1024]
    num_nodes = 1
    
    fig, axes = plt.subplots(1, len(transfer_sizes_mb), figsize=(18, 6))
    
    for i, transfer_size_mb in enumerate(transfer_sizes_mb):
        transfer_size_bytes = transfer_size_mb * 1024 * 1024
        
        # Filter data
        filtered_df = filter_data_by_conditions(
            df, transfer_size=transfer_size_bytes, num_nodes=num_nodes
        )
        
        if not filtered_df.empty:
            # Create subplot
            ax = axes[i]
            
            for storage_type in filtered_df['storageType'].unique():
                storage_data = filtered_df[filtered_df['storageType'] == storage_type]
                
                for operation in ['write', 'read']:
                    op_data = storage_data[storage_data['operation'] == operation]
                    if not op_data.empty:
                        ax.scatter(op_data['numTasks'], op_data['trMiB'], 
                                   label=f'{storage_type} {operation}', 
                                   marker='o' if operation == 'write' else 's',
                                   s=100)
            
            ax.set_xlabel('Number of Tasks')
            ax.set_ylabel('Throughput (MiB/s)')
            ax.set_title(f'{transfer_size_mb}MB Transfer Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, f'No data for {transfer_size_mb}MB', 
                         ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'storage_comparison_multiple_transfer_sizes.pdf'), bbox_inches='tight')
    plt.show()

def transfer_size_analysis(df: pd.DataFrame, plot_dir: str):
    """Perform transfer size analysis."""
    print("\n=== Transfer Size Analysis ===")
    
    # Check if DataFrame is empty
    if df.empty:
        print("Warning: DataFrame is empty. Skipping transfer size analysis.")
        return
    
    # Analyze transfer size impact for different storage types
    storage_types = ['beegfs', 'ssd', 'nfs', 'tmpfs']
    num_nodes = 1
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, storage_type in enumerate(storage_types):
        if i < len(axes):
            # Filter data
            filtered_df = filter_data_by_conditions(
                df, storage_type=storage_type, num_nodes=num_nodes
            )
            
            if not filtered_df.empty:
                ax = axes[i]
                
                for operation in ['write', 'read']:
                    op_data = filtered_df[filtered_df['operation'] == operation]
                    if not op_data.empty:
                        # Convert transfer size to MB for x-axis
                        transfer_sizes_mb = op_data['transferSize'] / (1024 * 1024)
                        ax.scatter(transfer_sizes_mb, op_data['trMiB'], 
                                   label=f'{operation}', 
                                   marker='o' if operation == 'write' else 's',
                                   s=100)
                
                ax.set_xlabel('Transfer Size (MB)')
                ax.set_ylabel('Throughput (MiB/s)')
                ax.set_title(f'{storage_type.upper()} - {num_nodes} Node(s)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
            else:
                axes[i].text(0.5, 0.5, f'No data for {storage_type}', 
                             ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'transfer_size_analysis_all_storage.pdf'), bbox_inches='tight')
    plt.show()
    
    # Detailed analysis for a specific storage type
    storage_type = 'beegfs'  # Change this to analyze different storage types
    num_nodes = 1
    
    print(f"Detailed transfer size analysis for {storage_type.upper()} with {num_nodes} node(s)")
    
    plot_transfer_size_analysis(df, 
                               storage_type=storage_type, 
                               num_nodes=num_nodes,
                               title=f'Transfer Size Analysis - {storage_type.upper()}, {num_nodes} Node(s)',
                               save_path=os.path.join(plot_dir, f'transfer_size_analysis_{storage_type}.pdf'))

def scaling_analysis(df: pd.DataFrame, plot_dir: str):
    """Perform scaling analysis across different numbers of nodes."""
    print("\n=== Scaling Analysis (Multiple Nodes) ===")
    
    # Check if DataFrame is empty
    if df.empty:
        print("Warning: DataFrame is empty. Skipping scaling analysis.")
        return
    
    # Analyze scaling behavior across different numbers of nodes
    transfer_size_mb = 64
    transfer_size_bytes = transfer_size_mb * 1024 * 1024
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, storage_type in enumerate(['beegfs', 'ssd', 'nfs', 'tmpfs']):
        if i < len(axes):
            ax = axes[i]
            
            # Filter data for this storage type and transfer size
            filtered_df = filter_data_by_conditions(
                df, storage_type=storage_type, transfer_size=transfer_size_bytes
            )
            
            if not filtered_df.empty:
                for operation in ['write', 'read']:
                    op_data = filtered_df[filtered_df['operation'] == operation]
                    if not op_data.empty:
                        # Group by number of nodes and calculate mean throughput
                        scaling_data = op_data.groupby('numNodes')['trMiB'].mean().reset_index()
                        ax.plot(scaling_data['numNodes'], scaling_data['trMiB'], 
                                marker='o' if operation == 'write' else 's',
                                label=f'{operation}', linewidth=2, markersize=8)
                
                ax.set_xlabel('Number of Nodes')
                ax.set_ylabel('Average Throughput (MiB/s)')
                ax.set_title(f'{storage_type.upper()} - {transfer_size_mb}MB Transfer')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, f'No data for {storage_type}', 
                        ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'scaling_analysis_all_storage.pdf'), bbox_inches='tight')
    plt.show()

def summary_statistics_analysis(df: pd.DataFrame, plot_dir: str):
    """Generate summary statistics and heatmap."""
    print("\n=== Summary Statistics ===")
    
    # Check if DataFrame is empty
    if df.empty:
        print("Warning: DataFrame is empty. Cannot generate summary statistics.")
        return pd.DataFrame()
    
    # Generate summary statistics
    summary_stats = get_summary_statistics(df)
    print("Summary Statistics:")
    print(summary_stats)
    
    # Create a heatmap of average throughput by storage type and operation
    pivot_data = df.groupby(['storageType', 'operation'])['trMiB'].mean().unstack()
    
    # Check if pivot_data is empty before creating heatmap
    if pivot_data.empty:
        print("Warning: No data available for heatmap. Skipping heatmap generation.")
    else:
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Average Throughput (MiB/s)'})
        plt.title('Average Throughput by Storage Type and Operation')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'throughput_heatmap.pdf'), bbox_inches='tight')
        plt.show()
    
    return summary_stats

def analyze_specific_conditions(df: pd.DataFrame, storage_type: str, transfer_size_mb: int, num_nodes: int, plot_dir: str):
    """
    Analyze data for specific conditions and create detailed plots.
    """
    transfer_size_bytes = transfer_size_mb * 1024 * 1024
    
    # Filter data
    filtered_df = filter_data_by_conditions(
        df, storage_type=storage_type, transfer_size=transfer_size_bytes, num_nodes=num_nodes
    )
    
    if filtered_df.empty:
        print(f"No data found for {storage_type}, {transfer_size_mb}MB, {num_nodes} nodes")
        return
    
    print(f"Analysis for {storage_type.upper()}, {transfer_size_mb}MB transfer, {num_nodes} node(s):")
    print(f"Number of data points: {len(filtered_df)}")
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Throughput vs Number of Tasks
    ax1 = axes[0]
    for operation in ['write', 'read']:
        op_data = filtered_df[filtered_df['operation'] == operation]
        if not op_data.empty:
            ax1.scatter(op_data['numTasks'], op_data['trMiB'], 
                        label=f'{operation}', 
                        marker='o' if operation == 'write' else 's',
                        s=100)
    
    ax1.set_xlabel('Number of Tasks')
    ax1.set_ylabel('Throughput (MiB/s)')
    ax1.set_title('Throughput vs Number of Tasks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total Time vs Number of Tasks
    ax2 = axes[1]
    for operation in ['write', 'read']:
        op_data = filtered_df[filtered_df['operation'] == operation]
        if not op_data.empty:
            ax2.scatter(op_data['numTasks'], op_data['totalTime'], 
                        label=f'{operation}', 
                        marker='o' if operation == 'write' else 's',
                        s=100)
    
    ax2.set_xlabel('Number of Tasks')
    ax2.set_ylabel('Total Time (seconds)')
    ax2.set_title('Total Time vs Number of Tasks')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'detailed_analysis_{storage_type}_{transfer_size_mb}mb_{num_nodes}nodes.pdf'), 
                bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nStatistics:")
    print(filtered_df.groupby('operation')[['trMiB', 'totalTime']].describe())

def export_data(df: pd.DataFrame, summary_stats: pd.DataFrame):
    """Export filtered datasets and summary statistics."""
    print("\n=== Data Export ===")
    
    # Check if DataFrame is empty
    if df.empty:
        print("Warning: DataFrame is empty. Cannot export data.")
        return
    
    # Export data for each storage type separately
    for storage_type in df['storageType'].unique():
        storage_df = filter_data_by_conditions(df, storage_type=storage_type)
        filename = f'ior_data_{storage_type}.csv'
        storage_df.to_csv(filename, index=False)
        print(f"Exported {storage_type} data to {filename} ({len(storage_df)} records)")
    
    # Export summary statistics only if not empty
    if not summary_stats.empty:
        summary_stats.to_csv('ior_summary_statistics.csv')
        print("\nExported summary statistics to ior_summary_statistics.csv")
    else:
        print("\nWarning: No summary statistics to export.")

def main():
    """Main function to run the IOR analysis."""
    parser = argparse.ArgumentParser(description='IOR Benchmark Analysis Script')
    parser.add_argument('--data-dir', type=str, default='ior_data',
                       help='Directory containing IOR benchmark data (default: ior_data)')
    parser.add_argument('--csv-file', type=str, default='updated_master_ior_df.csv',
                       help='CSV file to load data from if data directory not found (default: updated_master_ior_df.csv)')
    parser.add_argument('--storage-type', type=str, default='beegfs',
                       help='Storage type for detailed analysis (default: beegfs)')
    parser.add_argument('--transfer-size', type=int, default=64,
                       help='Transfer size in MB for detailed analysis (default: 64)')
    parser.add_argument('--num-nodes', type=int, default=1,
                       help='Number of nodes for detailed analysis (default: 1)')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots (useful for data processing only)')
    
    args = parser.parse_args()
    
    print("IOR Benchmark Analysis Script")
    print("=" * 50)
    
    # Set up plotting
    plot_dir = setup_plotting()
    
    # Collect and process data
    df = collect_and_process_data(args.data_dir, args.csv_file)
    
    # Check if DataFrame is empty after processing
    if df.empty:
        print("Error: No data available for analysis. Please check your data directory and files.")
        return
    
    if not args.skip_plots:
        # Perform various analyses
        storage_comparison_analysis(df, plot_dir)
        transfer_size_analysis(df, plot_dir)
        scaling_analysis(df, plot_dir)
        summary_stats = summary_statistics_analysis(df, plot_dir)
        
        # Custom analysis
        print("\n=== Custom Analysis ===")
        analyze_specific_conditions(df, args.storage_type, args.transfer_size, args.num_nodes, plot_dir)
    else:
        # Just generate summary statistics without plots
        summary_stats = get_summary_statistics(df)
        print("Summary Statistics:")
        print(summary_stats)
    
    # Export data
    export_data(df, summary_stats)
    
    # Final summary
    print("\n=== Analysis Complete ===")
    print(f"Total records processed: {len(df)}")
    print(f"Storage types analyzed: {sorted(df['storageType'].unique())}")
    print(f"Number of nodes tested: {sorted(df['numNodes'].unique())}")
    print(f"Transfer sizes tested: {sorted(df['transferSize'].unique() // (1024*1024))} MB")
    if not args.skip_plots:
        print(f"\nAll plots have been saved to the '{plot_dir}' directory.")

if __name__ == "__main__":
    main()
