#!/usr/bin/env python3
"""
Test script for averaged statistics functionality.
This script demonstrates how to calculate averaged statistics for IOR benchmark data.
"""

import pandas as pd
import numpy as np
import os
from ior_utils import *

def test_averaged_statistics():
    """Test the averaged statistics calculation functionality."""
    
    print("=== Testing Averaged Statistics Functionality ===\n")
    
    # Load existing master DataFrame (updated with cp_data)
    if os.path.exists('updated_master_ior_df.csv'):
        print("Loading updated master DataFrame (with cp_data integration)...")
        df = load_master_ior_df('updated_master_ior_df.csv')
        print(f"Loaded DataFrame shape: {df.shape}")
    elif os.path.exists('master_ior_df.csv'):
        print("Loading original master DataFrame...")
        df = load_master_ior_df('master_ior_df.csv')
        print(f"Loaded DataFrame shape: {df.shape}")
    else:
        print("No existing master DataFrame found. Please run the main analysis first.")
        return
    
    # Display original data overview
    print("\nOriginal data overview:")
    # Handle NaN values in storageType before sorting
    storage_types = df['storageType'].dropna().unique()
    print(f"Storage types: {sorted(storage_types)}")
    print(f"Number of records: {len(df)}")
    
    # Check for NaN values in storageType
    nan_count = df['storageType'].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} rows have NaN values in storageType column")
        # Remove rows with NaN storageType for processing
        df = df.dropna(subset=['storageType'])
        print(f"Remaining records after removing NaN storageType: {len(df)}")
    
    # --- Averaging logic: keep only averaged or single-trial rows ---
    print("\nCalculating averaged statistics (only averaged or single-trial rows will be saved)...")
    
    group_cols = ['storageType', 'operation', 'randomOffset', 'transferSize', 'aggregateFilesizeMB',
                  'numTasks', 'numNodes', 'tasksPerNode', 'parallelism']
    
    averaged_rows = []
    for storage_type in df['storageType'].unique():
        if storage_type.startswith('ave_'):
            continue
        storage_data = df[df['storageType'] == storage_type]
        grouped = storage_data.groupby(group_cols, dropna=False)
        for group_keys, group_df in grouped:
            if len(group_df) == 1:
                # Only one trial, keep as is, but mark as averaged
                row = group_df.iloc[0].copy()
                row['storageType'] = f'ave_{storage_type}'
                averaged_rows.append(row)
            else:
                # Multiple trials, average
                mean_row = group_df.iloc[0].copy()
                mean_row['trMiB'] = group_df['trMiB'].mean()
                mean_row['totalTime'] = group_df['totalTime'].mean()
                mean_row['storageType'] = f'ave_{storage_type}'
                averaged_rows.append(mean_row)
    
    averaged_df = pd.DataFrame(averaged_rows)
    print(f"Averaged/Single-trial data shape: {averaged_df.shape}")
    print(f"Averaged storage types: {sorted(averaged_df['storageType'].unique())}")
    
    # Save the averaged-only DataFrame
    output_file = 'master_ior_df_averaged.csv'
    save_master_ior_df(averaged_df, output_file)
    print(f"\nAveraged-only master DataFrame saved to: {output_file}")
    
    # # Export individual averaged CSV files
    # print("\nExporting individual averaged CSV files...")
    # for storage_type in ['beegfs', 'ssd', 'nfs', 'tmpfs']:
    #     averaged_storage_type = f'ave_{storage_type}'
    #     if averaged_storage_type in averaged_df['storageType'].unique():
    #         storage_df = averaged_df[averaged_df['storageType'] == averaged_storage_type]
    #         filename = f'ior_data_{averaged_storage_type}.csv'
    #         storage_df.to_csv(filename, index=False)
    #         print(f"  Exported {averaged_storage_type} data to {filename} ({len(storage_df)} records)")
    
    # Create a simple comparison plot
    print("\nCreating comparison plot...")
    try:
        # Example: Compare averaged for beegfs
        storage_type = 'beegfs'
        transfer_size_mb = 64
        transfer_size_bytes = transfer_size_mb * 1024 * 1024
        num_nodes = 1
        
        averaged_data = filter_data_by_conditions(
            averaged_df, 
            storage_type=f'ave_{storage_type}', 
            transfer_size=transfer_size_bytes, 
            num_nodes=num_nodes
        )
        
        if not averaged_data.empty:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            for operation in ['write', 'read']:
                op_data = averaged_data[averaged_data['operation'] == operation]
                if not op_data.empty:
                    plt.scatter(op_data['numTasks'], op_data['trMiB'], 
                               label=f'{storage_type} {operation} (averaged)', 
                               marker='^',
                               s=150, alpha=0.9, edgecolors='black')
            plt.xlabel('Number of Tasks')
            plt.ylabel('Throughput (MiB/s)')
            plt.title(f'Averaged Data - {storage_type.upper()}, {transfer_size_mb}MB Transfer, {num_nodes} Node(s)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_dir = "plot"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(os.path.join(plot_dir, f'test_averaged_{storage_type}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.show()
            print(f"  Comparison plot saved to: {os.path.join(plot_dir, f'test_averaged_{storage_type}.png')}")
        else:
            print("  No data found for the specified conditions")
    except Exception as e:
        print(f"  Error creating plot: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_averaged_statistics() 