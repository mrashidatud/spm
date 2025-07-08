#!/usr/bin/env python3
"""
Script to merge cp_data with master IOR DataFrame and save as updated_master_ior_df.csv
"""

import pandas as pd
import os

def merge_cp_data_with_ior():
    """Merge cp_data with master IOR DataFrame and save the result."""
    
    print("=== Merging CP Data with Master IOR DataFrame ===\n")
    
    # Check if master_move_data.csv exists
    if not os.path.exists('master_move_data.csv'):
        print("Error: master_move_data.csv not found. Please run move_data_bench_analysis.ipynb first.")
        return
    
    # Check if master_ior_df.csv exists
    if not os.path.exists('master_ior_df.csv'):
        print("Error: master_ior_df.csv not found. Please run ior_analysis_new.ipynb first.")
        return
    
    # Load the cp_data DataFrame
    print("Loading cp_data from master_move_data.csv...")
    cp_df = pd.read_csv('master_move_data.csv')
    print(f"CP data shape: {cp_df.shape}")
    print(f"CP data columns: {list(cp_df.columns)}")
    
    # Load the master IOR dataframe
    print("\nLoading master IOR DataFrame...")
    ior_df = pd.read_csv('master_ior_df.csv')
    print(f"Master IOR DataFrame shape: {ior_df.shape}")
    print(f"Master IOR DataFrame columns: {list(ior_df.columns)}")
    
    # Define keys used for joining
    merge_keys = ['operation', 'randomOffset', 'transferSize', 'aggregateFilesizeMB',
                  'numTasks', 'numNodes', 'tasksPerNode', 'parallelism']
    
    # Check if all merge keys exist in both DataFrames
    missing_keys_cp = [key for key in merge_keys if key not in cp_df.columns]
    missing_keys_ior = [key for key in merge_keys if key not in ior_df.columns]
    
    if missing_keys_cp:
        print(f"Warning: Missing keys in cp_data: {missing_keys_cp}")
    if missing_keys_ior:
        print(f"Warning: Missing keys in master IOR data: {missing_keys_ior}")
    
    # Make sure merge keys have the same dtype in both DataFrames (convert to str here)
    print("\nPreparing merge keys...")
    for key in merge_keys:
        if key in cp_df.columns and key in ior_df.columns:
            cp_df[key] = cp_df[key].astype(str)
            ior_df[key] = ior_df[key].astype(str)
    
    # Perform the outer merge (keep all rows and columns from both DataFrames)
    print("Performing merge...")
    merged_df = pd.merge(ior_df, cp_df, on=merge_keys, how='outer')
    
    # Save the merged DataFrame
    output_file = 'updated_master_ior_df.csv'
    merged_df.to_csv(output_file, index=False)
    
    # Show final shape and new columns
    print(f"\nMerge completed successfully!")
    print(f"Merged DataFrame shape: {merged_df.shape}")
    print(f"New columns added: {set(merged_df.columns) - set(ior_df.columns)}")
    print(f"Updated master DataFrame saved to: {output_file}")
    
    # Display summary of the merged data
    print(f"\nSummary of merged data:")
    print(f"Original IOR records: {len(ior_df)}")
    print(f"CP data records: {len(cp_df)}")
    print(f"Merged records: {len(merged_df)}")
    
    # Show storage types in merged data
    if 'storageType' in merged_df.columns:
        storage_types = merged_df['storageType'].dropna().unique()
        print(f"Storage types in merged data: {sorted(storage_types)}")
    
    # Show data types in merged DataFrame
    print(f"\nData types in merged DataFrame:")
    for col in merged_df.columns:
        print(f"  {col}: {merged_df[col].dtype}")
    
    print("\n=== Merge Complete ===")

if __name__ == "__main__":
    merge_cp_data_with_ior()