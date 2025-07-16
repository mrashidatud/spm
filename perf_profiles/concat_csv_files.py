#!/usr/bin/env python3
"""
Script to concatenate master_ior_df.csv and master_move_data.csv
First calculates averaged statistics for IOR data, then concatenates with cp data.
"""

import pandas as pd
import os

def calculate_averaged_statistics(df):
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
        if storage_type.startswith('ave_') or pd.isna(storage_type):  # Skip already averaged data
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

def concat_csv_files_ave():
    """Concatenate master_ior_df.csv and master_move_data.csv into updated_master_ior_df.csv."""
    
    print("=== Concatenating CSV Files with Averaging ===\n")
    
    # Check if both files exist
    ior_file = 'master_ior_df.csv'
    move_file = 'master_move_data.csv'
    
    if not os.path.exists(ior_file):
        print(f"Error: {ior_file} not found. Please run ior_analysis_new.ipynb first.")
        return
    
    if not os.path.exists(move_file):
        print(f"Error: {move_file} not found. Please run move_data_bench_analysis.ipynb first.")
        return
    
    # Load both CSV files
    print(f"Loading {ior_file}...")
    ior_df = pd.read_csv(ior_file)
    print(f"  Shape: {ior_df.shape}")
    print(f"  Columns: {list(ior_df.columns)}")
    
    # Handle NaN values in storageType before processing
    nan_count = ior_df['storageType'].isna().sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} rows have NaN values in storageType column")
        ior_df = ior_df.dropna(subset=['storageType'])
        print(f"  Remaining records after removing NaN storageType: {len(ior_df)}")
    
    # Calculate averaged statistics for IOR data
    print(f"\nProcessing IOR data with averaging...")
    ior_df_with_averages = calculate_averaged_statistics(ior_df)
    print(f"  IOR data shape after averaging: {ior_df_with_averages.shape}")
    
    # Filter to keep only averaged values (ave_ prefixed) and original non-IOR storage types
    print(f"\nFiltering to keep only averaged values...")
    original_storage_types = set(ior_df['storageType'].unique())
    averaged_storage_types = [st for st in ior_df_with_averages['storageType'].unique() if st.startswith('ave_')]
    
    # Keep only averaged IOR data and any non-IOR data (like cp data that might have been added)
    ior_df_averaged_only = ior_df_with_averages[
        (ior_df_with_averages['storageType'].str.startswith('ave_')) | 
        (~ior_df_with_averages['storageType'].isin(original_storage_types))
    ].copy()
    
    print(f"  Original IOR storage types: {sorted(original_storage_types)}")
    print(f"  Averaged storage types: {sorted(averaged_storage_types)}")
    print(f"  IOR data shape after filtering (averaged only): {ior_df_averaged_only.shape}")
    
    # Save averaged IOR data (averaged only)
    ior_averaged_file = 'master_ior_df_averaged.csv'
    ior_df_averaged_only.to_csv(ior_averaged_file, index=False)
    print(f"  Averaged-only IOR data saved to: {ior_averaged_file}")
    
    print(f"\nLoading {move_file}...")
    move_df = pd.read_csv(move_file)
    print(f"  Shape: {move_df.shape}")
    print(f"  Columns: {list(move_df.columns)}")
    
    # Check if columns match
    ior_cols = set(ior_df_averaged_only.columns)
    move_cols = set(move_df.columns)
    
    if ior_cols != move_cols:
        print(f"\nWarning: Column mismatch detected!")
        print(f"Columns in {ior_file} (averaged only): {sorted(ior_cols)}")
        print(f"Columns in {move_file}: {sorted(move_cols)}")
        print(f"Missing in move_data: {ior_cols - move_cols}")
        print(f"Extra in move_data: {move_cols - ior_cols}")
        
        # Try to align columns
        common_cols = ior_cols.intersection(move_cols)
        print(f"\nUsing common columns: {sorted(common_cols)}")
        
        ior_df_averaged_only = ior_df_averaged_only[list(common_cols)]
        move_df = move_df[list(common_cols)]
    else:
        print(f"\n✓ Column structure matches between both files")
    
    # Concatenate the DataFrames
    print(f"\nConcatenating DataFrames...")
    combined_df = pd.concat([ior_df_averaged_only, move_df], ignore_index=True)
    
    # Save the combined DataFrame
    output_file = 'updated_master_ior_df.csv'
    combined_df.to_csv(output_file, index=False)
    
    # Show results
    print(f"\n=== Concatenation Complete ===")
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Output saved to: {output_file}")
    
    # Display summary
    print(f"\nSummary:")
    print(f"  Original {ior_file} records: {len(ior_df)}")
    print(f"  Averaged-only IOR records: {len(ior_df_averaged_only)}")
    print(f"  {move_file} records: {len(move_df)}")
    print(f"  Combined records: {len(combined_df)}")
    
    # Show storage types if available
    if 'storageType' in combined_df.columns:
        storage_types = combined_df['storageType'].dropna().unique()
        averaged_types = [st for st in storage_types if st.startswith('ave_')]
        cp_types = [st for st in storage_types if not st.startswith('ave_')]
        print(f"  Averaged IOR storage types: {sorted(averaged_types)}")
        print(f"  CP storage types: {sorted(cp_types)}")
    
    # Show sample of combined data
    print(f"\nSample of combined data:")
    print(combined_df.head(3))
    
    # # Export individual averaged CSV files for each storage type
    # print(f"\nExporting individual averaged CSV files...")
    # for storage_type in ['beegfs', 'ssd', 'nfs', 'tmpfs']:
    #     averaged_storage_type = f'ave_{storage_type}'
    #     if averaged_storage_type in combined_df['storageType'].unique():
    #         storage_df = combined_df[combined_df['storageType'] == averaged_storage_type]
    #         filename = f'ior_data_{averaged_storage_type}.csv'
    #         storage_df.to_csv(filename, index=False)
    #         print(f"  Exported {averaged_storage_type} data to {filename} ({len(storage_df)} records)")
    
    print(f"\n=== Done ===")

def concat_csv_files():
    """
    Concatenate master_ior_df.csv and master_move_data.csv into updated_master_ior_df.csv.
    """
    
    print("=== Concatenating CSV Files ===\n")
    
    # Check if both files exist
    ior_file = 'master_ior_df.csv'
    move_file = 'master_move_data.csv'
    
    if not os.path.exists(ior_file):
        print(f"Error: {ior_file} not found. Please run ior_analysis_new.ipynb first.")
        return
    
    if not os.path.exists(move_file):
        print(f"Error: {move_file} not found. Please run move_data_bench_analysis.ipynb first.")
        return
    
    # Load both CSV files
    print(f"Loading {ior_file}...")
    ior_df = pd.read_csv(ior_file)
    print(f"  Shape: {ior_df.shape}")
    print(f"  Columns: {list(ior_df.columns)}")
    
    # Handle NaN values in storageType before processing
    if 'storageType' in ior_df.columns:
        nan_count = ior_df['storageType'].isna().sum()
        if nan_count > 0:
            print(f"  Warning: {nan_count} rows have NaN values in storageType column")
            ior_df = ior_df.dropna(subset=['storageType'])
            print(f"  Remaining records after removing NaN storageType: {len(ior_df)}")
    
    print(f"\nLoading {move_file}...")
    move_df = pd.read_csv(move_file)
    print(f"  Shape: {move_df.shape}")
    print(f"  Columns: {list(move_df.columns)}")
    
    # Check if columns match
    ior_cols = set(ior_df.columns)
    move_cols = set(move_df.columns)
    
    if ior_cols != move_cols:
        print(f"\nWarning: Column mismatch detected!")
        print(f"Columns in {ior_file}: {sorted(ior_cols)}")
        print(f"Columns in {move_file}: {sorted(move_cols)}")
        print(f"Missing in move_data: {ior_cols - move_cols}")
        print(f"Extra in move_data: {move_cols - ior_cols}")
        
        # Try to align columns
        common_cols = ior_cols.intersection(move_cols)
        print(f"\nUsing common columns: {sorted(common_cols)}")
        
        ior_df = ior_df[list(common_cols)]
        move_df = move_df[list(common_cols)]
    else:
        print(f"\n✓ Column structure matches between both files")
    
    # Concatenate the DataFrames
    print(f"\nConcatenating DataFrames...")
    combined_df = pd.concat([ior_df, move_df], ignore_index=True)
    
    # Save the combined DataFrame
    output_file = 'updated_master_ior_df.csv'
    combined_df.to_csv(output_file, index=False)
    
    # Show results
    print(f"\n=== Concatenation Complete ===")
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Output saved to: {output_file}")
    
    # Display summary
    print(f"\nSummary:")
    print(f"  {ior_file} records: {len(ior_df)}")
    print(f"  {move_file} records: {len(move_df)}")
    print(f"  Combined records: {len(combined_df)}")
    
    # Show storage types if available
    if 'storageType' in combined_df.columns:
        storage_types = sorted(combined_df['storageType'].dropna().unique())
        print(f"  Storage types: {storage_types}")
    
    # Show sample of combined data
    print(f"\nSample of combined data:")
    print(combined_df.head(3))
    
    print(f"\n=== Done ===")

if __name__ == "__main__":
    concat_csv_files() 