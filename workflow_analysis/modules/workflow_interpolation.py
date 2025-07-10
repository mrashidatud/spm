"""
Interpolation module for workflow analysis.
Contains functions for 4D interpolation to estimate transfer rates.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .workflow_config import MULTI_NODES, STORAGE_LIST



def calculate_4d_interpolation_with_extrapolation(df_ior_sorted, 
                                                  target_operation,
                                                  target_aggregateFilesizeMB, 
                                                  target_numNodes, 
                                                  target_parallelism, 
                                                  target_transfer_size,
                                                  par_col,
                                                  transferRate_column,
                                                  multi_nodes=True):
    """
    Perform 4D interpolation or extrapolation based on bounds for aggregateFilesizeMB, 
    numNodes, parallelism, and transferSize.

    Parameters:
    - df_ior_sorted: DataFrame containing the sorted IOR data (should be pre-filtered by storage type)
    - target_operation: The operation to filter by
    - target_aggregateFilesizeMB: Target aggregate file size (MB)
    - target_numNodes: Target number of nodes
    - target_parallelism: Target parallelism value
    - target_transfer_size: Target transfer size
    - par_col: The column name representing parallelism ('tasksPerNode' or 'parallelism')
    - transferRate_column: Column name of the transfer rate values to interpolate (e.g., 'trMiB')
    - multi_nodes: Boolean indicating if using multi-node setup

    Returns:
    - tuple: (interpolated_transfer_rate, transfer_size_slope)
    """
    
    def get_bounds(values, target):
        """Get bounds for interpolation or next two bounds when outside the range."""
        values = sorted(values)
        values = np.array(values, dtype=np.float64)

        if len(values) < 2:
            return (values[0], values[0]) if len(values) == 1 else (None, None)

        if target <= values[0]:
            return values[0], values[1]
        if target >= values[-1]:
            return values[-2], values[-1]

        lower = np.max(values[values < target], initial=None)
        upper = np.min(values[values >= target], initial=None)
        return lower, upper

    def calculate_interpolation(target_val, lower, upper, low_val, high_val):
        """Perform interpolation or extrapolation for a single dimension."""
        if target_val < lower:
            # Extrapolate below lower bound
            slope = (high_val - low_val) / (upper - lower) if upper != lower else 0
            return low_val - slope * (lower - target_val), slope
        elif target_val > upper:
            # Extrapolate above upper bound
            slope = (high_val - low_val) / (upper - lower) if upper != lower else 0
            return high_val + slope * (target_val - upper), slope
        else:
            # Interpolate within bounds
            if upper > lower:
                slope = (target_val - lower) / (upper - lower)
                return low_val + slope * (high_val - low_val), slope
            else:
                return low_val, 1
    

    # Filter by operation
    df_ior_filtered = df_ior_sorted[df_ior_sorted['operation'] == target_operation].copy()
    
    if df_ior_filtered.empty:
        raise ValueError(f"No rows found for the specified operation: {target_operation}.")

    # Filter for numNodes
    if multi_nodes:
        numNodes_values = sorted(df_ior_filtered['numNodes'].unique())
        lower_nodes, upper_nodes = get_bounds(numNodes_values, target_numNodes)
        df_ior_filtered = df_ior_filtered[df_ior_filtered['numNodes'].isin([lower_nodes, upper_nodes])]
    else:
        df_ior_filtered = df_ior_filtered[df_ior_filtered['numNodes'] == 1]

    # Parallelism bounds
    parallelism_values = sorted(df_ior_filtered[par_col].unique())
    lower_tasks, upper_tasks = get_bounds(parallelism_values, target_parallelism)
    df_ior_filtered = df_ior_filtered[df_ior_filtered[par_col].isin([lower_tasks, upper_tasks])]
    
    # Transfer size bounds
    transfer_values = sorted(df_ior_filtered['transferSize'].unique())
    lower_transfer, upper_transfer = get_bounds(transfer_values, target_transfer_size)
    df_ior_filtered = df_ior_filtered[df_ior_filtered['transferSize'].isin([lower_transfer, upper_transfer])]

    # Aggregate file size bounds
    aggregate_size_values = sorted(df_ior_filtered['aggregateFilesizeMB'].unique())
    lower_size, upper_size = get_bounds(aggregate_size_values, target_aggregateFilesizeMB)
    df_ior_filtered = df_ior_filtered[df_ior_filtered['aggregateFilesizeMB'].isin([lower_size, upper_size])]
    
    filtered_df = df_ior_filtered

    # Get bounds for each dimension
    agg_values = filtered_df['aggregateFilesizeMB'].unique()
    node_values = filtered_df['numNodes'].unique()
    par_values = filtered_df[par_col].unique()
    ts_values = filtered_df['transferSize'].unique()

    agg_lower, agg_upper = get_bounds(agg_values, target_aggregateFilesizeMB)
    node_lower, node_upper = get_bounds(node_values, target_numNodes)
    par_lower, par_upper = get_bounds(par_values, target_parallelism)
    ts_lower, ts_upper = get_bounds(ts_values, target_transfer_size)

    # Calculate mean values for each dimension
    agg_low_val = filtered_df.loc[filtered_df['aggregateFilesizeMB'] == agg_lower, transferRate_column].mean()
    agg_high_val = filtered_df.loc[filtered_df['aggregateFilesizeMB'] == agg_upper, transferRate_column].mean()
    
    # Handle NaN values
    if pd.isna(agg_high_val):
        agg_high_val = filtered_df.loc[filtered_df['aggregateFilesizeMB'] == agg_upper, transferRate_column].dropna().mean()

    node_low_val = filtered_df.loc[filtered_df['numNodes'] == node_lower, transferRate_column].mean()
    node_high_val = filtered_df.loc[filtered_df['numNodes'] == node_upper, transferRate_column].mean()

    par_low_val = filtered_df.loc[filtered_df[par_col] == par_lower, transferRate_column].mean()
    par_high_val = filtered_df.loc[filtered_df[par_col] == par_upper, transferRate_column].mean()

    ts_low_val = filtered_df.loc[filtered_df['transferSize'] == ts_lower, transferRate_column].mean()
    ts_high_val = filtered_df.loc[filtered_df['transferSize'] == ts_upper, transferRate_column].mean()

    # Interpolate for each dimension
    totalSize_interpolated, agg_slope = calculate_interpolation(target_aggregateFilesizeMB, agg_lower, agg_upper, agg_low_val, agg_high_val)
    node_interpolated, node_slope = calculate_interpolation(target_numNodes, node_lower, node_upper, node_low_val, node_high_val)
    par_interpolated, par_slope = calculate_interpolation(target_parallelism, par_lower, par_upper, par_low_val, par_high_val)
    ts_interpolated, ts_slope = calculate_interpolation(target_transfer_size, ts_lower, ts_upper, ts_low_val, ts_high_val)

    # Combine results (weighted average)
    combined_weight = 1.0
    total_weight = combined_weight
    estimated_trMiB_storage = combined_weight * (totalSize_interpolated + node_interpolated + par_interpolated + ts_interpolated) / 4

    # Normalize by total weight
    if total_weight > 0:
        estimated_trMiB_storage /= total_weight
    else:
        raise ValueError("No valid rows for interpolation or extrapolation. Check input bounds.")

    return estimated_trMiB_storage, ts_slope



def estimate_transfer_rates_for_workflow(wf_pfs_df, df_ior_sorted, storage_list, allowed_parallelism=None, multi_nodes=True, debug=False):
    """
    Estimate transfer rates for all tasks in a workflow dataframe using 4D interpolation.
    Handles cp/scp operations and ensures all parallelism values in cp/scp rows are included in allowed_parallelism.
    cp is operation 2, scp is operation 3.
    
    Parameters:
    - wf_pfs_df: Workflow DataFrame
    - df_ior_sorted: IOR benchmark data
    - storage_list: List of storage types
    - allowed_parallelism: List of allowed parallelism values (defaults to [1, max_parallelism] if None)
    - multi_nodes: Whether to use multi-node mode
    - debug: Whether to enable debug output
    """
    # Suppress Python warnings
    warnings.filterwarnings('ignore')
    # Set default allowed_parallelism if not provided
    if allowed_parallelism is None:
        # Get max parallelism from the workflow data
        max_parallelism = wf_pfs_df['parallelism'].max() if 'parallelism' in wf_pfs_df.columns else 1
        allowed_parallelism = [1, max_parallelism]
        if debug:
            print(f"Using default allowed_parallelism: {allowed_parallelism}")
    
    # Ensure all parallelism values in cp/scp rows are included
    cp_scp_parallelism = set(wf_pfs_df.loc[wf_pfs_df['operation'].isin(['cp', 'scp']), 'parallelism'].unique())
    allowed_parallelism = sorted(set(allowed_parallelism).union(cp_scp_parallelism))

    # Debug: Check for stage_in and stage_out tasks
    if debug:
        stage_tasks = wf_pfs_df[wf_pfs_df['taskName'].str.contains('stage_in|stage_out', na=False)]
        print(f"Found {len(stage_tasks)} stage_in/stage_out tasks:")
        for _, task in stage_tasks.iterrows():
            print(f"  Task: {task['taskName']}, Operation: {task['operation']}, Storage: {task['storageType']}")

    for index, row in wf_pfs_df.iterrows():
        operation = row['operation']
        transfer_size = row['transferSize']
        aggregateFilesizeMB = row['aggregateFilesizeMB']
        numNodes = row['numNodes']
        task_name = row.get('taskName', 'Unknown')

        # Debug: Check if this is a staging task
        is_staging_task = 'stage_in' in task_name or 'stage_out' in task_name
        if debug and is_staging_task:
            print(f"Processing staging task: {task_name}, Operation: {operation}, Storage: {row.get('storageType', 'N/A')}")

        # Determine parallelism range based on setup
        if multi_nodes:
            task_parallelism = row['tasksPerNode']
            parallelism_range = [task_parallelism]
            par_col = 'tasksPerNode'
        else:
            task_parallelism = row['parallelism']
            parallelism_range = [p for p in allowed_parallelism if p <= task_parallelism]
            par_col = 'parallelism'

        # For cp/scp, use the storageType column directly
        if operation in ['cp', 'scp']:
            storage_types = [row['storageType']]
        else:
            storage_types = storage_list

        for storage in storage_types:
            for parallelism in parallelism_range:
                col_name_tr_storage = f"estimated_trMiB_{storage}_{parallelism}p"
                col_name_ts_slope = f"estimated_ts_slope_{storage}_{parallelism}p"

                try:
                    # Map storage types to the actual values in IOR data storageType column
                    storage_filter = f'{storage}'
                    if storage == 'pfs':
                        storage_filter = 'beegfs'

                    df_ior_storage = df_ior_sorted[df_ior_sorted['storageType'] == storage_filter]

                    if df_ior_storage.empty:
                        if debug:
                            print(f"No data found for storage type: {storage_filter}")
                        continue

                    # Set operation code: cp=2, scp=3, else use row['operation']
                    if operation == 'cp':
                        op_code = 2
                    elif operation == 'scp':
                        op_code = 3
                    else:
                        op_code = row['operation']

                    estimated_trMiB_storage, ts_slope = calculate_4d_interpolation_with_extrapolation(
                        df_ior_storage,
                        op_code,
                        aggregateFilesizeMB,
                        numNodes,
                        parallelism,
                        transfer_size,
                        par_col,
                        'trMiB',
                        multi_nodes
                    )

                except ValueError as e:
                    if debug:
                        print(f"Error calculating transfer rate for {storage} storage, parallelism {parallelism}: {e}")
                    estimated_trMiB_storage = None
                    ts_slope = None

                if col_name_tr_storage not in wf_pfs_df.columns:
                    wf_pfs_df[col_name_tr_storage] = None
                if col_name_ts_slope not in wf_pfs_df.columns:
                    wf_pfs_df[col_name_ts_slope] = None

                wf_pfs_df.at[index, col_name_tr_storage] = estimated_trMiB_storage
                if ts_slope is not None:
                    wf_pfs_df.at[index, col_name_ts_slope] = float(ts_slope)

                if debug and is_staging_task:
                    print(f"Staging Task[{task_name}] Storage[{storage}] "
                          f"Parallelism[{parallelism}] aggregateFilesizeMB[{aggregateFilesizeMB}] "
                          f"-> {col_name_tr_storage} = {estimated_trMiB_storage}")
                elif debug:
                    print(f"Task[{task_name}] Storage[{storage}] "
                          f"Parallelism[{parallelism}] aggregateFilesizeMB[{aggregateFilesizeMB}] "
                          f"-> {col_name_tr_storage} = {estimated_trMiB_storage}")

    # Debug: Summary of staging tasks processed
    if debug:
        stage_tasks_processed = wf_pfs_df[wf_pfs_df['taskName'].str.contains('stage_in|stage_out', na=False)]
        print(f"\nTransfer rate estimation completed.")
        print(f"Total staging tasks processed: {len(stage_tasks_processed)}")
        if not stage_tasks_processed.empty:
            print("Sample staging tasks with transfer rates:")
            sample_cols = ['taskName', 'operation', 'storageType'] + [col for col in wf_pfs_df.columns if 'estimated_trMiB_' in col]
            print(stage_tasks_processed[sample_cols].head(3))

    return wf_pfs_df


def calculate_aggregate_filesize_per_node(wf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregate file size per node by grouping by taskName and numNodes.
    
    Parameters:
    - wf_df: Workflow DataFrame
    
    Returns:
    - DataFrame: Updated DataFrame with aggregateFilesizeMB calculated per node
    """
    # Step 1: Rename the original column to preserve it
    result_df = wf_df.rename(columns={"aggregateFilesizeMB": "aggregateFilesizeMBtask"})

    # Step 2: Group by taskName and numNodes, compute sum, then divide by numNodes
    group_sums = (
        result_df
        .groupby(["taskName", "numNodes"], as_index=False)["aggregateFilesizeMBtask"]
        .sum()
    )

    # Step 3: Compute the new aggregateFilesizeMB as sum / numNodes
    group_sums["aggregateFilesizeMB"] = group_sums["aggregateFilesizeMBtask"] / group_sums["numNodes"]

    # Step 4: Keep only the new column and keys for merging
    group_sums = group_sums[["taskName", "numNodes", "aggregateFilesizeMB"]]

    # Step 5: Merge back to the original dataframe
    result_df = result_df.merge(group_sums, on=["taskName", "numNodes"], how="left")
    
    return result_df 