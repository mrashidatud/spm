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
                                                  multi_nodes=True,
                                                  debug=False):
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
            result = low_val - slope * (lower - target_val)
            # Ensure extrapolation doesn't go negative
            if result < 0:
                # Use the closest positive value, or the least negative if both are negative
                if low_val > 0 and high_val > 0:
                    result = min(low_val, high_val)  # Use the smaller positive value
                elif low_val > 0:
                    result = low_val
                elif high_val > 0:
                    result = high_val
                else:
                    result = max(low_val, high_val)  # Use the least negative value
            return result, slope
        elif target_val > upper:
            # Extrapolate above upper bound
            slope = (high_val - low_val) / (upper - lower) if upper != lower else 0
            result = high_val + slope * (target_val - upper)
            # Ensure extrapolation doesn't go negative
            if result < 0:
                # Use the closest positive value, or the least negative if both are negative
                if low_val > 0 and high_val > 0:
                    result = min(low_val, high_val)  # Use the smaller positive value
                elif low_val > 0:
                    result = low_val
                elif high_val > 0:
                    result = high_val
                else:
                    result = max(low_val, high_val)  # Use the least negative value
            return result, slope
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

    if debug:
        print(f"\n[DEBUG] Original IOR data analysis:")
        print(f"[DEBUG] Total IOR data shape: {df_ior_sorted.shape}")
        print(f"[DEBUG] Data for operation '{target_operation}': {len(df_ior_filtered)} rows")
        
        # Check for negative values in the original data
        negative_tr_original = df_ior_sorted[df_ior_sorted[transferRate_column] < 0]
        if not negative_tr_original.empty:
            print(f"[WARNING] Found {len(negative_tr_original)} rows with negative {transferRate_column} in original data:")
            print(negative_tr_original[[transferRate_column, 'operation', 'storageType', 'aggregateFilesizeMB', 'numNodes', par_col, 'transferSize']].head())
        
        # Check for negative values in other columns
        for col in ['aggregateFilesizeMB', 'numNodes', par_col, 'transferSize']:
            if col in df_ior_sorted.columns:
                negative_values = df_ior_sorted[df_ior_sorted[col] < 0]
                if not negative_values.empty:
                    print(f"[WARNING] Found {len(negative_values)} rows with negative {col} in original data:")
                    print(negative_values[col].head())
        
        print(f"[DEBUG] Available operations in IOR data: {sorted(df_ior_sorted['operation'].unique())}")
        print(f"[DEBUG] Available storage types: {sorted(df_ior_sorted['storageType'].unique())}")

    # Filter for numNodes
    if multi_nodes:
        numNodes_values = sorted(df_ior_filtered['numNodes'].unique())
        lower_nodes, upper_nodes = get_bounds(numNodes_values, target_numNodes)
        df_ior_filtered = df_ior_filtered[df_ior_filtered['numNodes'].isin([lower_nodes, upper_nodes])]
        if debug:
            print(f"[DEBUG] After numNodes filtering: {len(df_ior_filtered)} rows (bounds: {lower_nodes}-{upper_nodes})")
    else:
        df_ior_filtered = df_ior_filtered[df_ior_filtered['numNodes'] == 1]
        if debug:
            print(f"[DEBUG] After numNodes filtering (single node): {len(df_ior_filtered)} rows")

    # Parallelism bounds
    parallelism_values = sorted(df_ior_filtered[par_col].unique())
    lower_tasks, upper_tasks = get_bounds(parallelism_values, target_parallelism)
    df_ior_filtered = df_ior_filtered[df_ior_filtered[par_col].isin([lower_tasks, upper_tasks])]
    if debug:
        print(f"[DEBUG] After parallelism filtering: {len(df_ior_filtered)} rows (bounds: {lower_tasks}-{upper_tasks})")
    
    # Transfer size bounds
    transfer_values = sorted(df_ior_filtered['transferSize'].unique())
    lower_transfer, upper_transfer = get_bounds(transfer_values, target_transfer_size)
    df_ior_filtered = df_ior_filtered[df_ior_filtered['transferSize'].isin([lower_transfer, upper_transfer])]
    if debug:
        print(f"[DEBUG] After transfer size filtering: {len(df_ior_filtered)} rows (bounds: {lower_transfer}-{upper_transfer})")

    # Aggregate file size bounds
    aggregate_size_values = sorted(df_ior_filtered['aggregateFilesizeMB'].unique())
    lower_size, upper_size = get_bounds(aggregate_size_values, target_aggregateFilesizeMB)
    if debug:
        print(f"[DEBUG] Aggregate size bounds: {lower_size}-{upper_size} for target {target_aggregateFilesizeMB}")
        print(f"[DEBUG] Available aggregate sizes: {aggregate_size_values[:5]}...{aggregate_size_values[-5:]}")
    df_ior_filtered = df_ior_filtered[df_ior_filtered['aggregateFilesizeMB'].isin([lower_size, upper_size])]
    if debug:
        print(f"[DEBUG] After aggregate size filtering: {len(df_ior_filtered)} rows (bounds: {lower_size}-{upper_size})")
        
        # Check for negative values after each filtering step
        negative_tr_after_filter = df_ior_filtered[df_ior_filtered[transferRate_column] < 0]
        if not negative_tr_after_filter.empty:
            print(f"[WARNING] Found {len(negative_tr_after_filter)} rows with negative {transferRate_column} after filtering:")
            print(negative_tr_after_filter[[transferRate_column, 'aggregateFilesizeMB', 'numNodes', par_col, 'transferSize']].head())

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

    if debug:
        print(f"\n[DEBUG] Data used for bounds calculation:")
        print(f"[DEBUG] Aggregate size bounds: {agg_lower} - {agg_upper}")
        agg_low_data = filtered_df[filtered_df['aggregateFilesizeMB'] == agg_lower]
        agg_high_data = filtered_df[filtered_df['aggregateFilesizeMB'] == agg_upper]
        print(f"[DEBUG] Data for agg_lower ({agg_lower}): {len(agg_low_data)} rows")
        if not agg_low_data.empty:
            print(f"[DEBUG]   {transferRate_column} values: {agg_low_data[transferRate_column].tolist()}")
        print(f"[DEBUG] Data for agg_upper ({agg_upper}): {len(agg_high_data)} rows")
        if not agg_high_data.empty:
            print(f"[DEBUG]   {transferRate_column} values: {agg_high_data[transferRate_column].tolist()}")
        
        print(f"[DEBUG] Node bounds: {node_lower} - {node_upper}")
        node_low_data = filtered_df[filtered_df['numNodes'] == node_lower]
        node_high_data = filtered_df[filtered_df['numNodes'] == node_upper]
        print(f"[DEBUG] Data for node_lower ({node_lower}): {len(node_low_data)} rows")
        if not node_low_data.empty:
            print(f"[DEBUG]   {transferRate_column} values: {node_low_data[transferRate_column].tolist()}")
        print(f"[DEBUG] Data for node_upper ({node_upper}): {len(node_high_data)} rows")
        if not node_high_data.empty:
            print(f"[DEBUG]   {transferRate_column} values: {node_high_data[transferRate_column].tolist()}")
        
        print(f"[DEBUG] Parallelism bounds: {par_lower} - {par_upper}")
        par_low_data = filtered_df[filtered_df[par_col] == par_lower]
        par_high_data = filtered_df[filtered_df[par_col] == par_upper]
        print(f"[DEBUG] Data for par_lower ({par_lower}): {len(par_low_data)} rows")
        if not par_low_data.empty:
            print(f"[DEBUG]   {transferRate_column} values: {par_low_data[transferRate_column].tolist()}")
        print(f"[DEBUG] Data for par_upper ({par_upper}): {len(par_high_data)} rows")
        if not par_high_data.empty:
            print(f"[DEBUG]   {transferRate_column} values: {par_high_data[transferRate_column].tolist()}")
        
        print(f"[DEBUG] Transfer size bounds: {ts_lower} - {ts_upper}")
        ts_low_data = filtered_df[filtered_df['transferSize'] == ts_lower]
        ts_high_data = filtered_df[filtered_df['transferSize'] == ts_upper]
        print(f"[DEBUG] Data for ts_lower ({ts_lower}): {len(ts_low_data)} rows")
        if not ts_low_data.empty:
            print(f"[DEBUG]   {transferRate_column} values: {ts_low_data[transferRate_column].tolist()}")
        print(f"[DEBUG] Data for ts_upper ({ts_upper}): {len(ts_high_data)} rows")
        if not ts_high_data.empty:
            print(f"[DEBUG]   {transferRate_column} values: {ts_high_data[transferRate_column].tolist()}")

    # Debug: Check for NaN or negative values in the bounds
    debug_bounds = {
        'agg': (agg_low_val, agg_high_val, agg_lower, agg_upper),
        'node': (node_low_val, node_high_val, node_lower, node_upper),
        'par': (par_low_val, par_high_val, par_lower, par_upper),
        'ts': (ts_low_val, ts_high_val, ts_lower, ts_upper)
    }
    
    if debug:
        print(f"[DEBUG] Target values: agg={target_aggregateFilesizeMB}, nodes={target_numNodes}, par={target_parallelism}, ts={target_transfer_size}")
        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")
        print(f"[DEBUG] Available values in filtered data:")
        print(f"  aggregateFilesizeMB: {sorted(filtered_df['aggregateFilesizeMB'].unique())}")
        print(f"  numNodes: {sorted(filtered_df['numNodes'].unique())}")
        print(f"  {par_col}: {sorted(filtered_df[par_col].unique())}")
        print(f"  transferSize: {sorted(filtered_df['transferSize'].unique())}")
        print(f"  {transferRate_column}: {sorted(filtered_df[transferRate_column].unique())}")
        
        # Check for negative values in the transfer rate column
        negative_tr_values = filtered_df[filtered_df[transferRate_column] < 0]
        if not negative_tr_values.empty:
            print(f"[WARNING] Found {len(negative_tr_values)} rows with negative {transferRate_column} values:")
            print(negative_tr_values[[transferRate_column, 'aggregateFilesizeMB', 'numNodes', par_col, 'transferSize']].head())
        
        # Check for negative values in other columns
        for col in ['aggregateFilesizeMB', 'numNodes', par_col, 'transferSize']:
            negative_values = filtered_df[filtered_df[col] < 0]
            if not negative_values.empty:
                print(f"[WARNING] Found {len(negative_values)} rows with negative {col} values:")
                print(negative_values[col].head())
    
    for dim, (low_val, high_val, lower_bound, upper_bound) in debug_bounds.items():
        if pd.isna(low_val) or pd.isna(high_val):
            print(f"[WARNING] NaN values in {dim} bounds: low={low_val}, high={high_val}")
        if low_val < 0 or high_val < 0:
            print(f"[WARNING] Negative values in {dim} bounds: low={low_val}, high={high_val}")
        if lower_bound is None or upper_bound is None:
            print(f"[WARNING] None bounds in {dim}: lower={lower_bound}, upper={upper_bound}")
        if debug:
            print(f"[DEBUG] {dim} bounds: {lower_bound}-{upper_bound}, values: {low_val}-{high_val}")
            # Show target value for aggregateFilesizeMB dimension
            if dim == 'agg':
                target_val = target_aggregateFilesizeMB
                print(f"[DEBUG] Target aggregateFilesizeMB: {target_val}")
                if target_val < lower_bound:
                    print(f"[DEBUG] Extrapolating BELOW bounds: target={target_val} < lower={lower_bound}")
                elif target_val > upper_bound:
                    print(f"[DEBUG] Extrapolating ABOVE bounds: target={target_val} > upper={upper_bound}")
                    slope = (high_val - low_val) / (upper_bound - lower_bound) if upper_bound != lower_bound else 0
                    extrapolated = high_val + slope * (target_val - upper_bound)
                    print(f"[DEBUG] Extrapolation calculation: {high_val} + {slope} * ({target_val} - {upper_bound}) = {extrapolated}")
                else:
                    print(f"[DEBUG] Interpolating within bounds")

    # Interpolate for each dimension
    totalSize_interpolated, agg_slope = calculate_interpolation(target_aggregateFilesizeMB, agg_lower, agg_upper, agg_low_val, agg_high_val)
    node_interpolated, node_slope = calculate_interpolation(target_numNodes, node_lower, node_upper, node_low_val, node_high_val)
    par_interpolated, par_slope = calculate_interpolation(target_parallelism, par_lower, par_upper, par_low_val, par_high_val)
    ts_interpolated, ts_slope = calculate_interpolation(target_transfer_size, ts_lower, ts_upper, ts_low_val, ts_high_val)

    # Debug: Print interpolation values
    debug_info = {
        'agg': (totalSize_interpolated, agg_low_val, agg_high_val, agg_lower, agg_upper),
        'node': (node_interpolated, node_low_val, node_high_val, node_lower, node_upper),
        'par': (par_interpolated, par_low_val, par_high_val, par_lower, par_upper),
        'ts': (ts_interpolated, ts_low_val, ts_high_val, ts_lower, ts_upper)
    }
    
    # Validate that interpolated values are reasonable (not negative for transfer rates)
    for dim, (interpolated_val, low_val, high_val, lower_bound, upper_bound) in debug_info.items():
        if interpolated_val < 0:
            print(f"[WARNING] Negative interpolation for {dim}: {interpolated_val} (bounds: {lower_bound}-{upper_bound}, values: {low_val}-{high_val})")
            # Use the closest positive value instead
            if low_val > 0:
                debug_info[dim] = (low_val, low_val, high_val, lower_bound, upper_bound)
            elif high_val > 0:
                debug_info[dim] = (high_val, low_val, high_val, lower_bound, upper_bound)
            else:
                debug_info[dim] = (max(low_val, high_val), low_val, high_val, lower_bound, upper_bound)  # Use the least negative value
    
    # Update the interpolated values after validation
    totalSize_interpolated = debug_info['agg'][0]
    node_interpolated = debug_info['node'][0]
    par_interpolated = debug_info['par'][0]
    ts_interpolated = debug_info['ts'][0]

    # Combine results using a more robust approach
    # Instead of simple averaging, use the median of positive values or the maximum if all are negative
    interpolated_values = [totalSize_interpolated, node_interpolated, par_interpolated, ts_interpolated]
    positive_values = [v for v in interpolated_values if v > 0]
    
    if positive_values:
        # Use median of positive values for more robust estimation
        estimated_trMiB_storage = np.median(positive_values)
    else:
        # If all values are negative, use the least negative one
        estimated_trMiB_storage = max(interpolated_values)
        print(f"[WARNING] All interpolated values are negative: {interpolated_values}")
        print(f"  Using least negative value: {estimated_trMiB_storage}")

    # Final validation: ensure the result is positive
    if estimated_trMiB_storage < 0:
        print(f"[WARNING] Negative final transfer rate: {estimated_trMiB_storage}")
        print(f"  Component values: agg={totalSize_interpolated}, node={node_interpolated}, par={par_interpolated}, ts={ts_interpolated}")
        # Use the maximum of the component values as a fallback
        estimated_trMiB_storage = max(totalSize_interpolated, node_interpolated, par_interpolated, ts_interpolated)
        if estimated_trMiB_storage < 0:
            # If all components are negative, use the least negative one
            estimated_trMiB_storage = max(totalSize_interpolated, node_interpolated, par_interpolated, ts_interpolated)
    
    # Ensure the final result is positive (transfer rates should never be negative)
    if estimated_trMiB_storage < 0:
        print(f"[ERROR] Transfer rate is still negative after all corrections: {estimated_trMiB_storage}")
        print(f"  Using absolute value as fallback")
        estimated_trMiB_storage = abs(estimated_trMiB_storage)

    # Normalize by total weight
    # total_weight is not defined in this function, so this line is commented out or removed if not needed.
    # if total_weight > 0:
    #     estimated_trMiB_storage /= total_weight
    # else:
    #     raise ValueError("No valid rows for interpolation or extrapolation. Check input bounds.")

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
    
    # Ensure all parallelism values in cp/scp/none rows are included
    cp_scp_parallelism = set(wf_pfs_df.loc[wf_pfs_df['operation'].isin(['cp', 'scp', 'none']), 'parallelism'].unique())
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

        # Handle virtual producers (operation = 'none') - set transfer rate to 0
        if operation == 'none':
            if debug:
                print(f"Setting transfer rate to 0 for virtual producer: {task_name} (operation = {operation})")
            
            # Set transfer rate to 0 for all storage types and parallelism levels
            for storage in storage_list:
                for parallelism in allowed_parallelism:
                    col_name_tr_storage = f"estimated_trMiB_{storage}_{parallelism}p"
                    col_name_ts_slope = f"estimated_ts_slope_{storage}_{parallelism}p"
                    
                    if col_name_tr_storage not in wf_pfs_df.columns:
                        wf_pfs_df[col_name_tr_storage] = None
                    if col_name_ts_slope not in wf_pfs_df.columns:
                        wf_pfs_df[col_name_ts_slope] = None
                    
                    wf_pfs_df.at[index, col_name_tr_storage] = 0.0
                    wf_pfs_df.at[index, col_name_ts_slope] = 0.0
            
            continue

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
                    if storage == 'pfs': # FIXME: temporary fix for pfs storage type
                        storage_filter = 'beegfs'

                    df_ior_storage = df_ior_sorted[df_ior_sorted['storageType'] == storage_filter]

                    if df_ior_storage.empty:
                        if debug:
                            print(f"No data found for storage type: {storage_filter}")
                        continue

                    estimated_trMiB_storage, ts_slope = calculate_4d_interpolation_with_extrapolation(
                        df_ior_storage,
                        operation,
                        aggregateFilesizeMB,
                        numNodes,
                        parallelism,
                        transfer_size,
                        par_col,
                        'trMiB',
                        multi_nodes,
                        debug and (task_name == 'individuals')  # Only debug for the problematic task
                    )
                    

                except ValueError as e:
                    if debug:
                        print(f"Error calculating transfer rate for {storage} storage, parallelism {parallelism}: {e}")
                    estimated_trMiB_storage = None
                    ts_slope = None
                except Exception as e:
                    if debug:
                        print(f"Unexpected error calculating transfer rate for {storage} storage, parallelism {parallelism}: {e}")
                    estimated_trMiB_storage = None
                    ts_slope = None

                if col_name_tr_storage not in wf_pfs_df.columns:
                    wf_pfs_df[col_name_tr_storage] = None
                if col_name_ts_slope not in wf_pfs_df.columns:
                    wf_pfs_df[col_name_ts_slope] = None

                wf_pfs_df.at[index, col_name_tr_storage] = estimated_trMiB_storage
                if ts_slope is not None:
                    wf_pfs_df.at[index, col_name_ts_slope] = float(ts_slope)

                # Debug: Check for negative transfer rate
                if estimated_trMiB_storage is not None and estimated_trMiB_storage < 0:
                    print(f"[NEGATIVE TRANSFER RATE] Task: {task_name}, Operation: {operation}, Storage: {storage}, Parallelism: {parallelism}, "
                          f"aggregateFilesizeMB: {aggregateFilesizeMB}, numNodes: {numNodes}, transfer_size: {transfer_size}, "
                          f"col_name: {col_name_tr_storage}, estimated_trMiB_storage: {estimated_trMiB_storage}, ts_slope: {ts_slope}")

                if debug and is_staging_task:
                    print(f"Staging Task[{task_name}] Storage[{storage}] "
                          f"Parallelism[{parallelism}] aggregateFilesizeMB[{aggregateFilesizeMB}] "
                          f"-> {col_name_tr_storage} = {estimated_trMiB_storage}")
                elif debug:
                    print(f"Task[{task_name}] Storage[{storage}] "
                          f"Parallelism[{parallelism}] aggregateFilesizeMB[{aggregateFilesizeMB}] "
                          f"-> {col_name_tr_storage} = {estimated_trMiB_storage}")

    return wf_pfs_df


def calculate_aggregate_filesize_per_node(wf_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Calculate aggregate file size per node for each unique (taskName, operation, numNodes).
    
    For each (taskName, operation, numNodes) combination:
    - For 'read' and 'write' operations: Group by unique taskPID and sum aggregateFilesizeMBtask values
    - For 'cp', 'scp', and 'none' operations: Use file-based calculation with prepared file size dictionary
    
    The result is assigned to 'aggregateFilesizeMB' for all rows with that (taskName, operation, numNodes) combination.
    """
    if debug:
        print(f"\n=== calculate_aggregate_filesize_per_node DEBUG ===")
        print(f"Input DataFrame shape: {wf_df.shape}")
        print(f"Columns: {list(wf_df.columns)}")
        print(f"Sample data:")
        print(wf_df[['taskName', 'operation', 'fileName', 'aggregateFilesizeMB']].head())
    
    # Step 1: Rename the original column to preserve it
    result_df = wf_df.rename(columns={"aggregateFilesizeMB": "aggregateFilesizeMBtask"}).copy()
    result_df["aggregateFilesizeMB"] = None
    
    if debug:
        print(f"After renaming - aggregateFilesizeMBtask sample values:")
        sample_values = result_df['aggregateFilesizeMBtask'].head(10).tolist()
        print(f"  {sample_values}")
    
    # Step 2: Prepare file size dictionary for cp/scp/none operations
    # Get file sizes from regular tasks (not stage_in or stage_out) with read and write operations
    regular_tasks_df = result_df[
        (~result_df['taskName'].str.contains('stage_in|stage_out', na=False)) & 
        (result_df['operation'].isin(['read', 'write']))
    ]
    file_size_dict = {}
    
    if debug:
        print(f"\nPreparing file size dictionary from regular tasks with read and write operations...")
        print(f"Regular tasks with read/write operations: {len(regular_tasks_df)} rows")
    
    for idx, row in regular_tasks_df.iterrows():
        file_name = row['fileName'].strip()
        if file_name and file_name not in file_size_dict:
            file_size_dict[file_name] = row['aggregateFilesizeMBtask']
        elif file_name in file_size_dict:
            # If multiple sizes exist for the same file, take the smallest one
            current_size = file_size_dict[file_name]
            new_size = row['aggregateFilesizeMBtask']
            if new_size < current_size:
                file_size_dict[file_name] = new_size
    
    # Also add files from cp and scp operations in regular tasks
    cp_scp_tasks_df = result_df[
        (~result_df['taskName'].str.contains('stage_in|stage_out', na=False)) & 
        (result_df['operation'].isin(['cp', 'scp']))
    ]
    
    if debug:
        print(f"Adding files from cp/scp operations in regular tasks: {len(cp_scp_tasks_df)} rows")
    
    for idx, row in cp_scp_tasks_df.iterrows():
        file_names_str = row['fileName'].strip()
        if file_names_str:
            # Parse comma-delimited file names
            file_names = [f.strip() for f in file_names_str.split(',') if f.strip()]
            for file_name in file_names:
                if file_name not in file_size_dict:
                    file_size_dict[file_name] = row['aggregateFilesizeMBtask']
                elif file_name in file_size_dict:
                    # If multiple sizes exist for the same file, take the smallest one
                    current_size = file_size_dict[file_name]
                    new_size = row['aggregateFilesizeMBtask']
                    if new_size < current_size:
                        file_size_dict[file_name] = new_size
    
    if debug:
        print(f"File size dictionary created with {len(file_size_dict)} unique files")
        print(f"Sample file sizes: {dict(list(file_size_dict.items())[:5])}")
    
    # Step 3: Process each unique (taskName, operation) combination
    unique_combinations = result_df[['taskName', 'operation']].drop_duplicates()
    
    if debug:
        print(f"\nProcessing {len(unique_combinations)} unique (taskName, operation) combinations...")
    
    for _, combo in unique_combinations.iterrows():
        task_name = combo['taskName']
        operation = combo['operation']
        
        # Get all rows for this combination
        mask = (result_df['taskName'] == task_name) & (result_df['operation'] == operation)
        subset = result_df[mask]
        
        if debug:
            print(f"\nProcessing: {task_name} - {operation} ({len(subset)} rows)")
        
        if operation in ['read', 'write']:
            # For read and write operations: Group by unique taskPID and sum aggregateFilesizeMBtask values
            if 'taskPID' in subset.columns:
                task_pid_groups = subset.groupby('taskPID')
                total_sum = 0
                for task_pid, group in task_pid_groups:
                    group_sum = group['aggregateFilesizeMBtask'].sum()
                    total_sum += group_sum
                    if debug:
                        print(f"  taskPID {task_pid}: sum = {group_sum}")
                aggregate_value = total_sum
            else:
                # Fallback if taskPID column doesn't exist
                aggregate_value = subset['aggregateFilesizeMBtask'].sum()
                if debug:
                    print(f"  No taskPID column, using sum: {aggregate_value}")
        
        elif operation in ['cp', 'scp', 'none']:
            # For cp, scp, and none operations: Use file-based calculation
            if debug:
                print(f"  Using file-based calculation for {operation} operation")
            
            # Get unique file names from this subset
            unique_files = set()
            for _, row in subset.iterrows():
                file_names_str = row['fileName'].strip()
                if file_names_str:
                    # Parse comma-delimited file names for cp/scp operations
                    if operation in ['cp', 'scp']:
                        file_names = [f.strip() for f in file_names_str.split(',') if f.strip()]
                        unique_files.update(file_names)
                    else:
                        # For 'none' operations, treat as single file
                        unique_files.add(file_names_str)
            
            if debug:
                print(f"  Unique files in this subset: {unique_files}")
            
            # Calculate total size using the file size dictionary
            total_size = 0
            found_files = 0
            for file_name in unique_files:
                if file_name in file_size_dict:
                    total_size += file_size_dict[file_name]
                    found_files += 1
                    if debug:
                        print(f"    File '{file_name}': {file_size_dict[file_name]} MB")
                else:
                    if debug:
                        print(f"    File '{file_name}': NOT FOUND in dictionary")
            
            if debug:
                print(f"  Found {found_files}/{len(unique_files)} files in dictionary")
                print(f"  Total size: {total_size}")
            
            # For cp, scp, and none operations: Use total_size as base aggregate value
            aggregate_value = total_size
            
            if debug:
                print(f"  Base aggregate value: {aggregate_value}")
        
        else:
            # For other operations, use sum
            aggregate_value = subset['aggregateFilesizeMBtask'].sum()
            if debug:
                print(f"  Other operation '{operation}', using sum: {aggregate_value}")
        
        # Assign the calculated value to all rows in this subset
        result_df.loc[mask, 'aggregateFilesizeMB'] = aggregate_value
        
        if debug:
            print(f"  Final aggregateFilesizeMB for {task_name} - {operation}: {aggregate_value}")
    
    # Step 4: Multiply aggregateFilesizeMB by numNodes for cp and scp operations only
    if debug:
        print(f"\n=== Multiplying by numNodes for cp and scp operations ===")
    
    cp_scp_mask = result_df['operation'].isin(['cp', 'scp'])
    cp_scp_rows = result_df[cp_scp_mask]
    
    if debug:
        print(f"Found {len(cp_scp_rows)} rows with cp/scp operations")
        if len(cp_scp_rows) > 0:
            print(f"Sample cp/scp rows before multiplication:")
            sample_cp_scp = cp_scp_rows[['taskName', 'operation', 'aggregateFilesizeMB', 'numNodes']].head(5)
            print(sample_cp_scp.to_string())
    
    # Multiply aggregateFilesizeMB by numNodes for cp and scp operations
    result_df.loc[cp_scp_mask, 'aggregateFilesizeMB'] = (
        result_df.loc[cp_scp_mask, 'aggregateFilesizeMB'] * result_df.loc[cp_scp_mask, 'numNodes']
    )
    
    if debug:
        if len(cp_scp_rows) > 0:
            print(f"Sample cp/scp rows after multiplication:")
            sample_cp_scp_after = result_df[cp_scp_mask][['taskName', 'operation', 'aggregateFilesizeMB', 'numNodes']].head(5)
            print(sample_cp_scp_after.to_string())
    
    if debug:
        print(f"\n=== Summary ===")
        print(f"Updated {len(result_df)} rows with aggregateFilesizeMB values")
        print(f"Sample results:")
        sample_results = result_df[['taskName', 'operation', 'aggregateFilesizeMB', 'numNodes']].head(10)
        print(sample_results.to_string())
    
    return result_df