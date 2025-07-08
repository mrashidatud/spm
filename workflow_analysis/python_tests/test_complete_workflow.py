#!/usr/bin/env python3
"""
Test script to run the complete workflow analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from modules.workflow_data_utils import load_workflow_data, calculate_io_time_breakdown
from modules.workflow_interpolation import estimate_transfer_rates_for_workflow, calculate_aggregate_filesize_per_node
from modules.workflow_spm_calculator import calculate_spm_for_workflow
from modules.workflow_config import STORAGE_LIST, TEST_CONFIGS

def test_complete_workflow():
    """Test the complete workflow analysis."""
    WORKFLOW_NAME = "ddmd_4n_l"
    
    print("=== Step 1: Loading workflow data ===")
    wf_df, task_order_dict, all_wf_dict = load_workflow_data(WORKFLOW_NAME)
    print(f"Workflow data loaded: {len(wf_df)} records")
    print(f"Task definitions: {len(task_order_dict)}")
    print(f"Unique tasks: {list(wf_df['taskName'].unique())}")
    
    if len(wf_df) == 0:
        print("ERROR: No workflow data loaded!")
        return
    
    print("\n=== Step 2: Calculating I/O time breakdown ===")
    config = TEST_CONFIGS[WORKFLOW_NAME]
    num_nodes_list = config["NUM_NODES_LIST"]
    task_name_to_parallelism = {task: info['parallelism'] for task, info in task_order_dict.items()}
    io_breakdown = calculate_io_time_breakdown(wf_df, task_name_to_parallelism, num_nodes_list)
    print(f"I/O breakdown completed: {io_breakdown['total_io_time']:.2f} seconds total")
    
    print("\n=== Step 3: Calculating aggregate file size per node ===")
    wf_df = calculate_aggregate_filesize_per_node(wf_df)
    print("Aggregate file size calculation completed")
    
    print("\n=== Step 4: Estimating transfer rates ===")
    IOR_DATA_PATH = "../../perf_profiles/updated_master_ior_df.csv"
    if pd.io.common.file_exists(IOR_DATA_PATH):
        df_ior = pd.read_csv(IOR_DATA_PATH)
        print(f"Loaded {len(df_ior)} IOR benchmark records")
        
        # Get allowed_parallelism from config, with fallback to default
        allowed_parallelism = config.get("ALLOWED_PARALLELISM", None)
        wf_df = estimate_transfer_rates_for_workflow(wf_df, df_ior, STORAGE_LIST, allowed_parallelism)
        estimated_cols = [col for col in wf_df.columns if col.startswith('estimated_trMiB_')]
        print(f"Estimated transfer rate columns: {len(estimated_cols)}")
        
        # Check if estimated values are non-zero
        non_zero_count = 0
        total_values = 0
        for col in estimated_cols:
            non_zero_values = (wf_df[col] > 0).sum()
            total_values += len(wf_df)
            non_zero_count += non_zero_values
        
        print(f"Non-zero estimated values: {non_zero_count}/{total_values} ({non_zero_count/total_values*100:.1f}%)")
    else:
        print(f"Warning: IOR data file not found at {IOR_DATA_PATH}")
    
    print("\n=== Step 5: Calculating SPM values ===")
    spm_results = calculate_spm_for_workflow(wf_df)
    print(f"SPM calculation completed: {len(spm_results)} producer-consumer pairs")
    for pair in spm_results.keys():
        print(f"  - {pair}")
    
    print("\n=== Workflow analysis completed successfully! ===")

if __name__ == "__main__":
    test_complete_workflow() 