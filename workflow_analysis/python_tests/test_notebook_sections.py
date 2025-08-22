#!/usr/bin/env python3
"""
Test script to verify notebook sections work correctly.
"""

import pandas as pd
import numpy as np
import warnings
import os
import json

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.workflow_config import DEFAULT_WF, TEST_CONFIGS, STORAGE_LIST
from modules.workflow_data_utils import load_workflow_data, calculate_io_time_breakdown
from modules.workflow_interpolation import estimate_transfer_rates_for_workflow, calculate_aggregate_filesize_per_node
from modules.workflow_spm_calculator import calculate_spm_for_workflow, filter_storage_options, select_best_storage_and_parallelism
from modules.workflow_visualization import plot_all_visualizations

def test_notebook_sections():
    """Test all notebook sections."""
    WORKFLOW_NAME = "ddmd_4n_l"
    IOR_DATA_PATH = "../../perf_profiles/updated_master_ior_df.csv"
    
    print("=== Testing Notebook Sections ===\n")
    
    # Step 1: Load workflow data
    print("Step 1: Loading workflow data...")
    wf_df, task_order_dict, all_wf_dict = load_workflow_data(WORKFLOW_NAME)
    print(f"✓ Workflow data loaded: {len(wf_df)} records")
    print(f"✓ Task definitions: {len(task_order_dict)}")
    print(f"✓ Unique tasks: {list(wf_df['taskName'].unique())}")
    print(f"✓ Stages: {sorted(wf_df['stageOrder'].unique())}\n")
    
    # Step 2: Calculate I/O time breakdown
    print("Step 2: Calculating I/O time breakdown...")
    config = TEST_CONFIGS[WORKFLOW_NAME]
    num_nodes_list = config["NUM_NODES_LIST"]
    task_name_to_parallelism = {task: info['parallelism'] for task, info in task_order_dict.items()}
    io_breakdown = calculate_io_time_breakdown(wf_df, task_name_to_parallelism, num_nodes_list)
    print(f"✓ I/O breakdown completed: {io_breakdown['total_io_time']:.2f} seconds total\n")
    
    # Step 3: Calculate aggregate file size per node
    print("Step 3: Calculating aggregate file size per node...")
    wf_df = calculate_aggregate_filesize_per_node(wf_df)
    print("✓ Aggregate file size calculation completed\n")
    
    # Step 4: Estimate transfer rates
    print("Step 4: Estimating transfer rates...")
    if os.path.exists(IOR_DATA_PATH):
        df_ior = pd.read_csv(IOR_DATA_PATH)
        print(f"✓ Loaded {len(df_ior)} IOR benchmark records")
        
        # Get allowed parallelism and num_nodes from config
        allowed_parallelism = config.get("ALLOWED_PARALLELISM", None)
        
        wf_df = estimate_transfer_rates_for_workflow(wf_df, df_ior, STORAGE_LIST, allowed_parallelism)
        estimated_cols = [col for col in wf_df.columns if col.startswith('estimated_trMiB_')]
        print(f"✓ Estimated transfer rate columns: {len(estimated_cols)}")
        
        # Check non-zero values
        non_zero_count = 0
        total_values = 0
        for col in estimated_cols:
            non_zero_values = (wf_df[col] > 0).sum()
            total_values += len(wf_df)
            non_zero_count += non_zero_values
        
        print(f"✓ Non-zero estimated values: {non_zero_count}/{total_values} ({non_zero_count/total_values*100:.1f}%)\n")
    else:
        print(f"⚠ Warning: IOR data file not found at {IOR_DATA_PATH}\n")
    
    # Step 5: Calculate SPM values
    print("Step 5: Calculating SPM values...")
    spm_results = calculate_spm_for_workflow(wf_df, workflow_name=WORKFLOW_NAME)
    print(f"✓ SPM calculation completed: {len(spm_results)} producer-consumer pairs")
    for pair in spm_results.keys():
        print(f"  - {pair}")
    print()
    
    # Step 6: Filter and select best configurations
    if len(spm_results) > 0:
        print("Step 6: Filtering storage options...")
        filtered_spm_results = filter_storage_options(spm_results, WORKFLOW_NAME)
        print("✓ Storage options filtered")
        
        print("Step 6: Selecting best storage and parallelism...")
        best_results = select_best_storage_and_parallelism(spm_results, baseline=0)
        print("✓ Best configurations selected")
        for pair, config in best_results.items():
            print(f"  - {pair}: {config['best_storage_type']}, {config['best_parallelism']}")
        print()
    else:
        print("Step 6: Skipped - no SPM results available\n")
        filtered_spm_results = {}
        best_results = {}
    
    # Step 7: Save results
    print("Step 7: Saving results...")
    os.makedirs("../analysis_data", exist_ok=True)
    wf_df.to_csv(f'../analysis_data/{WORKFLOW_NAME}_workflow_data.csv', index=False)
    print(f"✓ Saved workflow data to: ../analysis_data/{WORKFLOW_NAME}_workflow_data.csv")
    
    if len(best_results) > 0:
        with open(f'../analysis_data/{WORKFLOW_NAME}_spm_results.json', 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            json.dump(best_results, f, default=convert_numpy, indent=2)
        print(f"✓ Saved SPM results to: ../analysis_data/{WORKFLOW_NAME}_spm_results.json")
    
    print("\n=== All notebook sections completed successfully! ===")

if __name__ == "__main__":
    test_notebook_sections() 