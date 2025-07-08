#!/usr/bin/env python3
"""
Test script to check interpolation function.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from modules.workflow_interpolation import estimate_transfer_rates_for_workflow
from modules.workflow_config import STORAGE_LIST

def test_interpolation():
    """Test the interpolation function."""
    print("Loading IOR data...")
    df_ior = pd.read_csv('../../perf_profiles/updated_master_ior_df.csv')
    print(f"IOR data shape: {df_ior.shape}")
    
    # Create test workflow data
    wf_df = pd.DataFrame({
        'operation': [0, 1],  # 0=write, 1=read
        'aggregateFilesizeMB': [100, 200],
        'numNodes': [1, 1],
        'transferSize': [1024, 2048]
    })
    print(f"Test workflow data shape: {wf_df.shape}")
    
    # Test interpolation with limited parallelism
    print("Testing interpolation with limited parallelism...")
    allowed_parallelism = [1, 3, 6, 12]  # DDMD workflow allowed parallelism
    num_nodes_list = [1, 2, 4]  # DDMD workflow allowed nodes
    
    result = estimate_transfer_rates_for_workflow(wf_df, df_ior, STORAGE_LIST, 
                                                allowed_parallelism, num_nodes_list)
    
    # Check results
    estimated_cols = [col for col in result.columns if col.startswith('estimated_trMiB_')]
    print(f"Estimated columns: {len(estimated_cols)}")
    
    print("Sample estimated values:")
    for col in estimated_cols[:5]:
        value = result[col].iloc[0]
        print(f"{col}: {value}")
    
    # Check if any values are non-zero
    non_zero_count = 0
    for col in estimated_cols:
        if result[col].iloc[0] > 0:
            non_zero_count += 1
    
    print(f"Non-zero estimated values: {non_zero_count}/{len(estimated_cols)}")

if __name__ == "__main__":
    test_interpolation() 