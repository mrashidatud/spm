#!/usr/bin/env python3
"""
Debug script to diagnose issues with filtered_spm_results producing empty CSV files.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.workflow_results_exporter import (
    extract_producer_consumer_results,
    save_producer_consumer_results,
    print_storage_analysis
)


def debug_filtered_spm_results(filtered_spm_results, wf_df):
    """
    Debug function to examine the structure of filtered_spm_results.
    """
    print("=== Debugging filtered_spm_results ===")
    
    # Check if filtered_spm_results is empty
    if not filtered_spm_results:
        print("❌ filtered_spm_results is empty or None")
        return
    
    print(f"✅ filtered_spm_results has {len(filtered_spm_results)} items")
    
    # Examine the structure of the first few items
    print("\n--- Structure Analysis ---")
    for i, (key, value) in enumerate(filtered_spm_results.items()):
        if i >= 3:  # Only show first 3 items
            break
        print(f"\nItem {i+1}: Key = '{key}'")
        print(f"  Type: {type(value)}")
        print(f"  Keys: {list(value.keys()) if isinstance(value, dict) else 'Not a dict'}")
        
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"    {k}: {type(v)} = {v}")
    
    # Check if it has the expected structure
    print("\n--- Expected Structure Check ---")
    has_best_storage = any('best_storage_type' in str(v) for v in filtered_spm_results.values())
    has_rank = any('rank' in str(v) for v in filtered_spm_results.values())
    
    print(f"Contains 'best_storage_type': {has_best_storage}")
    print(f"Contains 'rank': {has_rank}")
    
    # Try to extract results
    print("\n--- Attempting to Extract Results ---")
    try:
        results_df = extract_producer_consumer_results(filtered_spm_results, wf_df)
        print(f"✅ Successfully extracted {len(results_df)} rows")
        if not results_df.empty:
            print("Sample data:")
            print(results_df.head())
        else:
            print("❌ Extracted DataFrame is empty")
    except Exception as e:
        print(f"❌ Error extracting results: {e}")
        import traceback
        traceback.print_exc()


def create_alternative_extraction(filtered_spm_results, wf_df):
    """
    Create an alternative extraction method for different data structures.
    """
    print("\n=== Alternative Extraction Method ===")
    
    results_data = []
    
    for pair, data in filtered_spm_results.items():
        print(f"Processing pair: {pair}")
        print(f"  Data type: {type(data)}")
        print(f"  Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # Handle different possible structures
        if isinstance(data, dict):
            # Method 1: Direct storage and parallelism info
            if 'storage_type' in data:
                results_data.append({
                    'producer': pair.split(':')[0] if ':' in pair else 'unknown',
                    'producerStage': -1,  # Will be filled from wf_df
                    'consumer': pair.split(':')[1] if ':' in pair else 'unknown',
                    'consumerStage': -1,  # Will be filled from wf_df
                    'prodParallelism': data.get('prod_parallelism', np.nan),
                    'consParallelism': data.get('cons_parallelism', np.nan),
                    'p-c-Storage': data['storage_type'],
                    'p-c-SPM': data.get('spm_value', np.nan)
                })
            
            # Method 2: Nested structure with storage configurations
            elif 'storage_configs' in data:
                for config in data['storage_configs']:
                    results_data.append({
                        'producer': pair.split(':')[0] if ':' in pair else 'unknown',
                        'producerStage': -1,
                        'consumer': pair.split(':')[1] if ':' in pair else 'unknown',
                        'consumerStage': -1,
                        'prodParallelism': config.get('prod_par', np.nan),
                        'consParallelism': config.get('cons_par', np.nan),
                        'p-c-Storage': config.get('storage', 'unknown'),
                        'p-c-SPM': config.get('spm', np.nan)
                    })
            
            # Method 3: Simple key-value pairs
            else:
                # Try to infer structure from available keys
                storage_key = None
                spm_key = None
                
                for key in data.keys():
                    if 'storage' in key.lower():
                        storage_key = key
                    if 'spm' in key.lower() or 'rank' in key.lower():
                        spm_key = key
                
                if storage_key:
                    results_data.append({
                        'producer': pair.split(':')[0] if ':' in pair else 'unknown',
                        'producerStage': -1,
                        'consumer': pair.split(':')[1] if ':' in pair else 'unknown',
                        'consumerStage': -1,
                        'prodParallelism': np.nan,
                        'consParallelism': np.nan,
                        'p-c-Storage': data[storage_key],
                        'p-c-SPM': data.get(spm_key, np.nan) if spm_key else np.nan
                    })
    
    # Create DataFrame
    if results_data:
        results_df = pd.DataFrame(results_data)
        
        # Fill in stage information from wf_df
        task_stage_mapping = {}
        for _, row in wf_df.iterrows():
            task_name = row['taskName']
            stage_order = row['stageOrder']
            if task_name not in task_stage_mapping:
                task_stage_mapping[task_name] = stage_order
        
        for i, row in results_df.iterrows():
            if row['producer'] in task_stage_mapping:
                results_df.at[i, 'producerStage'] = task_stage_mapping[row['producer']]
            if row['consumer'] in task_stage_mapping:
                results_df.at[i, 'consumerStage'] = task_stage_mapping[row['consumer']]
        
        print(f"✅ Alternative extraction created {len(results_df)} rows")
        return results_df
    else:
        print("❌ No data extracted with alternative method")
        return pd.DataFrame()


def main():
    """
    Main debug function - you can call this with your actual data.
    """
    print("Debug script for filtered_spm_results")
    print("=" * 50)
    
    # This is a template - you'll need to replace with your actual data
    print("To use this debug script:")
    print("1. Copy the debug_filtered_spm_results() function")
    print("2. Call it with your actual filtered_spm_results and wf_df")
    print("3. Or call create_alternative_extraction() if the standard method fails")
    
    print("\nExample usage:")
    print("debug_filtered_spm_results(filtered_spm_results, wf_df)")
    print("alternative_df = create_alternative_extraction(filtered_spm_results, wf_df)")


if __name__ == "__main__":
    main() 