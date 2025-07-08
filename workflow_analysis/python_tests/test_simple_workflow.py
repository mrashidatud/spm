#!/usr/bin/env python3
"""
Simple test script to check workflow data loading.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from modules.workflow_data_utils import load_workflow_data
from modules.workflow_config import TEST_CONFIGS

def test_simple_workflow():
    """Test simple workflow data loading."""
    WORKFLOW_NAME = "ddmd_4n_l"
    
    print("=== Testing workflow data loading ===")
    try:
        wf_df, task_order_dict, all_wf_dict = load_workflow_data(WORKFLOW_NAME)
        print(f"✓ Workflow data loaded successfully")
        print(f"  - Records: {len(wf_df)}")
        print(f"  - Task definitions: {len(task_order_dict)}")
        print(f"  - Workflow dict entries: {len(all_wf_dict)}")
        
        if len(wf_df) > 0:
            print(f"  - Unique tasks: {list(wf_df['taskName'].unique())}")
            print(f"  - Operations: {list(wf_df['operation'].unique())}")
            print(f"  - Sample data shape: {wf_df.shape}")
            print("✓ All steps completed successfully!")
        else:
            print("⚠ Warning: No records loaded")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_workflow()