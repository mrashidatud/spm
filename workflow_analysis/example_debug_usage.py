#!/usr/bin/env python3
"""
Example script demonstrating the debug parameter usage in calculate_spm_for_workflow.

This script shows how to control verbose output when calculating SPM values.
"""

import pandas as pd
from modules.workflow_spm_calculator import calculate_spm_for_workflow

def create_sample_workflow_data():
    """Create a minimal sample workflow DataFrame for testing."""
    data = {
        'taskName': ['task1', 'task2', 'task3'],
        'stageOrder': [0, 1, 2],
        'operation': [1, 0, 1],  # read, write, read
        'taskPID': ['pid1', 'pid2', 'pid3'],
        'fileName': ['file1.txt', 'file2.txt', 'file3.txt'],
        'aggregateFilesizeMB': [100, 200, 150],
        'parallelism': [1, 2, 1],
        'opCount': [1, 1, 1],
        'prevTask': ['', 'task1', 'task2']
    }
    return pd.DataFrame(data)

def main():
    """Demonstrate debug parameter usage."""
    print("=== Debug Parameter Example ===\n")
    
    # Create sample data
    wf_df = create_sample_workflow_data()
    print(f"Sample workflow data shape: {wf_df.shape}")
    print(f"Tasks: {list(wf_df['taskName'])}")
    print()
    
    # Example 1: Run with debug=False (minimal output)
    print("1. Running with debug=False (minimal output):")
    print("-" * 50)
    try:
        spm_results = calculate_spm_for_workflow(wf_df, debug=False)
        print(f"✓ SPM calculation completed successfully")
        print(f"  Found {len(spm_results)} producer-consumer pairs")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # Example 2: Run with debug=True (verbose output)
    print("2. Running with debug=True (verbose output):")
    print("-" * 50)
    try:
        spm_results = calculate_spm_for_workflow(wf_df, debug=True)
        print(f"✓ SPM calculation completed successfully")
        print(f"  Found {len(spm_results)} producer-consumer pairs")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    print("=== Summary ===")
    print("• debug=False: Minimal output, suitable for production use")
    print("• debug=True: Verbose output, useful for debugging and development")
    print("• Default is debug=False for clean output")

if __name__ == "__main__":
    main() 