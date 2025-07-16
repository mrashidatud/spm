#!/usr/bin/env python3
"""
Test script for IOR utility functions
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ior_utils import *

def test_ior_utils():
    """Test the IOR utility functions"""
    
    # Define the data directory
    data_dir = "ior_data"
    
    print("=== Testing IOR Utility Functions ===\n")
    
    # Test 1: Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found!")
        print("Please make sure you're running this script from the perf_profiles directory.")
        return False
    
    print(f"✓ Data directory '{data_dir}' found")
    
    # Test 2: Collect all data
    print("\n1. Collecting all IOR benchmark data...")
    try:
        df = collect_ior_data(data_dir)
        print(f"✓ Successfully collected {len(df)} records")
        print_data_overview(df)
    except Exception as e:
        print(f"✗ Error collecting data: {e}")
        return False
    
    # Test 3: Test filtering
    print("\n2. Testing data filtering...")
    try:
        # Filter for beegfs only
        beegfs_df = filter_data_by_conditions(df, storage_type='beegfs')
        print(f"✓ Filtered beegfs data: {len(beegfs_df)} records")
        
        # Filter for specific transfer size
        transfer_size_64mb = 64 * 1024 * 1024  # 64MB in bytes
        filtered_df = filter_data_by_conditions(df, transfer_size=transfer_size_64mb)
        print(f"✓ Filtered 64MB transfer size data: {len(filtered_df)} records")
        
        # Filter for specific number of nodes
        one_node_df = filter_data_by_conditions(df, num_nodes=1)
        print(f"✓ Filtered 1-node data: {len(one_node_df)} records")
        
    except Exception as e:
        print(f"✗ Error filtering data: {e}")
        return False
    
    # Test 4: Test storage type filtering during collection
    print("\n3. Testing storage type filtering during collection...")
    try:
        selected_storage_types = ['beegfs', 'ssd']
        filtered_df = collect_ior_data(data_dir, storage_types=selected_storage_types)
        print(f"✓ Collected data for {selected_storage_types}: {len(filtered_df)} records")
        print(f"  Storage types in filtered data: {sorted(filtered_df['storageType'].unique())}")
    except Exception as e:
        print(f"✗ Error with storage type filtering: {e}")
        return False
    
    # Test 5: Save and load data
    print("\n4. Testing save and load functionality...")
    try:
        test_filename = "test_master_ior_df.csv"
        save_master_ior_df(df, test_filename)
        
        # Load the data back
        loaded_df = load_master_ior_df(test_filename)
        print(f"✓ Successfully saved and loaded data: {len(loaded_df)} records")
        
        # Clean up test file
        if os.path.exists(test_filename):
            os.remove(test_filename)
            print(f"✓ Cleaned up test file: {test_filename}")
            
    except Exception as e:
        print(f"✗ Error with save/load: {e}")
        return False
    
    # Test 6: Test summary statistics
    print("\n5. Testing summary statistics...")
    try:
        summary_stats = get_summary_statistics(df)
        print(f"✓ Generated summary statistics with {len(summary_stats)} groups")
        print("  Summary statistics shape:", summary_stats.shape)
    except Exception as e:
        print(f"✗ Error generating summary statistics: {e}")
        return False
    
    # Test 7: Test path extraction
    print("\n6. Testing path extraction...")
    try:
        # Find a sample JSON file
        sample_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    sample_files.append(os.path.join(root, file))
                    break
            if sample_files:
                break
        
        if sample_files:
            sample_file = sample_files[0]
            storage_type, num_nodes = extract_storage_info_from_path(sample_file)
            print(f"✓ Extracted storage info from {os.path.basename(sample_file)}:")
            print(f"  Storage type: {storage_type}")
            print(f"  Number of nodes: {num_nodes}")
        else:
            print("⚠ No sample JSON files found for testing")
            
    except Exception as e:
        print(f"✗ Error with path extraction: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("The IOR utility functions are working correctly.")
    print("\nYou can now:")
    print("1. Run the Jupyter notebook for visualization")
    print("2. Use the utility functions in your own scripts")
    print("3. Filter data by storage types as needed")
    
    return True

if __name__ == "__main__":
    success = test_ior_utils()
    if not success:
        sys.exit(1)