#!/usr/bin/env python3
"""
Test script for the workflow_results_exporter module.
Demonstrates how to export producer-consumer results to CSV format.
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
    create_detailed_producer_consumer_report,
    analyze_storage_distribution,
    print_storage_analysis
)


def create_sample_spm_results():
    """Create sample SPM results for testing."""
    
    # Sample workflow DataFrame
    wf_data = {
        'taskName': ['task1', 'task2', 'task3', 'task4'],
        'stageOrder': [1, 1, 2, 3],
        'inputSize': [100, 200, 300, 400],
        'outputSize': [150, 250, 350, 450]
    }
    wf_df = pd.DataFrame(wf_data)
    
    # Sample SPM results
    spm_results = {
        'input:task1': {
            'best_storage_type': 'lustre',
            'best_parallelism': 'lustre_4_8',
            'best_rank': 2.5
        },
        'task1:task2': {
            'best_storage_type': 'nvme',
            'best_parallelism': 'nvme_8_16',
            'best_rank': 1.8
        },
        'task2:task3': {
            'rank': {
                'lustre_4_8': [2.1, 2.3, 2.0],
                'nvme_8_16': [1.5, 1.7, 1.6],
                'ssd_16_32': [1.9, 2.0, 1.8]
            }
        },
        'task3:task4': {
            'best_storage_type': 'ssd',
            'best_parallelism': 'ssd_16_32',
            'best_rank': 3.2
        }
    }
    
    return spm_results, wf_df


def test_extract_results():
    """Test extracting producer-consumer results."""
    print("=== Testing extract_producer_consumer_results ===")
    
    spm_results, wf_df = create_sample_spm_results()
    
    results_df = extract_producer_consumer_results(spm_results, wf_df)
    
    print(f"Extracted {len(results_df)} producer-consumer pairs")
    print("\nResults DataFrame:")
    print(results_df)
    
    print(f"\nColumns: {list(results_df.columns)}")
    print(f"Data types: {results_df.dtypes.to_dict()}")
    
    return results_df


def test_save_to_csv(results_df):
    """Test saving results to CSV."""
    print("\n=== Testing save_producer_consumer_results ===")
    
    # Create sample SPM results for the save function
    spm_results, wf_df = create_sample_spm_results()
    
    # Save to CSV
    csv_path = save_producer_consumer_results(
        spm_results, 
        wf_df, 
        workflow_name="test_workflow",
        output_dir="../analysis_data",
        filename="test_producer_consumer_results.csv"
    )
    
    print(f"CSV saved to: {csv_path}")
    
    # Verify the file was created and has correct content
    if os.path.exists(csv_path):
        saved_df = pd.read_csv(csv_path)
        print(f"Saved file contains {len(saved_df)} rows")
        print("First few rows of saved file:")
        print(saved_df.head())
    
    return csv_path


def test_detailed_report():
    """Test creating detailed report."""
    print("\n=== Testing create_detailed_producer_consumer_report ===")
    
    spm_results, wf_df = create_sample_spm_results()
    
    report_path = create_detailed_producer_consumer_report(
        spm_results,
        wf_df,
        workflow_name="test_workflow",
        output_dir="../analysis_data"
    )
    
    print(f"Detailed report saved to: {report_path}")
    
    # Show first few lines of the report
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            lines = f.readlines()[:20]  # First 20 lines
            print("\nFirst 20 lines of report:")
            for line in lines:
                print(line.rstrip())


def test_storage_analysis(results_df):
    """Test storage distribution analysis."""
    print("\n=== Testing analyze_storage_distribution ===")
    
    analysis = analyze_storage_distribution(results_df)
    
    print("Storage Distribution Analysis:")
    print(f"Analysis keys: {list(analysis.keys())}")
    
    if 'storage_distribution' in analysis:
        print("\nStorage Type Usage:")
        for storage, count in analysis['storage_distribution'].items():
            print(f"  {storage}: {count} pairs")
    
    if 'spm_by_storage' in analysis:
        print("\nSPM Statistics by Storage Type:")
        for storage, stats in analysis['spm_by_storage'].items():
            print(f"  {storage}: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")
    
    # Test the print function
    print("\n=== Testing print_storage_analysis ===")
    print_storage_analysis(results_df)


def main():
    """Main test function."""
    print("Testing workflow_results_exporter module")
    print("=" * 50)
    
    try:
        # Test 1: Extract results
        results_df = test_extract_results()
        
        # Test 2: Save to CSV
        csv_path = test_save_to_csv(results_df)
        
        # Test 3: Create detailed report
        test_detailed_report()
        
        # Test 4: Storage analysis
        test_storage_analysis(results_df)
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print(f"Generated files:")
        print(f"  - CSV: {csv_path}")
        print(f"  - Report: ../analysis_data/test_workflow_producer_consumer_detailed_report.txt")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 