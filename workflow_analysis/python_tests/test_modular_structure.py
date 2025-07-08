"""
Test script to verify the modular workflow analysis structure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        from modules.workflow_config import DEFAULT_WF, TEST_CONFIGS, STORAGE_LIST
        print("✓ workflow_config imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import workflow_config: {e}")
        return False
    
    try:
        from modules.workflow_data_utils import load_workflow_data, calculate_io_time_breakdown
        print("✓ workflow_data_utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import workflow_data_utils: {e}")
        return False
    
    try:
        from modules.workflow_interpolation import estimate_transfer_rates_for_workflow
        print("✓ workflow_interpolation imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import workflow_interpolation: {e}")
        return False
    
    try:
        from modules.workflow_spm_calculator import calculate_spm_for_workflow
        print("✓ workflow_spm_calculator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import workflow_spm_calculator: {e}")
        return False
    
    try:
        from modules.workflow_visualization import plot_all_visualizations
        print("✓ workflow_visualization imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import workflow_visualization: {e}")
        return False
    
    try:
        from modules.workflow_analysis_main import run_workflow_analysis
        print("✓ workflow_analysis_main imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import workflow_analysis_main: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration values."""
    print("\nTesting configuration...")
    
    from modules.workflow_config import DEFAULT_WF, TEST_CONFIGS, STORAGE_LIST
    
    print(f"Default workflow: {DEFAULT_WF}")
    print(f"Available workflows: {list(TEST_CONFIGS.keys())}")
    print(f"Storage types: {STORAGE_LIST}")
    
    # Test that default workflow exists
    if DEFAULT_WF in TEST_CONFIGS:
        print("✓ Default workflow configuration exists")
    else:
        print("✗ Default workflow configuration missing")
        return False
    
    return True


def test_data_utils():
    """Test data utilities functions."""
    print("\nTesting data utilities...")
    
    from modules.workflow_data_utils import transform_store_code, decode_store_code, bytes_to_mb
    
    # Test storage code transformation
    assert transform_store_code("localssd") == 0
    assert transform_store_code("beegfs") == 1
    assert transform_store_code("tmpfs") == 3
    assert transform_store_code("nfs") == 4
    print("✓ Storage code transformation works")
    
    # Test storage code decoding
    assert decode_store_code(0) == "localssd"
    assert decode_store_code(1) == "beegfs"
    assert decode_store_code(3) == "tmpfs"
    assert decode_store_code(4) == "nfs"
    print("✓ Storage code decoding works")
    
    # Test bytes to MB conversion
    assert abs(bytes_to_mb("1024 KiB") - 1.0) < 0.001
    assert abs(bytes_to_mb("1 MiB") - 1.0) < 0.001
    assert abs(bytes_to_mb("1 GiB") - 1024.0) < 0.001
    print("✓ Bytes to MB conversion works")
    
    return True


def test_interpolation():
    """Test interpolation functions."""
    print("\nTesting interpolation functions...")
    
    from modules.workflow_interpolation import calculate_4d_interpolation_with_extrapolation
    
    # Create sample IOR data
    sample_data = pd.DataFrame({
        'operation': [0, 0, 0, 0, 1, 1, 1, 1],
        'aggregateFilesizeMB': [100, 100, 200, 200, 100, 100, 200, 200],
        'numNodes': [1, 2, 1, 2, 1, 2, 1, 2],
        'tasksPerNode': [1, 1, 1, 1, 1, 1, 1, 1],
        'transferSize': [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
        'trMiB': [100, 200, 150, 300, 80, 160, 120, 240]
    })
    
    # Test interpolation
    try:
        result, slope = calculate_4d_interpolation_with_extrapolation(
            sample_data, 0, 150, 1, 1, 1024, 'tasksPerNode', 'trMiB'
        )
        print(f"✓ Interpolation result: {result:.2f}, slope: {slope:.2f}")
    except Exception as e:
        print(f"✗ Interpolation failed: {e}")
        return False
    
    return True


def test_spm_calculator():
    """Test SPM calculator functions."""
    print("\nTesting SPM calculator...")
    
    from modules.workflow_spm_calculator import normalize_estT_values
    
    # Create sample SPM data
    sample_spm_data = {
        'test_pair': {
            'estT_prod': {'storage_1': [1.0, 2.0, 3.0]},
            'estT_cons': {'storage_1': [0.5, 1.0, 1.5]},
            'SPM': {'storage_1': [2.0, 2.0, 2.0]},
            'dsize_cons': {'storage_1': [100, 200, 300]},
            'dsize_prod': {'storage_1': [50, 100, 150]}
        }
    }
    
    try:
        normalized = normalize_estT_values(sample_spm_data)
        print("✓ SPM normalization works")
    except Exception as e:
        print(f"✗ SPM normalization failed: {e}")
        return False
    
    return True


def test_visualization():
    """Test visualization functions."""
    print("\nTesting visualization functions...")
    
    from modules.workflow_visualization import create_summary_report
    
    # Create sample data for testing
    sample_df = pd.DataFrame({
        'taskName': ['task1', 'task2'],
        'aggregateFilesizeMB': [100, 200],
        'totalTime': [10, 20],
        'opCount': [1000, 2000]
    })
    
    sample_spm_results = {
        'test_pair': {
            'best_storage_type': 'localssd',
            'best_parallelism': 'localssd_1p',
            'best_rank': 1.5,
            'avg_rank_by_storage': {'localssd': 1.5, 'beegfs': 2.0}
        }
    }
    
    sample_io_breakdown = {'task1': 10.0, 'task2': 20.0}
    
    try:
        create_summary_report(sample_df, sample_spm_results, sample_io_breakdown, 
                             save_path="../test_summary_report.txt")
        print("✓ Summary report generation works")
        
        # Clean up test file
        if os.path.exists("../test_summary_report.txt"):
            os.remove("../test_summary_report.txt")
    except Exception as e:
        print(f"✗ Summary report generation failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Testing modular workflow analysis structure...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_data_utils,
        test_interpolation,
        test_spm_calculator,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ Test {test.__name__} failed")
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Modular structure is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 