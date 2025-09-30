import argparse
import sys
import math
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd


def interpolate_transfer_rate(data_points: List[Tuple[float, float]], target_tasks: float) -> float:
    """
    Interpolate transfer rate for a given number of tasks using nearest neighbors.
    
    Args:
        data_points: List of (tasksPerNode, transferRate) tuples sorted by tasksPerNode
        target_tasks: Target number of tasks for interpolation
        
    Returns:
        Interpolated transfer rate
    """
    if not data_points:
        return 0.0
    
    # Sort by tasksPerNode to ensure proper ordering
    data_points = sorted(data_points, key=lambda x: x[0])
    
    # If only one data point, return its value
    if len(data_points) == 1:
        return data_points[0][1]
    
    # Find the two nearest neighbors
    tasks_values = [point[0] for point in data_points]
    transfer_rates = [point[1] for point in data_points]
    
    # If target is exactly at a data point, return that value
    if target_tasks in tasks_values:
        idx = tasks_values.index(target_tasks)
        return transfer_rates[idx]
    
    # Find the two nearest neighbors for interpolation
    if target_tasks <= tasks_values[0]:
        # Extrapolate below the minimum
        if len(data_points) >= 2:
            # Use first two points for extrapolation
            x1, y1 = data_points[0]
            x2, y2 = data_points[1]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            return y1 - slope * (x1 - target_tasks)
        else:
            return transfer_rates[0]
    
    elif target_tasks >= tasks_values[-1]:
        # Extrapolate above the maximum
        if len(data_points) >= 2:
            # Use last two points for extrapolation
            x1, y1 = data_points[-2]
            x2, y2 = data_points[-1]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            return y2 + slope * (target_tasks - x2)
        else:
            return transfer_rates[-1]
    
    else:
        # Interpolate between two points
        for i in range(len(tasks_values) - 1):
            if tasks_values[i] <= target_tasks <= tasks_values[i + 1]:
                x1, y1 = data_points[i]
                x2, y2 = data_points[i + 1]
                if x2 == x1:
                    return y1
                # Linear interpolation
                slope = (y2 - y1) / (x2 - x1)
                return y1 + slope * (target_tasks - x1)
    
    # Fallback (should not reach here)
    return transfer_rates[0]


def calculate_accuracy(actual: float, predicted: float) -> float:
    """
    Calculate accuracy as 1 - relative error, clamped between 0 and 1.
    
    Args:
        actual: Actual transfer rate
        predicted: Predicted transfer rate
        
    Returns:
        Accuracy value between 0 and 1
    """
    if actual == 0:
        return 1.0 if predicted == 0 else 0.0
    
    relative_error = abs(actual - predicted) / abs(actual)
    accuracy = max(0.0, 1.0 - relative_error)
    return accuracy


def evaluate_operation_accuracy(df: pd.DataFrame, operation: str) -> List[float]:
    """
    Evaluate accuracy for a specific operation by testing interpolation on each data point.
    
    Args:
        df: Filtered dataframe for the operation
        operation: Operation name ('read' or 'write')
        
    Returns:
        List of accuracy values for each test point
    """
    accuracies = []
    
    # Group by unique combinations of other parameters (excluding tasksPerNode)
    # We'll test interpolation for each unique combination
    group_cols = ['aggregateFilesizeMB', 'numNodes', 'transferSize', 'storageType']
    unique_groups = df.groupby(group_cols)
    
    print(f"\nEvaluating {operation} operation with {len(unique_groups)} unique parameter combinations...")
    
    for group_key, group_df in unique_groups:
        if len(group_df) < 2:  # Need at least 2 points for interpolation
            continue
            
        # Sort by tasksPerNode to create a line of data points
        group_df_sorted = group_df.sort_values('tasksPerNode')
        data_points = list(zip(group_df_sorted['tasksPerNode'], group_df_sorted['trMiB']))
        
        print(f"  Group {group_key}: {len(data_points)} data points")
        print(f"    tasksPerNode range: {group_df_sorted['tasksPerNode'].min()} - {group_df_sorted['tasksPerNode'].max()}")
        print(f"    trMiB range: {group_df_sorted['trMiB'].min():.2f} - {group_df_sorted['trMiB'].max():.2f}")
        
        # Test each data point as ground truth
        for i, (test_tasks, test_rate) in enumerate(data_points):
            # Create training data by excluding the test point
            training_points = [(tasks, rate) for j, (tasks, rate) in enumerate(data_points) if j != i]
            
            if len(training_points) < 1:  # Need at least 1 point for extrapolation
                continue

            # Interpolate using the training points
            predicted_rate = interpolate_transfer_rate(training_points, test_tasks)
            
            # Calculate accuracy
            accuracy = calculate_accuracy(test_rate, predicted_rate)
            accuracies.append(accuracy)
            
            print(f"    Test point {i+1}: tasks={test_tasks}, actual={test_rate:.2f}, predicted={predicted_rate:.2f}, accuracy={accuracy:.3f}")
    
    return accuracies


def evaluate_pattern(df: pd.DataFrame, pattern_name: str, transfer_size: float, num_nodes: int, aggregate_file_size: float, storage_type: str) -> Dict:
    """
    Evaluate interpolation accuracy for a specific I/O pattern.
    
    Args:
        df: Full IOR dataframe
        pattern_name: Name of the pattern for display
        transfer_size: Transfer size to filter by
        num_nodes: Number of nodes to filter by
        aggregate_file_size: Aggregate file size to filter by
        storage_type: Storage type to filter by
        
    Returns:
        Dictionary containing accuracy results for this pattern
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING PATTERN: {pattern_name}")
    print(f"{'='*80}")
    print(f"Storage Type: {storage_type}")
    print(f"Transfer Size: {transfer_size}")
    print(f"Number of Nodes: {num_nodes}")
    print(f"Aggregate File Size: {aggregate_file_size}")
    
    # Filter for the specific pattern parameters including storage type
    df_pattern = df[
        (df['storageType'] == storage_type) &
        (df['transferSize'] == transfer_size) &
        (df['numNodes'] == num_nodes) &
        (df['aggregateFilesizeMB'] == aggregate_file_size)
    ].copy()
    
    print(f"Data points for this pattern: {len(df_pattern)} rows")
    
    if len(df_pattern) == 0:
        print("No data found for this pattern.")
        return {
            'pattern_name': pattern_name,
            'storage_type': storage_type,
            'transfer_size': transfer_size,
            'num_nodes': num_nodes,
            'aggregate_file_size': aggregate_file_size,
            'read_results': {'accuracies': [], 'mean_accuracy': 0.0, 'std_accuracy': 0.0, 'count': 0},
            'write_results': {'accuracies': [], 'mean_accuracy': 0.0, 'std_accuracy': 0.0, 'count': 0},
            'overall_accuracy': 0.0,
            'total_test_points': 0
        }
    
    # Check available operations
    available_ops = df_pattern['operation'].unique()
    print(f"Available operations: {sorted(available_ops)}")
    
    # Split into read and write operations
    read_df = df_pattern[df_pattern['operation'] == 'read'].copy()
    write_df = df_pattern[df_pattern['operation'] == 'write'].copy()
    
    print(f"Read operations: {len(read_df)} rows")
    print(f"Write operations: {len(write_df)} rows")
    
    # Evaluate accuracy for each operation
    read_results = {'accuracies': [], 'mean_accuracy': 0.0, 'std_accuracy': 0.0, 'count': 0}
    write_results = {'accuracies': [], 'mean_accuracy': 0.0, 'std_accuracy': 0.0, 'count': 0}
    
    if len(read_df) > 0:
        print(f"\n--- Evaluating READ operations for {pattern_name} ---")
        read_accuracies = evaluate_operation_accuracy(read_df, 'read')
        if read_accuracies:
            read_results = {
                'accuracies': read_accuracies,
                'mean_accuracy': np.mean(read_accuracies),
                'std_accuracy': np.std(read_accuracies),
                'count': len(read_accuracies)
            }
    
    if len(write_df) > 0:
        print(f"\n--- Evaluating WRITE operations for {pattern_name} ---")
        write_accuracies = evaluate_operation_accuracy(write_df, 'write')
        if write_accuracies:
            write_results = {
                'accuracies': write_accuracies,
                'mean_accuracy': np.mean(write_accuracies),
                'std_accuracy': np.std(write_accuracies),
                'count': len(write_accuracies)
            }
    
    # Calculate overall accuracy for this pattern
    all_pattern_accuracies = read_results['accuracies'] + write_results['accuracies']
    overall_accuracy = np.mean(all_pattern_accuracies) if all_pattern_accuracies else 0.0
    
    # Print pattern results
    print(f"\n--- RESULTS FOR {pattern_name} ---")
    print(f"READ Operations:")
    if read_results['count'] > 0:
        print(f"  Number of test points: {read_results['count']}")
        print(f"  Mean accuracy: {read_results['mean_accuracy']:.4f}")
        print(f"  Std accuracy: {read_results['std_accuracy']:.4f}")
        print(f"  Min accuracy: {min(read_results['accuracies']):.4f}")
        print(f"  Max accuracy: {max(read_results['accuracies']):.4f}")
    else:
        print("  No data available")
    
    print(f"WRITE Operations:")
    if write_results['count'] > 0:
        print(f"  Number of test points: {write_results['count']}")
        print(f"  Mean accuracy: {write_results['mean_accuracy']:.4f}")
        print(f"  Std accuracy: {write_results['std_accuracy']:.4f}")
        print(f"  Min accuracy: {min(write_results['accuracies']):.4f}")
        print(f"  Max accuracy: {max(write_results['accuracies']):.4f}")
    else:
        print("  No data available")
    
    print(f"OVERALL for {pattern_name}:")
    print(f"  Total test points: {len(all_pattern_accuracies)}")
    print(f"  Overall accuracy: {overall_accuracy:.4f}")
    
    return {
        'pattern_name': pattern_name,
        'storage_type': storage_type,
        'transfer_size': transfer_size,
        'num_nodes': num_nodes,
        'aggregate_file_size': aggregate_file_size,
        'read_results': read_results,
        'write_results': write_results,
        'overall_accuracy': overall_accuracy,
        'total_test_points': len(all_pattern_accuracies)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate interpolation accuracy on IOR benchmark data for multiple I/O patterns with different storage types.")
    parser.add_argument("--input", default=None, help="Path to IOR benchmark data (CSV). Defaults to ../../perf_profiles/updated_master_ior_df.csv")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Default input path
    if args.input is None:
        import os
        here = os.path.dirname(__file__)
        input_path = os.path.normpath(os.path.join(here, "../../perf_profiles/updated_master_ior_df.csv"))
    else:
        input_path = args.input
    
    # Load data
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded data with {len(df)} rows from {input_path}")
    except Exception as e:
        print(f"Failed to read input CSV '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Check available storage types
    available_storage_types = df['storageType'].unique()
    print(f"Available storage types: {available_storage_types}")
    
    # Define the four I/O patterns to test
    patterns = [
        {
            'name': 'Pattern 1 (Most Common)',
            'storage_type': 'ssd',
            'transfer_size': 4096.0,
            'num_nodes': 4,
            'aggregate_file_size': 51200.0
        },
        {
            'name': 'Pattern 2 (Large Transfer)',
            'storage_type': 'beegfs',
            'transfer_size': 1048576.0,
            'num_nodes': 8,
            'aggregate_file_size': 51200.0
        },
        {
            'name': 'Pattern 3 (Small File)',
            'storage_type': 'beegfs',
            'transfer_size': 16777216.0,  # Fixed the typo from 167772161048576
            'num_nodes': 2,
            'aggregate_file_size': 5120.0
        },
        {
            'name': 'Pattern 4 (High Node Count)',
            'storage_type': 'ssd',
            'transfer_size': 1048576.0,
            'num_nodes': 16,
            'aggregate_file_size': 1024.0
        }
    ]
    
    # Evaluate each pattern
    all_results = []
    all_accuracies = []
    
    for pattern in patterns:
        result = evaluate_pattern(
            df,  # Use full dataframe instead of filtered
            pattern['name'],
            pattern['transfer_size'],
            pattern['num_nodes'],
            pattern['aggregate_file_size'],
            pattern['storage_type']
        )
        all_results.append(result)
        all_accuracies.extend(result['read_results']['accuracies'])
        all_accuracies.extend(result['write_results']['accuracies'])
    
    # Print comprehensive summary
    print(f"\n{'='*100}")
    print("COMPREHENSIVE ACCURACY SUMMARY")
    print(f"{'='*100}")
    print()
    
    # Individual pattern summaries
    for result in all_results:
        print(f"{result['pattern_name']}:")
        print(f"  Storage Type: {result['storage_type']}")
        print(f"  Transfer Size: {result['transfer_size']}")
        print(f"  Number of Nodes: {result['num_nodes']}")
        print(f"  Aggregate File Size: {result['aggregate_file_size']}")
        print(f"  READ Accuracy: {result['read_results']['mean_accuracy']:.4f} ({result['read_results']['count']} points)")
        print(f"  WRITE Accuracy: {result['write_results']['mean_accuracy']:.4f} ({result['write_results']['count']} points)")
        print(f"  Overall Pattern Accuracy: {result['overall_accuracy']:.4f} ({result['total_test_points']} points)")
        print()
    
    # Overall summary across all patterns
    if all_accuracies:
        overall_mean = np.mean(all_accuracies)
        overall_std = np.std(all_accuracies)
        total_points = len(all_accuracies)
        
        print(f"OVERALL SUMMARY ACROSS ALL PATTERNS:")
        print(f"  Total test points: {total_points}")
        print(f"  Overall mean accuracy: {overall_mean:.4f}")
        print(f"  Overall std accuracy: {overall_std:.4f}")
        print(f"  Min accuracy: {min(all_accuracies):.4f}")
        print(f"  Max accuracy: {max(all_accuracies):.4f}")
    else:
        print("No accuracy data available across all patterns.")


if __name__ == "__main__":
    main()
