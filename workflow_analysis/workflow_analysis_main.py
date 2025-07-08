"""
Main workflow analysis script.
Orchestrates the entire workflow analysis process using modular functions.
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any

# Import our modules
from modules.workflow_config import DEFAULT_WF, TEST_CONFIGS, STORAGE_LIST
from modules.workflow_data_utils import (
    load_workflow_data, calculate_io_time_breakdown
)
from modules.workflow_interpolation import (
    estimate_transfer_rates_for_workflow, calculate_aggregate_filesize_per_node
)
from modules.workflow_spm_calculator import (
    calculate_spm_for_workflow, filter_storage_options,
    display_top_sorted_averaged_rank, select_best_storage_and_parallelism
)
from modules.workflow_visualization import plot_all_visualizations
from modules.workflow_results_exporter import save_producer_consumer_results, print_storage_analysis


def run_workflow_analysis(workflow_name: str = DEFAULT_WF, 
                         ior_data_path: str = "../perf_profiles/updated_master_ior_df.csv",
                         save_results: bool = True) -> Dict[str, Any]:
    """
    Run the complete workflow analysis pipeline.
    
    Parameters:
    - workflow_name: Name of the workflow to analyze
    - ior_data_path: Path to the IOR benchmark data
    - save_results: Whether to save results to files
    
    Returns:
    - Dict: Analysis results
    """
    print(f"Starting workflow analysis for: {workflow_name}")
    print("=" * 60)
    
    # Step 1: Load workflow data
    print("\n1. Loading workflow data...")
    wf_df, task_order_dict, all_wf_dict = load_workflow_data(workflow_name)
    
    # Get configuration for the workflow
    config = TEST_CONFIGS[workflow_name]
    num_nodes_list = config["NUM_NODES_LIST"]
    
    # Create task name to parallelism mapping
    task_name_to_parallelism = {task: info['parallelism'] for task, info in task_order_dict.items()}
    
    print(f"   Loaded {len(wf_df)} workflow records")
    print(f"   Found {len(task_order_dict)} task definitions")
    print(f"   Unique tasks: {list(wf_df['taskName'].unique())}")
    
    # Step 2: Calculate I/O time breakdown
    print("\n2. Calculating I/O time breakdown...")
    io_breakdown = calculate_io_time_breakdown(wf_df, task_name_to_parallelism, num_nodes_list)
    
    # Step 3: Calculate aggregate file size per node
    print("\n3. Calculating aggregate file size per node...")
    wf_df = calculate_aggregate_filesize_per_node(wf_df)
    
    # Step 4: Load IOR benchmark data
    print("\n4. Loading IOR benchmark data...")
    if not os.path.exists(ior_data_path):
        print(f"   Warning: IOR data file not found at {ior_data_path}")
        print("   Skipping transfer rate estimation...")
        df_ior = pd.DataFrame()
    else:
        df_ior = pd.read_csv(ior_data_path)
        print(f"   Loaded {len(df_ior)} IOR benchmark records")
    
    # Step 5: Estimate transfer rates (if IOR data is available)
    if not df_ior.empty:
        print("\n5. Estimating transfer rates...")
        # Get allowed_parallelism from config, with fallback to default
        allowed_parallelism = config.get("ALLOWED_PARALLELISM", None)
        wf_df = estimate_transfer_rates_for_workflow(wf_df, df_ior, STORAGE_LIST, allowed_parallelism)
        print("   Transfer rate estimation completed")
    else:
        print("\n5. Skipping transfer rate estimation (no IOR data)")
    
    # Step 6: Calculate SPM values
    print("\n6. Calculating SPM values...")
    spm_results = calculate_spm_for_workflow(wf_df)
    print(f"   Calculated SPM for {len(spm_results)} producer-consumer pairs")
    
    # Step 7: Filter storage options
    print("\n7. Filtering storage options...")
    filtered_spm_results = filter_storage_options(spm_results, workflow_name)
    
    # Step 8: Select best storage and parallelism
    print("\n8. Selecting best storage and parallelism...")
    best_results = select_best_storage_and_parallelism(spm_results, baseline=0)
    
    # Step 9: Display top results
    print("\n9. Displaying top results...")
    display_top_sorted_averaged_rank(filtered_spm_results, top_n=10)
    
    # Step 10: Generate visualizations
    print("\n10. Generating visualizations...")
    if save_results:
        plot_all_visualizations(wf_df, best_results, io_breakdown['task_io_time_adjust'])
    
    # Step 11: Export producer-consumer results to CSV
    print("\n11. Exporting producer-consumer results...")
    if save_results:
        csv_path = save_producer_consumer_results(best_results, wf_df, workflow_name)
        
        # Print storage analysis
        from modules.workflow_results_exporter import extract_producer_consumer_results
        results_df = extract_producer_consumer_results(best_results, wf_df)
        print_storage_analysis(results_df)
    
    # Step 12: Save additional results
    if save_results:
        print("\n12. Saving additional results...")
        os.makedirs("./analysis_data", exist_ok=True)
        
        # Save workflow DataFrame
        wf_df.to_csv(f'./analysis_data/{workflow_name}_workflow_data.csv', index=False)
        print(f"   Saved workflow data to: ./analysis_data/{workflow_name}_workflow_data.csv")
        
        # Save SPM results
        import json
        with open(f'./analysis_data/{workflow_name}_spm_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(best_results, f, default=convert_numpy, indent=2)
        print(f"   Saved SPM results to: ./analysis_data/{workflow_name}_spm_results.json")
    
    print("\n" + "=" * 60)
    print("Workflow analysis completed successfully!")
    
    # Return results
    return {
        'workflow_df': wf_df,
        'task_order_dict': task_order_dict,
        'all_wf_dict': all_wf_dict,
        'io_breakdown': io_breakdown,
        'spm_results': spm_results,
        'filtered_spm_results': filtered_spm_results,
        'best_results': best_results
    }


def analyze_multiple_workflows(workflow_names: list = None, 
                              ior_data_path: str = "../perf_profiles/updated_master_ior_df.csv") -> Dict[str, Any]:
    """
    Analyze multiple workflows and compare results.
    
    Parameters:
    - workflow_names: List of workflow names to analyze
    - ior_data_path: Path to the IOR benchmark data
    
    Returns:
    - Dict: Results for all workflows
    """
    if workflow_names is None:
        workflow_names = list(TEST_CONFIGS.keys())
    
    all_results = {}
    
    print(f"Analyzing {len(workflow_names)} workflows...")
    print("=" * 60)
    
    for workflow_name in workflow_names:
        if workflow_name in TEST_CONFIGS:
            print(f"\nAnalyzing workflow: {workflow_name}")
            try:
                results = run_workflow_analysis(workflow_name, ior_data_path, save_results=True)
                all_results[workflow_name] = results
            except Exception as e:
                print(f"   Error analyzing {workflow_name}: {e}")
                all_results[workflow_name] = {'error': str(e)}
        else:
            print(f"   Warning: Unknown workflow '{workflow_name}', skipping...")
    
    return all_results


def main():
    """Main function to run the workflow analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Workflow Analysis Tool')
    parser.add_argument('--workflow', '-w', type=str, default=DEFAULT_WF,
                       help=f'Workflow name to analyze (default: {DEFAULT_WF})')
    parser.add_argument('--ior-data', '-i', type=str, 
                       default="../perf_profiles/updated_master_ior_df.csv",
                       help='Path to IOR benchmark data CSV file')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Analyze all available workflows')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    if args.all:
        # Analyze all workflows
        results = analyze_multiple_workflows(ior_data_path=args.ior_data)
        print(f"\nCompleted analysis of {len(results)} workflows")
    else:
        # Analyze single workflow
        results = run_workflow_analysis(
            workflow_name=args.workflow,
            ior_data_path=args.ior_data,
            save_results=not args.no_save
        )
        print(f"\nCompleted analysis of workflow: {args.workflow}")
    
    return results


if __name__ == "__main__":
    main() 