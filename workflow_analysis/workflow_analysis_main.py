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
    display_top_sorted_averaged_rank, select_best_storage_and_parallelism,
    calculate_averages_and_rank
)
from modules.workflow_visualization import plot_all_visualizations
from modules.workflow_results_exporter import save_producer_consumer_results, print_storage_analysis, extract_producer_consumer_results
from modules.workflow_data_staging import insert_data_staging_rows


def run_workflow_analysis(workflow_name: str = DEFAULT_WF, 
                         ior_data_path: str = "../perf_profiles/updated_master_ior_df.csv",
                         save_results: bool = True, csv_filename: str = "workflow_data.csv") -> Dict[str, Any]:
    """
    Run the complete workflow analysis pipeline.
    
    Parameters:
    - workflow_name: Name of the workflow to analyze
    - ior_data_path: Path to the IOR benchmark data
    - save_results: Whether to save results to files
    - csv_filename: Name of the CSV file to load (default: "workflow_data.csv")
    
    Returns:
    - Dict: Analysis results
    """
    print(f"Starting workflow analysis for: {workflow_name}")
    print("=" * 60)
    
    # Step 1: Load workflow data
    print("\n1. Loading workflow data...")
    wf_df, task_order_dict, all_wf_dict = load_workflow_data(workflow_name, csv_filename=csv_filename)
    
    # Get configuration for the workflow
    config = TEST_CONFIGS[workflow_name]
    num_nodes_list = config["NUM_NODES_LIST"]
    
    # Create task name to parallelism mapping
    task_name_to_parallelism = {task: info['parallelism'] for task, info in task_order_dict.items()}
    
    print(f"   Loaded {len(wf_df)} workflow records")
    print(f"   Found {len(task_order_dict)} task definitions")
    print(f"   Unique tasks: {list(wf_df['taskName'].unique())}")
    print(f"   Stages: {sorted(wf_df['stageOrder'].unique())}")
    
    # Step 2: Calculate I/O time breakdown
    print("\n2. Calculating I/O time breakdown...")
    io_breakdown = calculate_io_time_breakdown(wf_df, task_name_to_parallelism, num_nodes_list)
    
    # Step 2.1: Calculate aggregate file size per node
    print("\n2.1. Calculating aggregate file size per node...")
    wf_df = calculate_aggregate_filesize_per_node(wf_df)
    print(f"Updated columns: {[col for col in wf_df.columns if 'aggregateFilesizeMB' in col]}")
    
    # Step 3: Insert data staging rows
    print("\n3. Inserting data staging rows...")
    wf_df = insert_data_staging_rows(wf_df)
    print(f"   Added staging rows, total records: {len(wf_df)}")
    
    # Step 4: Load IOR benchmark data
    print("\n4. Loading IOR benchmark data...")
    if not os.path.exists(ior_data_path):
        print(f"   Warning: IOR data file not found at {ior_data_path}")
        print("   Skipping transfer rate estimation...")
        df_ior = pd.DataFrame()
    else:
        df_ior = pd.read_csv(ior_data_path)
        print(f"   Loaded {len(df_ior)} IOR benchmark records")
    
    # Step 5: Estimate transfer rates
    if not df_ior.empty:
        print("\n5. Estimating transfer rates...")
        # Get allowed_parallelism from config, with fallback to default
        cp_scp_parallelism = set(wf_df.loc[wf_df['operation'].isin(['cp', 'scp']), 'parallelism'].unique())
        ALLOWED_PARALLELISM = TEST_CONFIGS[workflow_name]["ALLOWED_PARALLELISM"]
        allowed_parallelism = sorted(set(ALLOWED_PARALLELISM).union(cp_scp_parallelism))
        print(f"   Allowed parallelism: {allowed_parallelism}")
        wf_df = estimate_transfer_rates_for_workflow(
            wf_df, df_ior, STORAGE_LIST, allowed_parallelism, multi_nodes=True, debug=False)
        print("   Transfer rate estimation completed")
    else:
        print("\n5. Skipping transfer rate estimation (no IOR data)")
    
    # Step 6: Calculate SPM values
    print("\n6. Calculating SPM values...")
    spm_results = calculate_spm_for_workflow(wf_df, debug=False)
    print(f"   Calculated SPM for {len(spm_results)} producer-consumer pairs")
    
    # Add ranking step to match notebook
    spm_results = calculate_averages_and_rank(spm_results, debug=False)
    
    # Debug: Check SPM results structure
    if spm_results:
        print(f"   SPM result keys: {list(spm_results.keys())}")
        sample_pair = list(spm_results.keys())[0]
        print(f"   Sample pair '{sample_pair}' structure: {list(spm_results[sample_pair].keys())}")
        if 'rank' in spm_results[sample_pair]:
            print(f"   Rank data keys: {list(spm_results[sample_pair]['rank'].keys())}")
        else:
            print(f"   No 'rank' key found in sample pair")
    
    # Step 7: Filter storage options
    print("\n7. Filtering storage options...")
    filtered_spm_results = filter_storage_options(spm_results, workflow_name)
    
    # Step 8: Select best storage and parallelism
    print("\n8. Selecting best storage and parallelism...")
    best_results = select_best_storage_and_parallelism(spm_results, baseline=0)
    
    # Step 9: Display top results
    print("\n9. Displaying top results...")
    display_top_sorted_averaged_rank(spm_results, top_n=20)
    
    # Step 10: Generate visualizations (currently not working, skipping)
    print("\n10. Generating visualizations...")
    print("   Skipping visualizations for now...")
    # if save_results:
    #     plot_all_visualizations(wf_df, best_results, io_breakdown['task_io_time_adjust'])
    
    # Step 11: Export producer-consumer results to CSV
    print("\n11. Exporting producer-consumer results...")
    if save_results:
        results_df = extract_producer_consumer_results(spm_results, wf_df)
        print_storage_analysis(results_df)
        output_dir = "workflow_spm_results"
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = f"{workflow_name}_filtered_spm_results.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Saved to: {csv_path}")
    
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
            
            json.dump(spm_results, f, default=convert_numpy, indent=2)
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
                              ior_data_path: str = "../perf_profiles/updated_master_ior_df.csv",
                              csv_filename: str = "workflow_data.csv") -> Dict[str, Any]:
    """
    Analyze multiple workflows and compare results.
    
    Parameters:
    - workflow_names: List of workflow names to analyze
    - ior_data_path: Path to the IOR benchmark data
    - csv_filename: Name of the CSV file to load (default: "workflow_data.csv")
    
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
                results = run_workflow_analysis(workflow_name, ior_data_path, save_results=True, csv_filename=csv_filename)
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
    parser.add_argument('--csv-filename', '-c', type=str, default="workflow_data.csv",
                       help='Name of the workflow CSV file to load (default: workflow_data.csv)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Analyze all available workflows')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    if args.all:
        # Analyze all workflows
        results = analyze_multiple_workflows(ior_data_path=args.ior_data, csv_filename=args.csv_filename)
        print(f"\nCompleted analysis of {len(results)} workflows")
    else:
        # Analyze single workflow
        results = run_workflow_analysis(
            workflow_name=args.workflow,
            ior_data_path=args.ior_data,
            save_results=not args.no_save,
            csv_filename=args.csv_filename
        )
        print(f"\nCompleted analysis of workflow: {args.workflow}")
    
    return results


if __name__ == "__main__":
    main() 