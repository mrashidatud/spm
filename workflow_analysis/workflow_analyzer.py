"""
Workflow Analyzer Script.
Performs complete workflow analysis on a CSV dataframe.
This script handles Steps 2+ of the workflow analysis pipeline.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from typing import Dict, Any

# Import our modules
from modules.workflow_config import DEFAULT_WF, TEST_CONFIGS, STORAGE_LIST
from modules.workflow_data_utils import calculate_io_time_breakdown
from modules.workflow_interpolation import (
    estimate_transfer_rates_for_workflow, calculate_aggregate_filesize_per_node
)
from modules.workflow_spm_calculator import (
    calculate_spm_for_edges, calculate_spm_from_wfg, filter_storage_options,
    select_best_storage_and_parallelism, display_top_sorted_averaged_rank,
    calculate_averages_and_rank
)
from modules.workflow_visualization import plot_all_visualizations
from modules.workflow_results_exporter import save_producer_consumer_results, print_storage_analysis, extract_producer_consumer_results
from modules.workflow_data_staging import insert_data_staging_rows


def analyze_workflow_from_csv(csv_file_path: str,
                             workflow_name: str = None,
                             ior_data_path: str = "../perf_profiles/updated_master_ior_df.csv",
                             save_results: bool = True) -> Dict[str, Any]:
    """
    Run the complete workflow analysis pipeline from a CSV file.
    
    Parameters:
    - csv_file_path: Path to the CSV file containing workflow data
    - workflow_name: Name of the workflow (if not specified, will be inferred from filename)
    - ior_data_path: Path to the IOR benchmark data
    - save_results: Whether to save results to files
    
    Returns:
    - Dict: Analysis results
    """
    # Determine workflow name from filename if not provided
    if workflow_name is None:
        filename = os.path.basename(csv_file_path)
        # Extract workflow name from pattern: {workflow_name}_workflow_data*.csv
        if "_workflow_data" in filename:
            workflow_name = filename.split("_workflow_data")[0]
            print(f"Extracted workflow name '{workflow_name}' from filename: {filename}")
        else:
            # Fallback: remove .csv extension
            workflow_name = filename.replace(".csv", "")
            print(f"Extracted workflow name '{workflow_name}' from filename (fallback): {filename}")
    else:
        print(f"Using provided workflow name: {workflow_name}")
    
    print(f"Starting workflow analysis for: {workflow_name}")
    print(f"Input CSV file: {csv_file_path}")
    print("=" * 60)
    
    # Load workflow data from CSV
    print("\n1. Loading workflow data from CSV...")
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    wf_df_original = pd.read_csv(csv_file_path)
    print(f"   Loaded {len(wf_df_original)} workflow records from CSV")
    print(f"   Columns: {list(wf_df_original.columns)}")
    
    # Debug: Check storageType values right after CSV loading
    print(f"   DEBUG: storageType values after CSV loading: {sorted(wf_df_original['storageType'].unique())}")
    print(f"   DEBUG: storageType data type after CSV loading: {wf_df_original['storageType'].dtype}")
    print(f"   DEBUG: Sample storageType values after CSV loading: {wf_df_original['storageType'].head(10).tolist()}")
    
    # Create a copy to ensure we don't modify the original data
    wf_df = wf_df_original.copy()
    print(f"   Created working copy of the data (original CSV file will remain unchanged)")
    
    # Get configuration for the workflow
    if workflow_name not in TEST_CONFIGS:
        print(f"   Warning: Workflow '{workflow_name}' not found in TEST_CONFIGS")
        print(f"   Available workflows: {list(TEST_CONFIGS.keys())}")
        # Try to infer basic config from data
        num_nodes_list = sorted(wf_df['numNodes'].unique()) if 'numNodes' in wf_df.columns else [1]
        task_name_to_parallelism = {}
        if 'taskName' in wf_df.columns and 'parallelism' in wf_df.columns:
            task_name_to_parallelism = wf_df.groupby('taskName')['parallelism'].first().to_dict()
    else:
        config = TEST_CONFIGS[workflow_name]
        num_nodes_list = config["NUM_NODES_LIST"]
        
        # Create task name to parallelism mapping
        if 'taskName' in wf_df.columns and 'parallelism' in wf_df.columns:
            task_name_to_parallelism = wf_df.groupby('taskName')['parallelism'].first().to_dict()
        else:
            task_name_to_parallelism = {}
    
    print(f"   Unique tasks: {list(wf_df['taskName'].unique()) if 'taskName' in wf_df.columns else 'N/A'}")
    print(f"   Stages: {sorted(wf_df['stageOrder'].unique()) if 'stageOrder' in wf_df.columns else 'N/A'}")
    
    # Save the original workflow data (before any modifications)
    if save_results:
        os.makedirs("./analysis_data", exist_ok=True)
        original_wf_df_path = f'./analysis_data/{workflow_name}_original_workflow_data.csv'
        wf_df_original.to_csv(original_wf_df_path, index=False)
        print(f"   Saved original workflow data to: {original_wf_df_path}")
    
    # Step 2: Calculate I/O time breakdown
    print("\n2. Calculating I/O time breakdown...")
    if task_name_to_parallelism:
        io_breakdown = calculate_io_time_breakdown(wf_df, task_name_to_parallelism, num_nodes_list)
    else:
        print("   Skipping I/O time breakdown (no task parallelism info)")
        io_breakdown = {}
    
    # Step 3: Insert data staging rows
    print("\n3. Inserting data staging rows...")
    wf_df = insert_data_staging_rows(wf_df, debug=False)
    print(f"   Added staging rows, total records: {len(wf_df)}")
    
    # Debug: Check storageType values after staging
    print(f"   DEBUG: storageType values after staging: {sorted(wf_df['storageType'].unique())}")
    print(f"   DEBUG: storageType data type after staging: {wf_df['storageType'].dtype}")
    print(f"   DEBUG: Sample storageType values after staging: {wf_df['storageType'].head(10).tolist()}")
    
    # Step 3.1: Calculate aggregate file size per node (AFTER staging rows are inserted)
    print("\n3.1. Calculating aggregate file size per node...")
    wf_df = calculate_aggregate_filesize_per_node(wf_df, debug=False)
    print(f"Updated columns: {[col for col in wf_df.columns if 'aggregateFilesizeMB' in col]}")
    
    # Save the modified workflow data to a CSV file (with different name to avoid overwriting input)
    if save_results:
        wf_df.to_csv(f'./analysis_data/{workflow_name}_processed_workflow_data.csv', index=False)
        print(f"   Saved processed workflow data to: ./analysis_data/{workflow_name}_processed_workflow_data.csv")
    
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
        if workflow_name in TEST_CONFIGS:
            cp_scp_parallelism = set(wf_df.loc[wf_df['operation'].isin(['cp', 'scp', 'none']), 'parallelism'].unique())
            ALLOWED_PARALLELISM = TEST_CONFIGS[workflow_name]["ALLOWED_PARALLELISM"]
            allowed_parallelism = sorted(set(ALLOWED_PARALLELISM).union(cp_scp_parallelism))
        else:
            allowed_parallelism = sorted(wf_df['parallelism'].unique())
        print(f"   Allowed parallelism: {allowed_parallelism}")
        wf_df = estimate_transfer_rates_for_workflow(
            wf_df, df_ior, STORAGE_LIST, allowed_parallelism, multi_nodes=True, debug=False)
        print("   Transfer rate estimation completed")
    else:
        print("\n5. Skipping transfer rate estimation (no IOR data)")
    
    # Step 6: Build workflow graph and add edges
    print("\n6. Building workflow graph and adding edges...")
    wfg_graph = calculate_spm_for_edges(wf_df, debug=False, workflow_name=workflow_name)
    print(f"   Built workflow graph and added edges")
    
    # Step 6.5: Calculate SPM values from the built graph
    print("\n6.5. Calculating SPM values from workflow graph...")
    # Note: This step has its own debug option separate from graph building
    spm_results = calculate_spm_from_wfg(wfg_graph, debug=False)  # Set to False by default, can be controlled separately
    print(f"   Calculated SPM for {len(spm_results)} producer-consumer pairs")
    
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
    if workflow_name in TEST_CONFIGS:
        filtered_spm_results = filter_storage_options(spm_results, workflow_name)
    else:
        print("   Skipping storage filtering (workflow not in config)")
        filtered_spm_results = spm_results
    
    # Step 8: Select best storage and parallelism
    print("\n8. Selecting best storage and parallelism...")
    best_results = select_best_storage_and_parallelism(spm_results, baseline=0)
    
    # Step 9: Display top results
    print("\n9. Displaying top results...")
    display_top_sorted_averaged_rank(spm_results, top_n=200)
    
    # Step 10: Generate visualizations (currently not working, skipping)
    print("\n10. Generating visualizations...")
    print("   Skipping visualizations for now...")
    # if save_results:
    #     plot_all_visualizations(wf_df, best_results, io_breakdown.get('task_io_time_adjust', {}))
    
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
        
        # Save workflow DataFrame (with different name to avoid overwriting input)
        wf_df.to_csv(f'./analysis_data/{workflow_name}_processed_workflow_data.csv', index=False)
        print(f"   Saved processed workflow data to: ./analysis_data/{workflow_name}_processed_workflow_data.csv")
        
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
        'io_breakdown': io_breakdown,
        'spm_results': spm_results,
        'filtered_spm_results': filtered_spm_results,
        'best_results': best_results
    }


def main():
    """Main function to run the workflow analyzer."""
    parser = argparse.ArgumentParser(description='Workflow Analyzer - Analyze workflow data from CSV')
    parser.add_argument('csv_file', type=str,
                       help='Path to the CSV file containing workflow data')
    parser.add_argument('--workflow', '-w', type=str, default=None,
                       help='Workflow name (if not specified, will be extracted from CSV filename pattern: {workflow_name}_workflow_data*.csv)')
    parser.add_argument('--ior-data', '-i', type=str, 
                       default="../perf_profiles/updated_master_ior_df.csv",
                       help='Path to IOR benchmark data CSV file')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Analyze workflow from CSV
    results = analyze_workflow_from_csv(
        csv_file_path=args.csv_file,
        workflow_name=args.workflow,
        ior_data_path=args.ior_data,
        save_results=not args.no_save
    )
    
    print(f"\nCompleted analysis of workflow from: {args.csv_file}")
    
    return results


if __name__ == "__main__":
    main()
