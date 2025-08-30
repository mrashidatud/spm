"""
Workflow Data Loader Script.
Loads all datalife JSON files for a given workflow and collects them into a CSV dataframe.
This script handles Step 1 of the workflow analysis pipeline.
"""

import pandas as pd
import os
import sys
import argparse
from typing import Dict, Any

# Import our modules
from modules.workflow_config import DEFAULT_WF, TEST_CONFIGS
from modules.workflow_data_utils import load_workflow_data


def load_and_save_workflow_data(workflow_name: str = DEFAULT_WF, 
                               output_dir: str = "./analysis_data",
                               csv_filename: str = "workflow_data.csv") -> str:
    """
    Load workflow data from JSON files and save to CSV format.
    
    Parameters:
    - workflow_name: Name of the workflow to load
    - output_dir: Directory to save the CSV file
    - csv_filename: Name of the CSV file to save
    
    Returns:
    - str: Path to the saved CSV file
    """
    print(f"Loading workflow data for: {workflow_name}")
    print("=" * 60)
    
    # Step 1: Load workflow data
    print("\n1. Loading workflow data from JSON files...")
    wf_df, task_order_dict, all_wf_dict = load_workflow_data(workflow_name, csv_filename=csv_filename, debug=False)

    # Get configuration for the workflow
    config = TEST_CONFIGS[workflow_name]
    num_nodes_list = config["NUM_NODES_LIST"]
    
    # Create task name to parallelism mapping
    task_name_to_parallelism = {task: info['parallelism'] for task, info in task_order_dict.items()}
    
    print(f"   Loaded {len(wf_df)} workflow records")
    print(f"   Found {len(task_order_dict)} task definitions")
    print(f"   Unique tasks: {list(wf_df['taskName'].unique())}")
    print(f"   Stages: {sorted(wf_df['stageOrder'].unique())}")
    
    # Save the workflow data to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{workflow_name}_workflow_data.csv")
    wf_df.to_csv(output_path, index=False)
    print(f"   Saved workflow data to: {output_path}")
    
    # Print summary information
    print(f"\nSummary:")
    print(f"   Workflow: {workflow_name}")
    print(f"   Total records: {len(wf_df)}")
    print(f"   Tasks: {list(wf_df['taskName'].unique())}")
    print(f"   Operations: {list(wf_df['operation'].unique())}")
    print(f"   Storage types: {list(wf_df['storageType'].unique())}")
    print(f"   Output file: {output_path}")
    
    return output_path


def print_available_workflows():
    """Print all available workflow names from the configuration."""
    print("Available workflows:")
    print("=" * 40)
    for workflow_name in sorted(TEST_CONFIGS.keys()):
        config = TEST_CONFIGS[workflow_name]
        print(f"  {workflow_name}")
        print(f"    - Nodes: {config['NUM_NODES_LIST']}")
        print(f"    - Parallelism: {config['ALLOWED_PARALLELISM']}")
        print(f"    - Data path: {config['exp_data_path']}")
        print()
    print(f"Default workflow: {DEFAULT_WF}")


def main():
    """Main function to run the workflow data loader."""
    parser = argparse.ArgumentParser(description='Workflow Data Loader - Load JSON files and save to CSV')
    parser.add_argument('--workflow', '-w', type=str, default=DEFAULT_WF,
                       help=f'Workflow name to load (default: {DEFAULT_WF})')
    parser.add_argument('--output-dir', '-o', type=str, default="./analysis_data",
                       help='Output directory for CSV file (default: ./analysis_data)')
    parser.add_argument('--csv-filename', '-c', type=str, default="workflow_data.csv",
                       help='Name of the workflow CSV file to load (default: workflow_data.csv)')
    parser.add_argument('--list-workflows', '-l', action='store_true',
                       help='List all available workflows and exit')
    
    args = parser.parse_args()
    
    if args.list_workflows:
        # Print available workflows and exit
        print_available_workflows()
        return
    
    # Load and save workflow data
    output_path = load_and_save_workflow_data(
        workflow_name=args.workflow,
        output_dir=args.output_dir,
        csv_filename=args.csv_filename
    )
    
    print(f"\n✓ Workflow data loading completed!")
    print(f"  Output file: {output_path}")
    print(f"  You can now use this CSV file with the workflow_analyzer.py script")


if __name__ == "__main__":
    main()
