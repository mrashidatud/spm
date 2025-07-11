"""
Template Workflow Generator

This module provides functions to generate artificial workflow data for testing
the workflow analysis system. It creates producer-consumer task relationships
with realistic I/O patterns and timing data.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
try:
    from .workflow_config import WF_PARAMS
except ImportError:
    from workflow_config import WF_PARAMS


def generate_template_workflow_data(
    workflow_name: str = "template_workflow",
    num_nodes: int = 4,
    base_file_size_mb: float = 100.0,
    time_variance: float = 0.2,
    debug: bool = False
) -> pd.DataFrame:
    """
    Generate artificial workflow data for testing the analysis system.
    
    Parameters:
    - workflow_name: Name of the workflow
    - num_nodes: Number of nodes to simulate
    - base_file_size_mb: Base file size in MB
    - time_variance: Variance in timing (0.0 to 1.0)
    - debug: Enable debug output
    
    Returns:
    - pd.DataFrame: Artificial workflow data
    """
    
    # Define task configurations
    tasks = {
        "task1": {
            "stage_order": 0,
            "parallelism": 4,
            "num_tasks": 4,
            "operation": 1,  # Read operation
            "files": ["input_data_1.txt", "input_data_2.txt", "input_data_3.txt", "input_data_4.txt"],
            "output_files": ["task1_output_1.dat", "task1_output_2.dat", "task1_output_3.dat", "task1_output_4.dat"],
            "file_size_multiplier": 1.0,
            "time_multiplier": 1.0
        },
        "task2": {
            "stage_order": 1,
            "parallelism": 2,
            "num_tasks": 2,
            "operation": 1,  # Read operation
            "files": ["task1_output_1.dat", "task1_output_2.dat"],
            "output_files": ["final_result_1.out", "final_result_2.out"],
            "file_size_multiplier": 0.8,
            "time_multiplier": 1.5
        }
    }
    
    # Generate data rows
    rows = []
    task_pid_counter = 1000
    
    for task_name, task_config in tasks.items():
        stage_order = task_config["stage_order"]
        parallelism = task_config["parallelism"]
        operation = task_config["operation"]
        files = task_config["files"]
        output_files = task_config["output_files"]
        file_size_multiplier = task_config["file_size_multiplier"]
        time_multiplier = task_config["time_multiplier"]
        
        # Generate read operations
        for i, file_name in enumerate(files):
            # Add some variance to file sizes
            file_size = base_file_size_mb * file_size_multiplier * (1 + np.random.uniform(-time_variance, time_variance))
            
            # Generate timing data
            base_time = 10.0 * time_multiplier  # Base time in seconds
            total_time = base_time * (1 + np.random.uniform(-time_variance, time_variance))
            
            # Calculate transfer rate (MB/s)
            transfer_rate = file_size / total_time if total_time > 0 else 0
            
            row = {
                'operation': operation,
                'randomOffset': 0,
                'transferSize': 4096,  # 4KB blocks
                'aggregateFilesizeMB': file_size,
                'numTasks': parallelism,
                'parallelism': parallelism,
                'totalTime': total_time,
                'numNodesList': [num_nodes],
                'numNodes': num_nodes,
                'tasksPerNode': int(np.ceil(parallelism / num_nodes)),
                'trMiB': transfer_rate,
                'storageType': 'beegfs',  # Default storage
                'opCount': parallelism,
                'taskName': task_name,
                'taskPID': task_pid_counter + i,
                'fileName': file_name,
                'stageOrder': stage_order,
                'prevTask': 'initial_data' if stage_order == 0 else 'task1',
                'aggregateFilesizeMBtask': file_size / parallelism
            }
            rows.append(row)
        
        # Generate write operations for task outputs
        for i, output_file in enumerate(output_files):
            file_size = base_file_size_mb * file_size_multiplier * (1 + np.random.uniform(-time_variance, time_variance))
            base_time = 8.0 * time_multiplier  # Write operations typically faster
            total_time = base_time * (1 + np.random.uniform(-time_variance, time_variance))
            transfer_rate = file_size / total_time if total_time > 0 else 0
            
            row = {
                'operation': 0,  # Write operation
                'randomOffset': 0,
                'transferSize': 4096,
                'aggregateFilesizeMB': file_size,
                'numTasks': parallelism,
                'parallelism': parallelism,
                'totalTime': total_time,
                'numNodesList': [num_nodes],
                'numNodes': num_nodes,
                'tasksPerNode': int(np.ceil(parallelism / num_nodes)),
                'trMiB': transfer_rate,
                'storageType': 'beegfs',
                'opCount': parallelism,
                'taskName': task_name,
                'taskPID': task_pid_counter + len(files) + i,
                'fileName': output_file,
                'stageOrder': stage_order,
                'prevTask': 'initial_data' if stage_order == 0 else 'task1',
                'aggregateFilesizeMBtask': file_size / parallelism
            }
            rows.append(row)
        
        task_pid_counter += 100
    
    # Create DataFrame with all required columns
    df = pd.DataFrame(rows)
    for col in WF_PARAMS:
        if col not in df.columns:
            df[col] = ''
    df = df[WF_PARAMS]  # Ensure column order
    
    if debug:
        print(f"Generated {len(df)} workflow data rows")
        print(f"Tasks: {df['taskName'].unique()}")
        print(f"Operations: {df['operation'].unique()}")
        print(f"File sizes range: {df['aggregateFilesizeMB'].min():.2f} - {df['aggregateFilesizeMB'].max():.2f} MB")
    
    return df


def create_template_workflow_structure(
    workflow_name: str = "template_workflow",
    output_dir: str = "./template_workflow",
    debug: bool = False,
    csv_filename: str = "workflow_data.csv"
) -> Tuple[str, str]:
    """
    Create a complete template workflow structure with script order and data.
    
    Parameters:
    - workflow_name: Name of the workflow
    - output_dir: Output directory for the template
    - debug: Enable debug output
    - csv_filename: Name of the CSV file to save (default: "workflow_data.csv")
    
    Returns:
    - Tuple[str, str]: Paths to script order file and data directory
    """
    
    # Create directory structure
    script_order_dir = f"{output_dir}"
    data_dir = f"{output_dir}/template_t1/t1"  # Place in t1 subdirectory for loader compatibility
    
    os.makedirs(script_order_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate workflow data
    wf_df = generate_template_workflow_data(workflow_name, debug=debug)
    
    # Save workflow data to CSV
    csv_path = f"{data_dir}/{csv_filename}"
    wf_df.to_csv(csv_path, index=False)
    
    if debug:
        print(f"Created template workflow structure:")
        print(f"  Script order: {script_order_dir}/template_script_order.json")
        print(f"  Data file: {csv_path}")
        print(f"  Data rows: {len(wf_df)}")
    
    return f"{script_order_dir}/template_script_order.json", data_dir


def add_workflow_to_config(
    workflow_name: str = "template_workflow",
    config_file: str = "modules/workflow_config.py"
) -> None:
    """
    Add the template workflow to the workflow configuration.
    
    Parameters:
    - workflow_name: Name of the workflow to add
    - config_file: Path to the configuration file
    """
    
    # Read current config
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    # Define template workflow config
    template_config = f'''
    "{workflow_name}": {{
        "SCRIPT_ORDER": "template_script_order",
        "NUM_NODES_LIST": [4],
        "ALLOWED_PARALLELISM": [1, 2, 4, 8],
        "exp_data_path": "./template_workflow",
        "test_folders": ["template_t1"]
    }},'''
    
    # Find the TEST_CONFIGS dictionary and add the template
    if 'TEST_CONFIGS = {' in config_content:
        # Insert before the closing brace of TEST_CONFIGS
        config_content = config_content.replace(
            'TEST_CONFIGS = {',
            f'TEST_CONFIGS = {{{template_config}'
        )
        
        # Write updated config
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"Added {workflow_name} to workflow configuration")


def generate_complete_template(
    workflow_name: str = "template_workflow",
    debug: bool = False,
    csv_filename: str = "workflow_data.csv"
) -> Dict[str, str]:
    """
    Generate a complete template workflow with all necessary files.
    
    Parameters:
    - workflow_name: Name of the template workflow
    - debug: Enable debug output
    - csv_filename: Name of the CSV file to save (default: "workflow_data.csv")
    
    Returns:
    - Dict[str, str]: Paths to created files
    """
    
    # Create workflow structure
    script_order_path, data_dir = create_template_workflow_structure(workflow_name, debug=debug, csv_filename=csv_filename)
    
    # Add to configuration
    add_workflow_to_config(workflow_name)
    
    # Generate additional test files
    test_files = generate_test_files(data_dir, debug=debug)
    
    result = {
        'script_order': script_order_path,
        'data_dir': data_dir,
        'test_files': test_files
    }
    
    if debug:
        print(f"\nTemplate workflow '{workflow_name}' created successfully!")
        print(f"Files created:")
        for key, path in result.items():
            print(f"  {key}: {path}")
    
    return result


def generate_test_files(data_dir: str, debug: bool = False) -> List[str]:
    """
    Generate additional test files for the template workflow.
    
    Parameters:
    - data_dir: Directory to create test files in
    - debug: Enable debug output
    
    Returns:
    - List[str]: Paths to created test files
    """
    
    test_files = []
    
    # Create some dummy files that match the workflow patterns
    files_to_create = [
        "input_data_1.txt",
        "input_data_2.txt", 
        "input_data_3.txt",
        "input_data_4.txt",
        "task1_output_1.dat",
        "task1_output_2.dat",
        "task1_output_3.dat",
        "task1_output_4.dat",
        "final_result_1.out",
        "final_result_2.out"
    ]
    
    for filename in files_to_create:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w') as f:
            f.write(f"# Template test file: {filename}\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n")
            f.write(f"# This is a dummy file for testing workflow analysis\n")
        test_files.append(filepath)
    
    if debug:
        print(f"Created {len(test_files)} test files in {data_dir}")
    
    return test_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Workflow Template Generator')
    parser.add_argument('--workflow-name', '-w', type=str, default="template_workflow",
                       help='Name of the template workflow (default: template_workflow)')
    parser.add_argument('--csv-filename', '-c', type=str, default="workflow_data.csv",
                       help='Name of the CSV file to save (default: workflow_data.csv)')
    parser.add_argument('--output-dir', '-o', type=str, default="./template_workflow",
                       help='Output directory for the template (default: ./template_workflow)')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    print("Generating template workflow...")
    result = generate_complete_template(
        workflow_name=args.workflow_name,
        debug=args.debug,
        csv_filename=args.csv_filename
    )
    print(f"\nTemplate created successfully!")
    print(f"Script order: {result['script_order']}")
    print(f"Data directory: {result['data_dir']}")
    print(f"CSV filename: {args.csv_filename}") 