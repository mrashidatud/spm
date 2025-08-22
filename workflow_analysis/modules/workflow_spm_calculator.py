"""
SPM Calculator module for workflow analysis.
Contains functions for building workflow graphs and calculating SPM values.
"""

import networkx as nx
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Any
from .workflow_config import STORAGE_LIST, MULTI_NODES, NORMALIZE
import re
import time

def convert_operation_to_string(operation):
    """
    Convert numeric operations to string operations.
    0 -> 'write', 1 -> 'read'
    """
    if isinstance(operation, (int, float)):
        if operation == 0:
            return 'write'
        elif operation == 1:
            return 'read'
        else:
            return str(operation)
    return operation

def extract_storage_type_from_key(key):
    """
    Extract storage type from estimated transfer rate key.
    Handles both single storage types (beegfs) and compound storage types (beegfs-ssd).
    
    Examples:
    - 'estimated_trMiB_beegfs_37p' -> 'beegfs'
    - 'estimated_trMiB_beegfs-ssd_37p' -> 'beegfs-ssd'
    """
    if not key.startswith('estimated_trMiB_'):
        return None
    
    parts = key.split('_')
    if len(parts) >= 3:
        # Handle both compound storage types (beegfs-ssd) and single storage types (beegfs)
        if len(parts) == 4:  # estimated_trMiB_beegfs_37p
            return parts[2]  # beegfs
        else:  # estimated_trMiB_beegfs-ssd_37p
            return '_'.join(parts[2:-1])  # beegfs-ssd
    return None

def is_valid_storage_match(prod_storage, cons_storage, prod_task_name=None, cons_task_name=None):
    """
    Check if storage types match according to the rules for stage_in and stage_out operations.
    
    Parameters:
    - prod_storage: Producer storage type
    - cons_storage: Consumer storage type  
    - prod_task_name: Producer task name (to check if it's stage_in-{taskName})
    - cons_task_name: Consumer task name (to check if it's stage_out-{taskName})
    
    Returns:
    - bool: True if storage types match according to rules
    """
    # Case 1: General cross-storage matching for cp/scp operations
    # If producer is single storage
    if '-' not in prod_storage:
        if prod_storage == "beegfs" and cons_storage in ("beegfs-ssd", "beegfs-tmpfs"):
            return True
        if prod_storage == "ssd" and cons_storage in ("ssd-ssd", "ssd-beegfs"):
            return True
        if prod_storage == "tmpfs" and cons_storage in ("tmpfs-tmpfs", "tmpfs-beegfs"):
            return True
    # If producer is combo
    if '-' in prod_storage:
        first, second = prod_storage.split('-')
        if prod_storage in ("beegfs-ssd", "ssd-ssd") and cons_storage == "ssd":
            return True
        if prod_storage in ("beegfs-tmpfs", "tmpfs-tmpfs") and cons_storage == "tmpfs":
            return True
        if prod_storage in ("ssd-beegfs", "tmpfs-beegfs") and cons_storage == "beegfs":
            return True
    # If consumer is combo (for cp/scp as consumer)
    if '-' in cons_storage:
        first, second = cons_storage.split('-')
        if prod_storage == first:
            return True
        
    # Case 2: Producer is stage_in-{taskName} and consumer is the corresponding taskName
    if prod_task_name and prod_task_name.startswith('stage_in-'):
        # Extract the actual task name from stage_in-{taskName}
        actual_task_name = prod_task_name.replace('stage_in-', '')
        
        # Check if consumer task name matches the actual task name
        if cons_task_name == actual_task_name:
            # Producer has compound storage (beegfs-ssd, beegfs-tmpfs, etc.)
            # Consumer should match the second storage in the compound
            if '-' in prod_storage:
                first, second = prod_storage.split('-')
                if cons_storage == second:
                    return True
            # Handle case where producer has single storage (e.g., 'beegfs')
            # For stage_in tasks with operation "none", beegfs should match with compound storage types that start with "beegfs-" or with "beegfs"
            elif prod_storage == 'beegfs':
                # Allow beegfs to match with compound storage types that start with "beegfs-" or with "beegfs"
                if cons_storage.startswith('beegfs-') or cons_storage == 'beegfs':
                    return True
            return False
    
    # Case 3: Producer is taskName and consumer is stage_out-{taskName}
    if cons_task_name and cons_task_name.startswith('stage_out-'):
        # Extract the actual task name from stage_out-{taskName}
        actual_task_name = cons_task_name.replace('stage_out-', '')
        
        # Check if producer task name matches the actual task name
        if prod_task_name == actual_task_name:
            # Consumer has compound storage (ssd-beegfs, ssd-ssd, tmpfs-beegfs, tmpfs-tmpfs)
            # Producer should match the first storage in the compound
            if '-' in cons_storage:
                first, second = cons_storage.split('-')
                if prod_storage == first:
                    return True
            # Handle case where consumer has single storage (e.g., 'beegfs')
            # For stage_out tasks with operation "none", beegfs should match with compound storage types that ends with "beegfs"
            elif cons_storage == 'beegfs':
                if prod_storage.endswith('beegfs'):
                    return True
            return False
    
    return False

def add_workflow_graph_nodes(wf_pfs_df, verbose=True):
    """
    Create a directed workflow graph from a pandas DataFrame.
    
    Parameters:
    - wf_pfs_df: DataFrame containing workflow task data
    - verbose: Boolean to control debug output
    
    Returns:
    - tuple: (WFG, stage_task_node_dict, stage_order_list)
        - WFG: NetworkX DiGraph object
        - stage_task_node_dict: Dictionary mapping stages to tasks to node lists
        - stage_order_list: Sorted list of stage orders
    """
    
    # Initialize the workflow graph as a directed graph
    WFG = nx.DiGraph()
    
    # Get the unique and sorted stage orders
    stage_order_list = sorted(float(x) for x in wf_pfs_df['stageOrder'].unique())
    if verbose:
        print(f"Stage order list: {stage_order_list}")
    
    # Dictionary to store task nodes by stage and task name
    stage_task_node_dict = {}
    
    # Add nodes to the graph
    for i, row in wf_pfs_df.iterrows():
        nodeName = f"{row['taskName']}:{row['taskPID']}:{row['fileName']}"  # Unique node identifier
        new_nodeData = row.to_dict()  # Node attributes as dictionary
        stageOrder = row['stageOrder']
        taskName = row['taskName']
        
        # Check if the node already exists
        if WFG.has_node(nodeName):
            # Update node data by replacing NaN values with new valid values
            existing_nodeData = WFG.nodes[nodeName]
            for key, value in new_nodeData.items():
                if key in existing_nodeData:
                    # Replace NaN with valid value if applicable
                    if (
                        (existing_nodeData[key] is None or 
                         (isinstance(existing_nodeData[key], float) and math.isnan(existing_nodeData[key]))) 
                        and value is not None and 
                        not (isinstance(value, float) and math.isnan(value))
                    ):
                        existing_nodeData[key] = value
                else:
                    # Add the new key-value pair
                    existing_nodeData[key] = value
            # Update the node data in the graph
            WFG.nodes[nodeName].update(existing_nodeData)
        else:
            # Add the node with attributes to the graph
            WFG.add_node(nodeName, **new_nodeData)
        
        # Populate the stage-task-node dictionary
        stage_task_node_dict.setdefault(stageOrder, {}).setdefault(taskName, []).append(nodeName)
    
    # Show graph statistics if verbose
    if verbose:
        print(f"Number of nodes: {WFG.number_of_nodes()}")
        print(f"First five nodes:")
        for info in list(WFG.nodes(data=True))[:5]:
            print(info)
    
    return WFG, stage_task_node_dict, stage_order_list


def print_graph_summary(WFG, stage_task_node_dict):
    """
    Print a summary of the workflow graph structure.
    
    Parameters:
    - WFG: NetworkX DiGraph object
    - stage_task_node_dict: Dictionary mapping stages to tasks to node lists
    """
    print(f"\n=== Workflow Graph Summary ===")
    print(f"Total nodes: {WFG.number_of_nodes()}")
    print(f"Total edges: {WFG.number_of_edges()}")
    print(f"Number of stages: {len(stage_task_node_dict)}")
    
    for stage_order in sorted(stage_task_node_dict.keys()):
        stage_data = stage_task_node_dict[stage_order]
        total_nodes_in_stage = sum(len(nodes) for nodes in stage_data.values())
        print(f"  Stage {stage_order}: {len(stage_data)} unique tasks, {total_nodes_in_stage} total nodes")
        for task_name, nodes in stage_data.items():
            print(f"    - {task_name}: {len(nodes)} nodes")

import pandas as pd
import math

def handle_stage_in_none_producers(prod_task_name, producer_group, consumer_df, WFG, edge_count, processed_pairs, processed_pairs_debug, debug=False):
    """
    Handle stage_in tasks with operation "none" (virtual producers).
    
    These are special virtual producers that represent data staging operations.
    They connect to consumers with the actual task name and 'read' operations.
    
    Parameters:
    - prod_task_name: Name of the producer task (e.g., 'stage_in-task1')
    - producer_group: DataFrame group containing producer nodes
    - consumer_df: DataFrame containing all consumer nodes
    - WFG: NetworkX DiGraph object
    - edge_count: Current edge count
    - processed_pairs: Current processed pairs count
    - processed_pairs_debug: List for debug tracking
    - debug: Debug flag
    
    Returns:
    - edge_count: Updated edge count
    - processed_pairs: Updated processed pairs count
    """
    # Extract the actual task name from stage_in-{taskName}
    actual_task_name = prod_task_name.replace('stage_in-', '')
    
    # Find consumers with the actual task name
    if actual_task_name not in consumer_df['taskName'].values:
        return edge_count, processed_pairs
        
    consumer_subset = consumer_df[consumer_df['taskName'] == actual_task_name]
    
    for _, producer_row in producer_group.iterrows():
        prod_node_name = producer_row['node_name']
        prod_fileName = producer_row['fileName']
        prod_node_data = producer_row['node_data']
        prod_op = convert_operation_to_string(producer_row.get('operation'))

        # For each consumer, check matching logic
        for _, consumer_row in consumer_subset.iterrows():
            cons_node_name = consumer_row['node_name']
            cons_fileName = consumer_row['fileName']
            cons_node_data = consumer_row['node_data']
            cons_op = consumer_row['operation']

            # Virtual producers can only connect to consumers with 'read' operations
            if cons_op != 'read':
                continue

            # For virtual producers, no file matching is needed - all read operations are valid
            # The virtual producer represents all data that the consumer task needs to read

            processed_pairs += 1
            pair_name = f"{prod_task_name}:{consumer_row['taskName']}"
            if pair_name not in processed_pairs_debug:
                processed_pairs_debug.append(pair_name)
            
            # Process edge attributes (same logic as below)
            edge_attributes = {}
            # First, find all available storage types from the actual keys
            all_prod_storages = set()
            all_cons_storages = set()
            
            # Pre-filter and cache keys for better performance
            prod_keys_by_storage = {}
            cons_keys_by_storage = {}
            
            for key in prod_node_data.keys():
                storage_type = extract_storage_type_from_key(key)
                if storage_type:
                    all_prod_storages.add(storage_type)
                    if storage_type not in prod_keys_by_storage:
                        prod_keys_by_storage[storage_type] = []
                    prod_keys_by_storage[storage_type].append(key)
            
            for key in cons_node_data.keys():
                storage_type = extract_storage_type_from_key(key)
                if storage_type:
                    all_cons_storages.add(storage_type)
                    if storage_type not in cons_keys_by_storage:
                        cons_keys_by_storage[storage_type] = []
                    cons_keys_by_storage[storage_type].append(key)
            
            if debug and prod_op == 'none':
                print(f"Stage_in none producer storages: {all_prod_storages}")
                print(f"Consumer storages: {all_cons_storages}")
            
            # Early exit if no storage types found
            if not all_prod_storages or not all_cons_storages:
                if debug:
                    print(f"No storage types found for {prod_task_name} -> {consumer_row['taskName']}")
                continue
            
            # Try all combinations of producer and consumer storage types
            for prod_storage in all_prod_storages:
                for cons_storage in all_cons_storages:
                    # For stage_in tasks with operation "none" (virtual producers), allow cross-storage matching
                    # These have storage type "beegfs" and are always valid
                    allow_cross = False
                    if (prod_op == 'none' and prod_storage == 'beegfs') or (cons_op == 'none' and cons_storage == 'beegfs'):
                        # Special case: stage_in-{taskName} with operation "none" and storage "beegfs"
                        # This represents virtual producers and are always valid
                        allow_cross = True
                    elif is_valid_storage_match(prod_storage, cons_storage, prod_task_name, consumer_row['taskName']):
                        allow_cross = True
                    
                    if not allow_cross:
                        continue
                    
                    # Use pre-filtered keys
                    prod_keys = prod_keys_by_storage.get(prod_storage, [])
                    cons_keys = cons_keys_by_storage.get(cons_storage, [])
                    
                    # Pre-calculate common values
                    prod_aggregateFilesizeMB = producer_row['aggregateFilesizeMB']
                    cons_aggregateFilesizeMB = consumer_row['aggregateFilesizeMB']
                    
                    # Validate aggregateFilesizeMB values early
                    if prod_aggregateFilesizeMB < 0:
                        if debug:
                            print(f"[WARNING] Negative prod_aggregateFilesizeMB: {prod_aggregateFilesizeMB} for {prod_task_name}")
                        continue
                    if prod_aggregateFilesizeMB == 0 and prod_op != 'none':
                        if debug:
                            print(f"[WARNING] Zero prod_aggregateFilesizeMB for non-virtual producer: {prod_task_name}")
                        continue
                    if cons_aggregateFilesizeMB <= 0:
                        if debug:
                            print(f"[WARNING] Negative/zero cons_aggregateFilesizeMB: {cons_aggregateFilesizeMB} for {consumer_row['taskName']}")
                        continue
                    
                    for prod_key in prod_keys:
                        try:
                            n_prod = prod_key.split('_')[-1]  # e.g., '1p', '24p'
                        except Exception:
                            continue
                        prod_estimated_trMiB = prod_node_data.get(prod_key)
                        prod_slope_key = f"estimated_ts_slope_{prod_storage}_{n_prod}"
                        prod_ts_slope = prod_node_data.get(prod_slope_key)
                        if prod_estimated_trMiB is None or (isinstance(prod_estimated_trMiB, float) and math.isnan(prod_estimated_trMiB)):
                            continue
                        if prod_ts_slope is None or (isinstance(prod_ts_slope, float) and math.isnan(prod_ts_slope)):
                            continue
                        
                        # Calculate estimated time based on operation type
                        if prod_op == 'none':
                            estT_prod = 0.0  # Virtual producers have zero time
                        elif cons_op == 'none':
                            estT_cons = 0.0
                        else:
                            estT_prod = prod_aggregateFilesizeMB / prod_estimated_trMiB if prod_estimated_trMiB and prod_estimated_trMiB > 0 else 0.0
                        
                        # Debug output for stage_in tasks
                        if debug and prod_task_name.startswith('stage_in-'):
                            print(f"[DEBUG] Stage_in SPM calculation for {prod_task_name} -> {consumer_row['taskName']}:")
                            print(f"  Producer: {prod_storage}_{n_prod}, Consumer: {cons_storage}_{n_cons}")
                            print(f"  prod_aggregateFilesizeMB: {prod_aggregateFilesizeMB}")
                            print(f"  prod_estimated_trMiB: {prod_estimated_trMiB}")
                            print(f"  estT_prod: {estT_prod}")
                        
                        for cons_key in cons_keys:
                            try:
                                n_cons = cons_key.split('_')[-1]  # e.g., '1p', '24p'
                            except Exception:
                                continue
                            cons_estimated_trMiB = cons_node_data.get(cons_key)
                            cons_slope_key = f"estimated_ts_slope_{cons_storage}_{n_cons}"
                            cons_ts_slope = cons_node_data.get(cons_slope_key)
                            if cons_estimated_trMiB is None or (isinstance(cons_estimated_trMiB, float) and math.isnan(cons_estimated_trMiB)):
                                continue
                            if cons_ts_slope is None or (isinstance(cons_ts_slope, float) and math.isnan(cons_ts_slope)):
                                continue
                            
                            # Handle consumer estimated time
                            if cons_estimated_trMiB == 0 or cons_estimated_trMiB is None:
                                estT_cons = 0.0  # Set to zero when transfer rate is zero or None
                            else:
                                estT_cons = cons_aggregateFilesizeMB / cons_estimated_trMiB

                            # Debug output for stage_in tasks (consumer side)
                            if debug and prod_task_name.startswith('stage_in-'):
                                print(f"  cons_aggregateFilesizeMB: {cons_aggregateFilesizeMB}")
                                print(f"  cons_estimated_trMiB: {cons_estimated_trMiB}")
                                print(f"  estT_cons: {estT_cons}")

                            # Validate that estT values are positive
                            if estT_cons <= 0:
                                if debug:
                                    print(f"[WARNING] Negative/zero estT_cons: {estT_cons}, cons_aggregateFilesizeMB: {cons_aggregateFilesizeMB}, cons_estimated_trMiB: {cons_estimated_trMiB}")
                                continue

                            # Calculate SPM based on producer type
                            if prod_op == 'none':
                                estT_prod = 0.0  # Stage_in tasks with operation "none" always have SPM = 0
                            elif cons_op == 'none':
                                estT_cons = 0.0
                            SPM = estT_prod + estT_cons #if estT_cons > 0 else float('inf')

                            # Debug output for final SPM calculation
                            if debug and prod_task_name.startswith('stage_in-'):
                                print(f"  Final SPM: {SPM}")
                                print(f"  Edge key: {prod_storage}_{n_prod.replace('p', '')}_{cons_storage}_{n_cons}")
                                print("  ---")

                            # Use both producer and consumer storage in the key
                            edge_key = f'{prod_storage}_{n_prod.replace("p", "")}_{cons_storage}_{n_cons}'
                            edge_attributes[f'estT_prod_{prod_storage}_{n_prod}'] = estT_prod
                            edge_attributes[f'estT_cons_{cons_storage}_{n_cons}'] = estT_cons
                            # Don't calculate SPM here - it will be calculated after averaging estT values
                            edge_count += 1
                # Add debug print if no edge_attributes were set
                if not edge_attributes and debug:
                    print(f"[WARNING] No SPM/estT values for edge {prod_task_name} -> {consumer_row['taskName']}")
                # Add common edge attributes
                edge_attributes.update({
                    'prod_aggregateFilesizeMB': producer_row['aggregateFilesizeMB'],
                    'cons_aggregateFilesizeMB': consumer_row['aggregateFilesizeMB'],
                    'prod_max_parallelism': producer_row['parallelism'],
                    'cons_max_parallelism': consumer_row['parallelism'],
                    'prod_stage_order': producer_row['stageOrder'],
                    'cons_stage_order': consumer_row['stageOrder'],
                    'prod_task_name': prod_task_name,
                    'cons_task_name': consumer_row['taskName'],
                    'file_name': prod_fileName,
                })
                if WFG.has_edge(prod_node_name, cons_node_name):
                    WFG.edges[prod_node_name, cons_node_name].update(edge_attributes)
                else:
                    WFG.add_edge(prod_node_name, cons_node_name, **edge_attributes)
    
    return edge_count, processed_pairs

def add_producer_consumer_edge(WFG, prod_nodes, cons_nodes, debug=False, workflow_name=None):
    """
    Add edges between producer and consumer nodes, including cp/scp nodes and handling fileName as a list for those.
    """
    import pandas as pd
    import re
    import time
    
    if debug:
        print(f"\n=== Processing producer-consumer edges ===")
        start_time = time.time()

    # Step 1: Create producer DataFrame (operation in ['read', 'write', 'cp', 'scp', 'none'])
    producer_data = []
    for node_name, node_data in WFG.nodes(data=True):
        op = node_data.get('operation')
        op_str = convert_operation_to_string(op)
        task_name = node_data.get('taskName', '')
        
        # Producers: regular tasks (read/write), stage_in tasks (cp/scp), and stage_in tasks with operation "none"
        if (op_str in ['read', 'write'] or 
            (op_str in ['cp', 'scp', 'none'] and 'stage_in' in task_name)):
            producer_data.append({
                'node_name': node_name,
                'taskName': node_data.get('taskName'),
                'fileName': node_data.get('fileName', '').strip(),
                'opCount': node_data.get('opCount'),
                'aggregateFilesizeMB': node_data.get('aggregateFilesizeMB'),
                'parallelism': node_data.get('parallelism'),
                'operation': op_str,
                'node_data': node_data,
                'stageOrder': node_data.get('stageOrder')
            })
    producer_df = pd.DataFrame(producer_data)
    if debug:
        print(f"Producers: {len(producer_df)} nodes")

    # Step 2: Create consumer DataFrame (operation in ['read', 'write', 'cp', 'scp'])
    consumer_data = []
    for node_name, node_data in WFG.nodes(data=True):
        op = node_data.get('operation')
        op_str = convert_operation_to_string(op)
        task_name = node_data.get('taskName', '')
        
        # Consumers: regular tasks (read/write) and stage_out tasks (cp/scp)
        # if (op_str in ['read', 'write', 'cp', 'scp'] and 
        #     (op_str in ['read', 'write'] or 'stage_out' in task_name)):
        if (op_str in ['read', 'write'] or 
            (op_str in ['cp', 'scp', 'none'] and 'stage_out' in task_name)):
            consumer_data.append({
                'node_name': node_name,
                'taskName': node_data.get('taskName'),
                'prevTask': node_data.get('prevTask'),
                'fileName': node_data.get('fileName', '').strip(),
                'opCount': node_data.get('opCount'),
                'aggregateFilesizeMB': node_data.get('aggregateFilesizeMB'),
                'parallelism': node_data.get('parallelism'),
                'operation': op_str,
                'node_data': node_data,
                'stageOrder': node_data.get('stageOrder')
            })
    consumer_df = pd.DataFrame(consumer_data)
    if debug:
        print(f"Consumers: {len(consumer_df)} nodes")

    # Check for potential missing pairs based on naming patterns
    missing_pairs = []
    producer_tasks = set(producer_df['taskName'].unique())
    consumer_tasks = set(consumer_df['taskName'].unique())

    # Look for stage_out pattern matches
    for prod_task in producer_tasks:
        # Skip stage_in tasks as they are not expected to have stage_out counterparts
        if 'stage_in' in prod_task:
            continue
            
        # Check for corresponding stage_out task
        expected_consumer = f'stage_out-{prod_task}'
        if expected_consumer not in consumer_tasks:
            missing_pairs.append((prod_task, expected_consumer))

    # Check for orphaned stage_out consumers (consumers without matching producers)
    for cons_task in consumer_tasks:
        if cons_task.startswith('stage_out-'):
            expected_producer = cons_task.replace('stage_out-', '')
            if expected_producer not in producer_tasks:
                if debug:
                    print(f"✗ Orphaned consumer: {cons_task} (no matching producer: {expected_producer})")

    for _, consumer_row in consumer_df.iterrows():
        prev_task = consumer_row.get('prevTask')
        consumer_task_name = consumer_row['taskName']
        stage_in_producer_name = f"stage_in-{consumer_task_name}"
        
        # Skip validation if this consumer has a stage_in producer with operation "none"
        if stage_in_producer_name in producer_tasks:
            # Check if the stage_in producer has operation "none"
            stage_in_producers = producer_df[producer_df['taskName'] == stage_in_producer_name]
            if any(convert_operation_to_string(prod.get('operation')) == 'none' for _, prod in stage_in_producers.iterrows()):
                continue
    
    # Only validate prevTask for consumers without stage_in producers with operation "none"
    if prev_task and prev_task not in producer_tasks:
        if debug:
            print(f"✗ Consumer {consumer_task_name} references missing producer: {prev_task}")

    if debug and missing_pairs:
        print(f"Summary: {len(missing_pairs)} potential missing stage_out pairs found")

    if producer_df.empty or consumer_df.empty:
        if debug:
            print("ERROR: No producers or consumers found!")
        return

    # Step 3: Find potential producer-consumer pairs
    producer_groups = producer_df.groupby('taskName')
    consumer_groups = consumer_df.groupby('prevTask')

    edge_count = 0
    processed_pairs = 0

    # Step 4: Match producer taskName with consumer prevTask
    processed_pairs_debug = []
    for prod_task_name, producer_group in producer_groups:
        # Check if this is a stage_in task with operation "none" (virtual producers)
        is_stage_in_none = False
        if prod_task_name.startswith('stage_in-'):
            # Check if any producer in this group has operation "none"
            for _, producer_row in producer_group.iterrows():
                prod_op = convert_operation_to_string(producer_row.get('operation'))
                if prod_op == 'none':
                    is_stage_in_none = True
                    break
        
        # Special handling for stage_in tasks with operation "none" (virtual producers)
        if is_stage_in_none:
            edge_count, processed_pairs = handle_stage_in_none_producers(
                prod_task_name, producer_group, consumer_df, WFG, 
                edge_count, processed_pairs, processed_pairs_debug, debug
            )
            continue  # Skip the regular prevTask matching for stage_in tasks with operation "none"

        # Special handling for stage_in tasks with operation "cp" or "scp"
        if prod_task_name.startswith('stage_in-') and any(convert_operation_to_string(prod.get('operation')) in ['cp', 'scp'] for _, prod in producer_group.iterrows()):
            # Extract the actual task name from stage_in-{taskName}
            actual_task_name = prod_task_name.replace('stage_in-', '')
            
            # Find consumers with the actual task name
            if actual_task_name in consumer_df['taskName'].values:
                consumer_subset = consumer_df[consumer_df['taskName'] == actual_task_name]
                
                for _, producer_row in producer_group.iterrows():
                    prod_node_name = producer_row['node_name']
                    prod_fileName = producer_row['fileName']
                    prod_node_data = producer_row['node_data']
                    prod_op = convert_operation_to_string(producer_row.get('operation'))

                    # Only process cp/scp operations
                    if prod_op not in ['cp', 'scp']:
                        continue

                    # For each consumer, check matching logic
                    for _, consumer_row in consumer_subset.iterrows():
                        cons_node_name = consumer_row['node_name']
                        cons_fileName = consumer_row['fileName']
                        cons_node_data = consumer_row['node_data']
                        cons_op = consumer_row['operation']

                        # File matching logic for stage_in cp/scp operations
                        prod_file_list = re.split(r",|'", prod_fileName)
                        prod_file_list = [f.strip() for f in prod_file_list if f.strip()]
                        cons_file_list = [cons_fileName]  # Consumer files are typically single files

                        # Check if any consumer file is in the producer file list
                        match = False
                        for f in cons_file_list:
                            if f in prod_file_list:
                                match = True
                                break

                        if not match:
                            continue

                        processed_pairs += 1
                        pair_name = f"{prod_task_name}:{consumer_row['taskName']}"
                        if pair_name not in processed_pairs_debug:
                            processed_pairs_debug.append(pair_name)
                        
                        # Process edge attributes (same logic as above)
                        edge_attributes = {}
                        # First, find all available storage types from the actual keys
                        all_prod_storages = set()
                        all_cons_storages = set()
                        
                        # Pre-filter and cache keys for better performance
                        prod_keys_by_storage = {}
                        cons_keys_by_storage = {}
                        
                        for key in prod_node_data.keys():
                            storage_type = extract_storage_type_from_key(key)
                            if storage_type:
                                all_prod_storages.add(storage_type)
                                if storage_type not in prod_keys_by_storage:
                                    prod_keys_by_storage[storage_type] = []
                                prod_keys_by_storage[storage_type].append(key)
                        
                        for key in cons_node_data.keys():
                            storage_type = extract_storage_type_from_key(key)
                            if storage_type:
                                all_cons_storages.add(storage_type)
                                if storage_type not in cons_keys_by_storage:
                                    cons_keys_by_storage[storage_type] = []
                                cons_keys_by_storage[storage_type].append(key)
                        
                        if debug and prod_op in ['cp', 'scp']:
                            print(f"Stage_in cp/scp producer storages: {all_prod_storages}")
                            print(f"Consumer storages: {all_cons_storages}")
                        
                        # Early exit if no storage types found
                        if not all_prod_storages or not all_cons_storages:
                            if debug:
                                print(f"No storage types found for {prod_task_name} -> {consumer_row['taskName']}")
                            continue
                        
                        # Try all combinations of producer and consumer storage types
                        for prod_storage in all_prod_storages:
                            for cons_storage in all_cons_storages:
                                # For cp/scp, allow cross-storage matching
                                allow_cross = False
                                if is_valid_storage_match(prod_storage, cons_storage, prod_task_name, consumer_row['taskName']):
                                    allow_cross = True
                                
                                if not allow_cross:
                                    continue
                                
                                # Use pre-filtered keys
                                prod_keys = prod_keys_by_storage.get(prod_storage, [])
                                cons_keys = cons_keys_by_storage.get(cons_storage, [])
                                
                                # Pre-calculate common values
                                prod_aggregateFilesizeMB = producer_row['aggregateFilesizeMB']
                                cons_aggregateFilesizeMB = consumer_row['aggregateFilesizeMB']
                                
                                # Validate aggregateFilesizeMB values early
                                if prod_aggregateFilesizeMB < 0:
                                    if debug:
                                        print(f"[WARNING] Negative prod_aggregateFilesizeMB: {prod_aggregateFilesizeMB} for {prod_task_name}")
                                    continue
                                if prod_aggregateFilesizeMB == 0:
                                    if debug:
                                        print(f"[WARNING] Zero prod_aggregateFilesizeMB for {prod_task_name}")
                                    continue
                                if cons_aggregateFilesizeMB <= 0:
                                    if debug:
                                        print(f"[WARNING] Negative/zero cons_aggregateFilesizeMB: {cons_aggregateFilesizeMB} for {consumer_row['taskName']}")
                                    continue
                                
                                for prod_key in prod_keys:
                                    try:
                                        n_prod = prod_key.split('_')[-1]  # e.g., '1p', '24p'
                                    except Exception:
                                        continue
                                    prod_estimated_trMiB = prod_node_data.get(prod_key)
                                    prod_slope_key = f"estimated_ts_slope_{prod_storage}_{n_prod}"
                                    prod_ts_slope = prod_node_data.get(prod_slope_key)
                                    if prod_estimated_trMiB is None or (isinstance(prod_estimated_trMiB, float) and math.isnan(prod_estimated_trMiB)):
                                        continue
                                    if prod_ts_slope is None or (isinstance(prod_ts_slope, float) and math.isnan(prod_ts_slope)):
                                        continue
                                    
                                    # Calculate estimated time for cp/scp operations
                                    estT_prod = prod_aggregateFilesizeMB / prod_estimated_trMiB if prod_estimated_trMiB and prod_estimated_trMiB > 0 else 0.0
                                    
                                    for cons_key in cons_keys:
                                        try:
                                            n_cons = cons_key.split('_')[-1]  # e.g., '1p', '24p'
                                        except Exception:
                                            continue
                                        cons_estimated_trMiB = cons_node_data.get(cons_key)
                                        cons_slope_key = f"estimated_ts_slope_{cons_storage}_{n_cons}"
                                        cons_ts_slope = cons_node_data.get(cons_slope_key)
                                        if cons_estimated_trMiB is None or (isinstance(cons_estimated_trMiB, float) and math.isnan(cons_estimated_trMiB)):
                                            continue
                                        if cons_ts_slope is None or (isinstance(cons_ts_slope, float) and math.isnan(cons_ts_slope)):
                                            continue
                                        
                                        # Handle consumer estimated time
                                        if cons_estimated_trMiB == 0 or cons_estimated_trMiB is None:
                                            estT_cons = 0.0  # Set to zero when transfer rate is zero or None
                                        else:
                                            estT_cons = cons_aggregateFilesizeMB / cons_estimated_trMiB

                                        # Validate that estT values are positive
                                        if estT_cons <= 0:
                                            if debug:
                                                print(f"[WARNING] Negative/zero estT_cons: {estT_cons}, cons_aggregateFilesizeMB: {cons_aggregateFilesizeMB}, cons_estimated_trMiB: {cons_estimated_trMiB}")
                                            continue

                                        SPM = estT_prod + estT_cons

                                        # Use both producer and consumer storage in the key
                                        edge_key = f'{prod_storage}_{n_prod.replace("p", "")}_{cons_storage}_{n_cons}'
                                        edge_attributes[f'estT_prod_{prod_storage}_{n_prod}'] = estT_prod
                                        edge_attributes[f'estT_cons_{cons_storage}_{n_cons}'] = estT_cons
                                        # Don't calculate SPM here - it will be calculated after averaging estT values
                                        edge_count += 1
                        
                        # Add debug print if no edge_attributes were set
                        if not edge_attributes and debug:
                            print(f"[WARNING] No SPM/estT values for edge {prod_task_name} -> {consumer_row['taskName']}")
                        # Add common edge attributes
                        edge_attributes.update({
                            'prod_aggregateFilesizeMB': producer_row['aggregateFilesizeMB'],
                            'cons_aggregateFilesizeMB': consumer_row['aggregateFilesizeMB'],
                            'prod_max_parallelism': producer_row['parallelism'],
                            'cons_max_parallelism': consumer_row['parallelism'],
                            'prod_stage_order': producer_row['stageOrder'],
                            'cons_stage_order': consumer_row['stageOrder'],
                            'prod_task_name': prod_task_name,
                            'cons_task_name': consumer_row['taskName'],
                            'file_name': prod_fileName,
                        })
                        if WFG.has_edge(prod_node_name, cons_node_name):
                            WFG.edges[prod_node_name, cons_node_name].update(edge_attributes)
                        else:
                            WFG.add_edge(prod_node_name, cons_node_name, **edge_attributes)
            continue  # Skip the regular prevTask matching for stage_in tasks with operation "cp" or "scp"

        # Find consumers that have this producer as their prevTask
        if prod_task_name in consumer_groups.groups:
            consumer_group = consumer_groups.get_group(prod_task_name)
            if debug and prod_task_name.startswith('stage_in-'):
                print(f"Processing stage_in task: {prod_task_name} with {len(consumer_group)} consumers")

            for _, producer_row in producer_group.iterrows():
                prod_node_name = producer_row['node_name']
                prod_fileName = producer_row['fileName']
                prod_node_data = producer_row['node_data']
                prod_op = producer_row['operation']

                # If producer is cp/scp, treat fileName as a list
                if prod_op in ['none','cp', 'scp']:
                    prod_file_list = re.split(r",|'", prod_fileName)
                    prod_file_list = [f.strip() for f in prod_file_list if f.strip()]
                else:
                    prod_file_list = [prod_fileName]

                # For each consumer, check matching logic
                for _, consumer_row in consumer_group.iterrows():
                    cons_node_name = consumer_row['node_name']
                    cons_fileName = consumer_row['fileName']
                    cons_node_data = consumer_row['node_data']
                    cons_op = consumer_row['operation']

                    # If consumer is cp/scp, treat fileName as a list
                    if cons_op in ['cp', 'scp']:
                        cons_file_list = re.split(r",|'", cons_fileName)
                        cons_file_list = [f.strip() for f in cons_file_list if f.strip()]
                    else:
                        cons_file_list = [cons_fileName]

                    # Matching logic:
                    match = False
                    if prod_op in ['none','cp', 'scp']:
                        for f in cons_file_list:
                            if f in prod_file_list:
                                match = True
                                break
                    elif cons_op in ['cp', 'scp']:
                        for f in prod_file_list:
                            if f in cons_file_list:
                                match = True
                                break
                    else:
                        if prod_fileName == cons_fileName:
                            match = True

                    if not match:
                        continue

                    processed_pairs += 1
                    pair_name = f"{prod_task_name}:{consumer_row['taskName']}"
                    if pair_name not in processed_pairs_debug:
                        processed_pairs_debug.append(pair_name)
                    edge_attributes = {}

                    # First, find all available storage types from the actual keys
                    all_prod_storages = set()
                    all_cons_storages = set()
                    
                    # Pre-filter and cache keys for better performance
                    prod_keys_by_storage = {}
                    cons_keys_by_storage = {}
                    
                    for key in prod_node_data.keys():
                        storage_type = extract_storage_type_from_key(key)
                        if storage_type:
                            all_prod_storages.add(storage_type)
                            if storage_type not in prod_keys_by_storage:
                                prod_keys_by_storage[storage_type] = []
                            prod_keys_by_storage[storage_type].append(key)
                    
                    for key in cons_node_data.keys():
                        storage_type = extract_storage_type_from_key(key)
                        if storage_type:
                            all_cons_storages.add(storage_type)
                            if storage_type not in cons_keys_by_storage:
                                cons_keys_by_storage[storage_type] = []
                            cons_keys_by_storage[storage_type].append(key)
                    
                    if debug and prod_op == 'none':
                        print(f"Stage_in none producer storages: {all_prod_storages}")
                        print(f"Consumer storages: {all_cons_storages}")
                    
                    # Early exit if no storage types found
                    if not all_prod_storages or not all_cons_storages:
                        if debug:
                            print(f"No storage types found for {prod_task_name} -> {consumer_row['taskName']}")
                        continue
                    
                    # Try all combinations of producer and consumer storage types
                    for prod_storage in all_prod_storages:
                        for cons_storage in all_cons_storages:
                            # For cp/scp, allow cross-storage matching
                            allow_cross = False
                            if prod_op in ['none','cp', 'scp'] or cons_op in ['cp', 'scp']:
                                if is_valid_storage_match(prod_storage, cons_storage, prod_task_name, consumer_row['taskName']):
                                    allow_cross = True
                            # For non-cp/scp, require exact match
                            if (prod_op in ['none','cp', 'scp'] or cons_op in ['cp', 'scp']):
                                if not allow_cross:
                                    continue
                            else:
                                # Strict match
                                if prod_storage != cons_storage:
                                    continue
                            
                            # Use pre-filtered keys
                            prod_keys = prod_keys_by_storage.get(prod_storage, [])
                            cons_keys = cons_keys_by_storage.get(cons_storage, [])
                            
                            # Pre-calculate common values
                            prod_aggregateFilesizeMB = producer_row['aggregateFilesizeMB']
                            cons_aggregateFilesizeMB = consumer_row['aggregateFilesizeMB']
                            
                            # Validate aggregateFilesizeMB values early
                            if prod_aggregateFilesizeMB < 0:
                                if debug:
                                    print(f"[WARNING] Negative prod_aggregateFilesizeMB: {prod_aggregateFilesizeMB} for {prod_task_name}")
                                continue
                            if prod_aggregateFilesizeMB == 0 and prod_op != 'none':
                                if debug:
                                    print(f"[WARNING] Zero prod_aggregateFilesizeMB for non-virtual producer: {prod_task_name}")
                                continue
                            if cons_aggregateFilesizeMB <= 0:
                                if debug:
                                    print(f"[WARNING] Negative/zero cons_aggregateFilesizeMB: {cons_aggregateFilesizeMB} for {consumer_row['taskName']}")
                                continue
                            
                            for prod_key in prod_keys:
                                try:
                                    n_prod = prod_key.split('_')[-1]  # e.g., '1p', '24p'
                                except Exception:
                                    continue
                                prod_estimated_trMiB = prod_node_data.get(prod_key)
                                prod_slope_key = f"estimated_ts_slope_{prod_storage}_{n_prod}"
                                prod_ts_slope = prod_node_data.get(prod_slope_key)
                                if prod_estimated_trMiB is None or (isinstance(prod_estimated_trMiB, float) and math.isnan(prod_estimated_trMiB)):
                                    continue
                                if prod_ts_slope is None or (isinstance(prod_ts_slope, float) and math.isnan(prod_ts_slope)):
                                    continue
                                
                                # Calculate estimated time based on operation type
                                if prod_op == 'none':
                                    estT_prod = 0.0  # Virtual producers have zero time
                                else:
                                    estT_prod = prod_aggregateFilesizeMB / prod_estimated_trMiB if prod_estimated_trMiB and prod_estimated_trMiB > 0 else 0.0
                                
                                # Debug output for stage_in tasks
                                if debug and prod_task_name.startswith('stage_in-'):
                                    print(f"[DEBUG] Stage_in SPM calculation for {prod_task_name} -> {consumer_row['taskName']}:")
                                    print(f"  Producer: {prod_storage}_{n_prod}, Consumer: {cons_storage}_{n_cons}")
                                    print(f"  prod_aggregateFilesizeMB: {prod_aggregateFilesizeMB}")
                                    print(f"  prod_estimated_trMiB: {prod_estimated_trMiB}")
                                    print(f"  estT_prod: {estT_prod}")
                                
                                for cons_key in cons_keys:
                                    try:
                                        n_cons = cons_key.split('_')[-1]  # e.g., '1p', '24p'
                                    except Exception:
                                        continue
                                    cons_estimated_trMiB = cons_node_data.get(cons_key)
                                    cons_slope_key = f"estimated_ts_slope_{cons_storage}_{n_cons}"
                                    cons_ts_slope = cons_node_data.get(cons_slope_key)
                                    if cons_estimated_trMiB is None or (isinstance(cons_estimated_trMiB, float) and math.isnan(cons_estimated_trMiB)):
                                        continue
                                    if cons_ts_slope is None or (isinstance(cons_ts_slope, float) and math.isnan(cons_ts_slope)):
                                        continue
                                    
                                    # Handle consumer estimated time
                                    if cons_estimated_trMiB == 0 or cons_estimated_trMiB is None:
                                        estT_cons = 0.0  # Set to zero when transfer rate is zero or None
                                    else:
                                        estT_cons = cons_aggregateFilesizeMB / cons_estimated_trMiB

                                    # Debug output for stage_in tasks (consumer side)
                                    if debug and prod_task_name.startswith('stage_in-'):
                                        print(f"  cons_aggregateFilesizeMB: {cons_aggregateFilesizeMB}")
                                        print(f"  cons_estimated_trMiB: {cons_estimated_trMiB}")
                                        print(f"  estT_cons: {estT_cons}")

                                    # Validate that estT values are positive
                                    if estT_cons <= 0:
                                        if debug:
                                            print(f"[WARNING] Negative/zero estT_cons: {estT_cons}, cons_aggregateFilesizeMB: {cons_aggregateFilesizeMB}, cons_estimated_trMiB: {cons_estimated_trMiB}")
                                        continue

                                    # Calculate SPM based on producer type
                                    if prod_op == 'none':
                                        estT_prod = 0.0  # Stage_in tasks with operation "none" always have SPM = 0
                                    SPM = estT_prod + estT_cons

                                    # Debug output for final SPM calculation
                                    if debug and prod_task_name.startswith('stage_in-'):
                                        print(f"  Final SPM: {SPM}")
                                        print(f"  Edge key: {prod_storage}_{n_prod.replace('p', '')}_{cons_storage}_{n_cons}")
                                        print("  ---")

                                    # Use both producer and consumer storage in the key
                                    edge_key = f'{prod_storage}_{n_prod.replace("p", "")}_{cons_storage}_{n_cons}'
                                    edge_attributes[f'estT_prod_{prod_storage}_{n_prod}'] = estT_prod
                                    edge_attributes[f'estT_cons_{cons_storage}_{n_cons}'] = estT_cons
                                    # Don't calculate SPM here - it will be calculated after averaging estT values
                                    edge_count += 1
                    # Add debug print if no edge_attributes were set
                    if not edge_attributes and debug:
                        print(f"[WARNING] No SPM/estT values for edge {prod_task_name} -> {consumer_row['taskName']}")
                    # Add common edge attributes
                    edge_attributes.update({
                        'prod_aggregateFilesizeMB': producer_row['aggregateFilesizeMB'],
                        'cons_aggregateFilesizeMB': consumer_row['aggregateFilesizeMB'],
                        'prod_max_parallelism': producer_row['parallelism'],
                        'cons_max_parallelism': consumer_row['parallelism'],
                        'prod_stage_order': producer_row['stageOrder'],
                        'cons_stage_order': consumer_row['stageOrder'],
                        'prod_task_name': prod_task_name,
                        'cons_task_name': consumer_row['taskName'],
                        'file_name': prod_fileName,
                    })
                    if WFG.has_edge(prod_node_name, cons_node_name):
                        WFG.edges[prod_node_name, cons_node_name].update(edge_attributes)
                    else:
                        WFG.add_edge(prod_node_name, cons_node_name, **edge_attributes)
    
    if debug:
        print(f"Processed {processed_pairs} pairs, created {edge_count} edge attributes, total edges: {len(WFG.edges)}")
        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    # Save intermediate results as CSV before exiting
    if workflow_name:
        try:
            import os
            # Create the output directory if it doesn't exist
            output_dir = "workflow_spm_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert the graph edges to a DataFrame for CSV export
            edge_data = []
            for edge in WFG.edges(data=True):
                producer_node, consumer_node, attributes = edge
                
                # Extract node information
                prod_node_data = WFG.nodes[producer_node]
                cons_node_data = WFG.nodes[consumer_node]
                
                
                # Create base row with common information
                base_row = {
                    'producer_node': producer_node,
                    'consumer_node': consumer_node,
                    'producer_task': prod_node_data.get('taskName', ''),
                    'consumer_task': cons_node_data.get('taskName', ''),
                    'producer_stage_order': attributes.get('prod_stage_order', ''),
                    'consumer_stage_order': attributes.get('cons_stage_order', ''),
                    'producer_operation': convert_operation_to_string(prod_node_data.get('operation')),
                    'consumer_operation': convert_operation_to_string(cons_node_data.get('operation')),
                    'producer_storage': prod_node_data.get('storageType', ''),
                    'consumer_storage': cons_node_data.get('storageType', ''),
                    'producer_parallelism': prod_node_data.get('parallelism', ''),
                    'consumer_parallelism': cons_node_data.get('parallelism', ''),
                    'producer_filesize_mb': prod_node_data.get('aggregateFilesizeMB', ''),
                    'consumer_filesize_mb': cons_node_data.get('aggregateFilesizeMB', ''),
                    'file_name': attributes.get('file_name', ''),
                }
                
                # Add main edge attributes (non-estT values)
                for key, value in attributes.items():
                    if key not in ['producer_node', 'consumer_node', 'producer_task', 'consumer_task']:
                        if not key.startswith('estT_prod_') and not key.startswith('estT_cons_'):
                            base_row[f'edge_{key}'] = value
                
                # Create separate rows for each estT key combination
                prod_estT_keys = []
                cons_estT_keys = []
                
                for key in attributes.keys():
                    if key.startswith('estT_prod_'):
                        prod_estT_keys.append(key)
                    elif key.startswith('estT_cons_'):
                        cons_estT_keys.append(key)
                
                # If we have estT keys, create separate rows for each combination
                if prod_estT_keys and cons_estT_keys:
                    for prod_key in prod_estT_keys:
                        for cons_key in cons_estT_keys:
                            row = base_row.copy()
                            row['estT_prod_key'] = prod_key
                            row['estT_cons_key'] = cons_key
                            edge_data.append(row)
                else:
                    # If no estT keys, just add the base row
                    edge_data.append(base_row)
            
            # Create DataFrame and save to CSV
            if edge_data:
                edge_df = pd.DataFrame(edge_data)
                csv_filename = f"{workflow_name}_intermediate_estT_results.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                edge_df.to_csv(csv_path, index=False)
                if debug:
                    print(f"   Saved intermediate estT results to: {csv_path}")
                    print(f"   Total rows saved: {len(edge_df)}")
                    print(f"   CSV columns: {list(edge_df.columns)}")
                    print(f"   Main columns: {[col for col in edge_df.columns if not col.startswith('edge_') and not col.startswith('estT_')]}")
                    print(f"   Edge columns: {[col for col in edge_df.columns if col.startswith('edge_')]}")
                    print(f"   estT columns: {[col for col in edge_df.columns if col.startswith('estT_')]}")
                    print(f"   Structure: One row per estT key combination (keys only, no values)")
            else:
                if debug:
                    print("   No edges found to save in intermediate results")
                    
        except Exception as e:
            if debug:
                print(f"   Warning: Could not save intermediate results: {e}")
    
    return WFG

def normalize_estT_values_g(SPM_estT_values: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Normalizes estT_prod, estT_cons, and SPM values globally across all producer-consumer pairs.

    Parameters:
    SPM_estT_values (dict): Dictionary containing 'estT_prod', 'estT_cons', 'SPM', 'dsize_cons', and 'dsize_prod' values.

    Returns:
    dict: A new dictionary with globally normalized values.
    """
    # Collect all values across the entire SPM_estT_values dictionary
    global_values = {key: [] for key in ['estT_prod', 'estT_cons', 'SPM', 'dsize_cons', 'dsize_prod']}
    for data in SPM_estT_values.values():
        for key in global_values.keys():
            global_values[key].extend(
                v for storage_n_values in data.get(key, {}).values() for v in storage_n_values
            )

    # Compute global min and max for each key
    global_min_max = {}
    for key, values in global_values.items():
        if values:
            global_min_max[key] = {'min': min(values), 'max': max(values)}
        else:
            global_min_max[key] = {'min': 0, 'max': 0}  # Default if no values

    # Create a new dictionary for normalized values
    normalized_SPM_estT_values = {}

    # Normalize each key within each producer-consumer pair
    for pair, data in SPM_estT_values.items():
        normalized_SPM_estT_values[pair] = {}
        for key in ['estT_prod', 'estT_cons', 'dsize_cons', 'dsize_prod']:
            min_val = global_min_max[key]['min']
            max_val = global_min_max[key]['max']

            normalized_SPM_estT_values[pair][key] = {}
            for storage_n, values in data.get(key, {}).items():
                if min_val == max_val:
                    normalized_values = [0.5] * len(values)  # If all values are the same
                else:
                    normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
                
                normalized_SPM_estT_values[pair][key][storage_n] = normalized_values

        # Normalize SPM with specific logic
        key = 'SPM'
        min_val = global_min_max[key]['min']
        max_val = global_min_max[key]['max']

        normalized_SPM_estT_values[pair][key] = {}
        for storage_n, values in data.get(key, {}).items():
            normalized_values = []
            for v in values:
                if v > 1:
                    # Normalize values > 1 into [1, 2]
                    normalized_value = 1 + (v - 1) / (max_val - 1) if max_val > 1 else 1
                else:
                    # Keep values <= 1 as is
                    normalized_value = v
                normalized_values.append(normalized_value)
            
            normalized_SPM_estT_values[pair][key][storage_n] = normalized_values

    return normalized_SPM_estT_values


def normalize_estT_values(SPM_estT_values: Dict[str, Dict[str, Any]], debug=False) -> Dict[str, Dict[str, Any]]:
    """
    Normalizes estT_prod, estT_cons, and SPM values across storage_n values within each producer-consumer pair.

    Parameters:
    SPM_estT_values (dict): Dictionary containing 'estT_prod', 'estT_cons', 'SPM', 'dsize_cons', and 'dsize_prod' values.
    debug (bool): Boolean to control debug output (default: False)

    Returns:
    dict: A new dictionary with normalized values within each pair.
    """
    if debug:
        print("Normalizing estT_prod, estT_cons, and SPM values within each producer-consumer pair.")
    # Create a new dictionary for normalized values
    normalized_SPM_estT_values = {}

    # Normalize each key within each producer-consumer pair
    for pair, data in SPM_estT_values.items():
        normalized_SPM_estT_values[pair] = {}

        # Compute pair-level min and max for each key
        pair_min_max = {}
        for key in ['estT_prod', 'estT_cons', 'SPM', 'dsize_cons', 'dsize_prod']:
            all_values = [
                v for storage_n_values in data.get(key, {}).values() for v in storage_n_values
            ]
            if all_values:
                pair_min_max[key] = {'min': min(all_values), 'max': max(all_values)}
            else:
                pair_min_max[key] = {'min': 0, 'max': 0}  # Default if no values

        # Normalize each key using pair-level min and max
        for key in ['estT_prod', 'estT_cons', 'dsize_cons', 'dsize_prod']:
            min_val = pair_min_max[key]['min']
            max_val = pair_min_max[key]['max']

            normalized_SPM_estT_values[pair][key] = {}
            for storage_n, values in data.get(key, {}).items():
                if min_val == max_val:
                    normalized_values = [0.5] * len(values)  # If all values are the same
                else:
                    normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
                
                normalized_SPM_estT_values[pair][key][storage_n] = normalized_values

        # Normalize SPM with specific logic for each pair
        key = 'SPM'
        min_val = pair_min_max[key]['min']
        max_val = pair_min_max[key]['max']

        normalized_SPM_estT_values[pair][key] = {}
        for storage_n, values in data.get(key, {}).items():
            normalized_values = []
            for v in values:
                if v > 1:
                    # Normalize values > 1 into [1, 2]
                    normalized_value = 1 + (v - 1) / (max_val - 1) if max_val > 1 else 1
                else:
                    # Keep values <= 1 as is
                    normalized_value = v
                normalized_values.append(normalized_value)
            
            normalized_SPM_estT_values[pair][key][storage_n] = normalized_values

    return normalized_SPM_estT_values


def calculate_averages_and_rank(SPM_estT_values, debug=False):
    """
    Averages list values for estT_prod and estT_cons, calculates SPM as ave_estT_prod + ave_estT_cons,
    and calculates rank for each storage_n.
    
    The SPM calculation now follows the correct approach:
    1. Average all estT_prod values for the same producer task
    2. Average all estT_cons values for the same consumer task  
    3. Calculate SPM = ave_estT_prod + ave_estT_cons
    
    rank = ave_dsize * (estT_prod + estT_cons) * abs(SPM - 1)
    
    Parameters:
    SPM_estT_values (dict): Dictionary containing 'estT_prod', 'estT_cons', and 'dsize' values.
    debug (bool): Boolean to control debug output (default: False)

    Returns:
    dict: Updated dictionary with averaged values, calculated SPM, and calculated ranks.
    """
    if debug:
        print(f"Calculating averages and ranks for {len(SPM_estT_values)} producer-consumer pairs...")
        
    for pair, data in SPM_estT_values.items():
        # Initialize a dictionary for ranks if not present
        if 'rank' not in data:
            data['rank'] = {}
        
        # Get all unique storage combinations from estT_prod keys
        storage_combinations = set()
        for prod_key in data['estT_prod'].keys():
            # Parse the producer key to get storage and parallelism info
            parts = prod_key.split('_')
            if len(parts) >= 2:
                prod_storage = parts[0]
                n_prod = parts[1].replace('p', '')
                
                # Find corresponding consumer keys for this producer storage
                for cons_key in data['estT_cons'].keys():
                    cons_parts = cons_key.split('_')
                    if len(cons_parts) >= 2:
                        cons_storage = cons_parts[0]
                        n_cons = cons_parts[1].replace('p', '')
                        
                        # Create storage combination key
                        storage_n = f"{prod_storage}_{n_prod}_{cons_storage}_{n_cons}p"
                        storage_combinations.add(storage_n)
        
        for storage_n in storage_combinations:
            # Parse the new format: {prod_storage}_{n_prod}_{cons_storage}_{n_cons}
            # Example: beegfs_1_ssd_24p -> prod_key: beegfs_1p, cons_key: ssd_24p
            parts = storage_n.split('_')
            if len(parts) >= 4:
                prod_storage = parts[0]
                n_prod = parts[1]
                cons_storage = parts[2]
                n_cons = parts[3]
                
                prod_key = f"{prod_storage}_{n_prod}p"
                cons_key = f"{cons_storage}_{n_cons}"
                prod_par = int(n_prod)
                cons_par = int(n_cons.replace('p', ''))
            else:
                # Fallback for old format compatibility
                prod_key = f"{'_'.join(storage_n.split('_')[:2])}p"
                cons_key = f"{storage_n.split('_')[0]}_{storage_n.split('_')[2][:-1]}p"
                prod_par = int(prod_key.split('_')[1].replace('p',''))
                cons_par = int(cons_key.split('_')[1].replace('p',''))
            
            # Average estT_prod
            estT_prod_values = data['estT_prod'].get(prod_key, [])
            avg_estT_prod = sum(estT_prod_values) / len(estT_prod_values) if estT_prod_values else 0.0
            # Average estT_cons
            estT_cons_values = data['estT_cons'].get(cons_key, [])
            avg_estT_cons = sum(estT_cons_values) / len(estT_cons_values) if estT_cons_values else 0.0
            
            # Calculate SPM after averaging estT values
            if avg_estT_cons > 0:
                avg_spm = avg_estT_prod + avg_estT_cons
            else:
                avg_spm = avg_estT_prod
            # Average dsize (corrected key access)
            dsize_prod_values = data['dsize_prod'].get("", [])  # Access dsize using the empty key as seen in the normalized output
            ave_prod_dsize = sum(dsize_prod_values) #/ len(dsize_prod_values) if dsize_prod_values else 1.0  # Default to 1.0 if no dsize
            dsize_cons_values = data['dsize_cons'].get("", [])  # Access dsize using the empty key as seen in the normalized output
            ave_cons_dsize = sum(dsize_cons_values) #/ len(dsize_cons_values) if dsize_cons_values else 1.0  # Default to 1.0 if no dsize
            
            if MULTI_NODES == False:
                prod_max_parallelism = data['par_prod'].get("", [])[0] # all should be same
                cons_max_parallelism = data['par_cons'].get("", [])[0] # all should be same
                prod_seq_tasks = prod_max_parallelism/prod_par
                cons_seq_tasks = cons_max_parallelism/cons_par
                # Calculate rank
                rank = (prod_seq_tasks * ave_prod_dsize * avg_estT_prod + cons_seq_tasks * cons_seq_tasks * avg_estT_cons)
            else:
                rank = avg_estT_prod  + avg_estT_cons 
            

            # Store averages back in the dictionary for reference
            data['estT_prod'][prod_key] = [avg_estT_prod]  # Replace list with a single averaged value
            data['estT_cons'][cons_key] = [avg_estT_cons]
            data['SPM'][storage_n] = [avg_spm]
            data['dsize_prod'][storage_n] = [ave_prod_dsize]
            data['dsize_cons'][storage_n] = [ave_cons_dsize]

            # Store the calculated rank
            data['rank'][storage_n] = [rank]

    if debug:
        print(f"Completed averaging and ranking calculations.")
    return SPM_estT_values


def calculate_sums_and_rank(SPM_estT_values, debug=False):
    """
    Sums list values for estT_prod, estT_cons, SPM, and dsize, and calculates rank for each storage_n.
        
    Parameters:
    SPM_estT_values (dict): Dictionary containing 'estT_prod', 'estT_cons', 'SPM', and 'dsize' values.
    debug (bool): Boolean to control debug output (default: False)

    Returns:
    dict: Updated dictionary with summed values and calculated ranks.
    """
    if debug:
        print(f"Calculating sums and ranks for {len(SPM_estT_values)} producer-consumer pairs...")
        
    for pair, data in SPM_estT_values.items():
        # Initialize a dictionary for ranks if not present
        if 'rank' not in data:
            data['rank'] = {}
        
        # get total dsize of all producers and consumers
        total_dsize_prod = sum([sum(v) for k, v in data['dsize_prod'].items()])
        total_dsize_cons = sum([sum(v) for k, v in data['dsize_cons'].items()])
                
        if data['SPM'].keys():

            for storage_n in data['SPM'].keys():
                # Parse the new format: {prod_storage}_{n_prod}_{cons_storage}_{n_cons}
                # Example: beegfs_1_ssd_24p -> prod_key: beegfs_1p, cons_key: ssd_24p
                parts = storage_n.split('_')
                if len(parts) >= 4:
                    prod_storage = parts[0]
                    n_prod = parts[1]
                    cons_storage = parts[2]
                    n_cons = parts[3]
                    
                    prod_key = f"{prod_storage}_{n_prod}p"
                    cons_key = f"{cons_storage}_{n_cons}"
                    prod_par = int(n_prod)
                    cons_par = int(n_cons.replace('p', ''))
                else:
                    # Fallback for old format compatibility
                    prod_key = f"{'_'.join(storage_n.split('_')[:2])}p"
                    cons_key = f"{storage_n.split('_')[0]}_{storage_n.split('_')[2][:-1]}p"
                    prod_par = int(prod_key.split('_')[1].replace('p', ''))
                    cons_par = int(cons_key.split('_')[1].replace('p', ''))
                
                # Sum estT_prod
                estT_prod_values = data['estT_prod'].get(prod_key, [])
                sum_estT_prod = sum(estT_prod_values) #if estT_prod_values else 0.0
                # Sum estT_cons
                estT_cons_values = data['estT_cons'].get(cons_key, [])
                sum_estT_cons = sum(estT_cons_values) #if estT_cons_values else 0.0
                # Average SPM
                spm_values = data['SPM'].get(storage_n, [])
                avg_spm = sum(spm_values) / len(spm_values) if spm_values else 0.0
                # Sum dsize (corrected key access)
                dsize_prod_values = data['dsize_prod'].get('prod_aggregateFilesizeMB', [])
                sum_prod_dsize = sum(dsize_prod_values) #if dsize_prod_values else 1.0  # Default to 1.0 if no dsize
                dsize_cons_values = data['dsize_cons'].get('cons_aggregateFilesizeMB', [])
                sum_cons_dsize = sum(dsize_cons_values) #if dsize_cons_values else 1.0  # Default to 1.0 if no dsize
                
                if MULTI_NODES == False:
                    prod_max_parallelism = data['par_prod'].get("", [])[0]  # all should be same
                    cons_max_parallelism = data['par_cons'].get("", [])[0]  # all should be same
                    prod_seq_tasks = prod_max_parallelism / prod_par
                    cons_seq_tasks = cons_max_parallelism / cons_par
                    # Calculate rank
                    rank = (prod_seq_tasks * sum_prod_dsize * sum_estT_prod +
                            cons_seq_tasks * sum_cons_dsize * sum_estT_cons)
                else:
                    prod_time_weight = sum_prod_dsize / (sum_prod_dsize + sum_cons_dsize)
                    cons_time_weight = sum_cons_dsize / (sum_prod_dsize + sum_cons_dsize)
                    
                    rank = prod_time_weight * sum_estT_prod + cons_time_weight * sum_estT_cons
                

                
                # Store sums back in the dictionary for reference
                data['estT_prod'][prod_key] = [sum_estT_prod]  # Replace list with a single summed value
                data['estT_cons'][cons_key] = [sum_estT_cons]
                data['SPM'][storage_n] = [avg_spm]  # Still using the average for SPM
                data['dsize_prod'][storage_n] = [sum_prod_dsize]
                data['dsize_cons'][storage_n] = [sum_cons_dsize]

                # Store the calculated rank
                data['rank'][storage_n] = [rank]

        else:
            # Handle case where no producer data is available
            pass

    if debug:
        print(f"Completed sum and ranking calculations for {len(SPM_estT_values)} producer-consumer pairs.")
    return SPM_estT_values


def filter_storage_options(combined_SPM_estT_values: Dict[str, Dict[str, Any]], 
                          workflow_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Filter storage options based on workflow-specific rules.
    
    Parameters:
    - combined_SPM_estT_values: Dictionary containing SPM values for all producer-consumer pairs
    - workflow_name: Name of the workflow for filtering rules
    
    Returns:
    - Dict[str, Dict[str, Any]]: Filtered SPM values
    """
    # For now, return the original values without filtering
    # This can be extended with workflow-specific filtering logic
    return combined_SPM_estT_values

def display_top_sorted_averaged_rank(combined_SPM_estT_values: Dict[str, Dict[str, Any]], 
                                    baseline: float = 0, top_n: int = 5) -> None:
    """
    Displays the top N storage_n selections based on averaged rank values closest to the baseline.
    
    Args:
        combined_SPM_estT_values (dict): Dictionary containing averaged SPM and rank values.
        baseline (float): Baseline to calculate closeness for sorting.
        top_n (int): Number of top results to display.
    """
    print(f"Top {top_n} Averaged SPM Values Closest to Baseline = {baseline}:\n")

    for pair, data in combined_SPM_estT_values.items():
        producer, consumer = pair.split(":")
        
        # Check if this is a virtual producer-consumer pair for initial tasks
        is_virtual_pair = producer.startswith('stage_in-')
        if is_virtual_pair:
            # Extract the actual task name from stage_in-{taskName}
            actual_task_name = producer.replace('stage_in-', '')
            print(f"Producer: {producer} (Stage-in task)")
            print(f"Consumer: {consumer}")
        else:
            print(f"Producer: {producer}, Consumer: {consumer}")
        
        # Check if this is the filtered structure (storage options as keys) or original structure (with 'rank' key)
        if 'rank' in data:
            # Original structure: data has 'rank' key
            rank_data = data['rank']
        else:
            # Filtered structure: storage options are keys, each containing 'rank'
            rank_data = {}
            for storage_n, storage_data in data.items():
                if 'rank' in storage_data:
                    rank_data[storage_n] = storage_data['rank']
        
        if not rank_data:
            print(f"  No rank data found for this pair")
            print()
            continue
        
        # Collect and sort rank values by closeness to the baseline
        sorted_spm = sorted(
            rank_data.items(),
            key=lambda item: abs(item[1][0] - baseline) if item[1] else float('inf')  # Handle empty rank values
        )
        
        top_n_displayed = 0
        
        for rank, (storage_n, avg_spm) in enumerate(sorted_spm, start=1):
            avg_spm_value = avg_spm[0] if avg_spm else float('inf')  # Extract the actual value
            print(f"- Rank {rank}: {storage_n} with Averaged rank = {avg_spm_value}")
            top_n_displayed += 1
            if top_n_displayed >= top_n:
                break
        print()  # Blank line for readability


def select_best_storage_and_parallelism(combined_SPM_estT_values: Dict[str, Dict[str, Any]], 
                                       baseline: float = 0) -> Dict[str, Dict[str, Any]]:
    """
    Selects the best storage type and parallelism level for each producer-consumer pair
    based on rank values and averages the rank of each storage type.

    Parameters:
    combined_SPM_estT_values (dict): Dictionary containing averaged SPM, estT values, and rank.
    baseline (float): Baseline for comparison.

    Returns:
    dict: Results containing the best storage type, parallelism, and averaged ranks.
    """
    print(f"Selecting best storage configurations for {len(combined_SPM_estT_values)} pairs...")
    
    results = {}  # Store the best storage, parallelism, and averaged ranks for each pair

    for pair, data in combined_SPM_estT_values.items():
        producer, consumer = pair.split(":")
        
        # Check if this is the filtered structure (storage options as keys) or original structure (with 'rank' key)
        if 'rank' in data:
            # Original structure: data has 'rank' key
            rank_data = data['rank']
        else:
            # Filtered structure: storage options are keys, each containing 'rank'
            rank_data = {}
            for storage_n, storage_data in data.items():
                if 'rank' in storage_data:
                    rank_data[storage_n] = storage_data['rank']
        
        if not rank_data:
            continue
        
        # Group rank values by producer storage type
        storage_groups = {}
        for storage_n, rank_values in rank_data.items():
            # Parse the new format: {prod_storage}_{n_prod}_{cons_storage}_{n_cons}
            parts = storage_n.split("_")
            if len(parts) >= 4:
                prod_storage_type = parts[0]  # Extract producer storage type (e.g., 'beegfs' or 'ssd')
            else:
                # Fallback for old format compatibility
                prod_storage_type = storage_n.split("_")[0]
            
            if prod_storage_type not in storage_groups:
                storage_groups[prod_storage_type] = []
            storage_groups[prod_storage_type].append((storage_n, rank_values[0]))  # Add rank values

        # Compute the average rank for each storage type
        averaged_storage_ranks = {}
        for storage_type, ranks in storage_groups.items():
            avg_rank = sum(rank for _, rank in ranks) / len(ranks) if ranks else float('inf')
            averaged_storage_ranks[storage_type] = avg_rank

        # Rank storage types by their average rank
        sorted_storages = sorted(averaged_storage_ranks.items(), key=lambda x: x[1])

        # Select the best storage type and its parallelism level
        best_storage_type = sorted_storages[0][0]
        best_storage_avg_rank = sorted_storages[0][1]

        # Find the best parallelism level within the best storage type
        best_parallelism, best_rank = min(storage_groups[best_storage_type], key=lambda x: x[1])

        # Store the results for this pair
        results[pair] = {
            "best_storage_type": best_storage_type,
            "best_parallelism": best_parallelism,
            "best_rank": best_rank,
            "avg_rank_by_storage": averaged_storage_ranks
        }

    print(f"Selected best storage configurations for {len(results)} pairs.")
    return results

def extract_SPM_estT_values(WFG):
    """
    Extract and store weighted SPM values for each producer-consumer pair, 
    including initial stage 1 nodes where the producer is 'initial_data'.
    
    Args:
        WFG (nx.DiGraph): A directed weighted graph with nodes and edges containing performance attributes.
    
    Returns:
        dict: Weighted SPM dictionary containing producer-consumer pairs, their SPM values, and task IO times.
    """
    SPM_estT_values = {}

    # Iterate through all edges in the graph
    for edge in WFG.edges(data=True):
        producer_node, consumer_node, attributes = edge
        prod_cons_pair = f"{WFG.nodes[producer_node]['taskName']}:{WFG.nodes[consumer_node]['taskName']}"
        
        if prod_cons_pair not in SPM_estT_values:
            SPM_estT_values[prod_cons_pair] = {
                'SPM': {},
                'estT_prod': {},
                'estT_cons': {},
                'rank': {},
                'par_prod': {},
                'par_cons': {},
                'dsize_prod': {},
                'dsize_cons': {},
            }
        
        for key, value in attributes.items():
            if key.startswith("SPM_"):
                storage_n = key.replace("SPM_", "")
                if storage_n not in SPM_estT_values[prod_cons_pair]['SPM']:
                    SPM_estT_values[prod_cons_pair]['SPM'][storage_n] = []
                SPM_estT_values[prod_cons_pair]['SPM'][storage_n].append(value)
            elif key.startswith("estT_prod_"):
                storage_n = key.replace("estT_prod_", "")
                if storage_n not in SPM_estT_values[prod_cons_pair]['estT_prod']:
                    SPM_estT_values[prod_cons_pair]['estT_prod'][storage_n] = []
                SPM_estT_values[prod_cons_pair]['estT_prod'][storage_n].append(value)
            elif key.startswith("estT_cons_"):
                storage_n = key.replace("estT_cons_", "")
                if storage_n not in SPM_estT_values[prod_cons_pair]['estT_cons']:
                    SPM_estT_values[prod_cons_pair]['estT_cons'][storage_n] = []
                SPM_estT_values[prod_cons_pair]['estT_cons'][storage_n].append(value)
            elif key == "prod_aggregateFilesizeMB":
                if "prod_aggregateFilesizeMB" not in SPM_estT_values[prod_cons_pair]['dsize_prod']:
                    SPM_estT_values[prod_cons_pair]['dsize_prod']['prod_aggregateFilesizeMB'] = []
                SPM_estT_values[prod_cons_pair]['dsize_prod']['prod_aggregateFilesizeMB'].append(value)
            elif key == "cons_aggregateFilesizeMB":
                if "cons_aggregateFilesizeMB" not in SPM_estT_values[prod_cons_pair]['dsize_cons']:
                    SPM_estT_values[prod_cons_pair]['dsize_cons']['cons_aggregateFilesizeMB'] = []
                SPM_estT_values[prod_cons_pair]['dsize_cons']['cons_aggregateFilesizeMB'].append(value)
            elif key == "prod_max_parallelism":
                if "prod_max_parallelism" not in SPM_estT_values[prod_cons_pair]['par_prod']:
                    SPM_estT_values[prod_cons_pair]['par_prod']['prod_max_parallelism'] = []
                SPM_estT_values[prod_cons_pair]['par_prod']['prod_max_parallelism'].append(value)
            elif key == "cons_max_parallelism":
                if "cons_max_parallelism" not in SPM_estT_values[prod_cons_pair]['par_cons']:
                    SPM_estT_values[prod_cons_pair]['par_cons']['cons_max_parallelism'] = []
                SPM_estT_values[prod_cons_pair]['par_cons']['cons_max_parallelism'].append(value)



    return SPM_estT_values

def calculate_spm_for_workflow(wf_df: pd.DataFrame, debug: bool = False, workflow_name: str = None) -> dict:
    """
    Calculate SPM values for the entire workflow, matching the robust logic of the original notebook.
    Now also accounts for cp and scp operation rows as producers and consumers in the workflow graph.
    
    For tasks that have no preceding tasks (initial tasks), creates virtual producer-consumer pairs
    with a time=0 write operation from a virtual producer to calculate SPM values for read operations.
    
    Parameters:
    - wf_df: DataFrame containing workflow task data
    - debug: Boolean to control debug output (default: False)
    - workflow_name: Name of the workflow for logging intermediate results (default: None)
    
    Returns:
    - dict: SPM values for all producer-consumer pairs, including virtual pairs for initial tasks
    """
    WFG, stage_task_node_dict, stage_order_list = add_workflow_graph_nodes(wf_df, verbose=debug)

    # Debug: Print stage order list and task node dictionary
    if debug:
        print(f"Stage order list: {stage_order_list}")
        print(f"Number of nodes: {len(WFG.nodes)}")
        print("First five nodes:")
        for i, (node_name, node_data) in enumerate(list(WFG.nodes(data=True))[:5]):
            print(f"({node_name}, {node_data})")

    # Collect all unique stage orders, including fractional ones (e.g., 0.5, 1, 1.5, 2, 2.5, ...)
    all_stage_orders = sorted(set(wf_df['stageOrder'].unique()), key=lambda x: float(x))

    # Collect all producer-consumer pairs to process in a single pass
    all_producer_consumer_pairs = []

    # Debug: Check for stage_out tasks
    stage_out_tasks = []
    for stageOrder, task_dict in stage_task_node_dict.items():
        for taskName, nodeNames in task_dict.items():
            if 'stage_out' in str(taskName):
                stage_out_tasks.append((stageOrder, taskName, nodeNames))
    
    if debug:
        print(f"Found {len(stage_out_tasks)} stage_out tasks")

    # First pass: collect regular stage-to-stage connections
    for currOrder in all_stage_orders:
        cons_nodes = stage_task_node_dict.get(currOrder, {})
        # Skip stage 0 if it exists (legacy workflows) - should not occur with 1-based numbering
        if currOrder == 0:
            if debug:
                print(f"Warning: Found stage 0 in workflow data - this should not occur with 1-based stage numbering")
            continue
        else:
            # Add edges between producer and consumer nodes
            prevOrder = None
            # For fractional stageOrders (e.g., 0.5, 1.5, 2.5), connect to the previous integer stage
            if isinstance(currOrder, float) and currOrder % 1 != 0:
                prevOrder = float(currOrder)
            else:
                prevOrder = currOrder - 1.0
            prod_nodes = stage_task_node_dict.get(prevOrder, {})
            if prod_nodes and cons_nodes:
                all_producer_consumer_pairs.append((prod_nodes, cons_nodes))

    # Second pass: collect cp/scp connections and stage_out connections
    if debug:
        print("\n=== Processing producer-consumer pairs ===")
    for stageOrder, task_dict in stage_task_node_dict.items():
        for taskName, nodeNames in task_dict.items():
            for nodeName in nodeNames:
                node_data = WFG.nodes[nodeName]
                op = node_data.get('operation')
                op_str = convert_operation_to_string(op)
                
                # Handle stage_in connections (cp/scp nodes as producers)
                if op_str in ['none','cp', 'scp'] and str(taskName).startswith('stage_in-'):
                    # Extract the actual task name from stage_in-{taskName}
                    actual_task_name = taskName.replace('stage_in-', '')
                    # Find the next integer stage where the actual task should be
                    next_stage = float(stageOrder) + 0.5 if float(stageOrder) % 1 == 0 else float(stageOrder) + 1.0
                    next_tasks = stage_task_node_dict.get(next_stage, {})
                    if actual_task_name in next_tasks:
                        all_producer_consumer_pairs.append(({taskName: [nodeName]}, {actual_task_name: next_tasks[actual_task_name]}))
                
                # Handle stage_out connections (cp/scp nodes as consumers)
                elif op_str in ['cp', 'scp'] and str(taskName).startswith('stage_out-'):
                    # Find the previous stage (stageOrder-0.5 or int-1)
                    prev_stage = float(stageOrder) - 0.5 if float(stageOrder) % 1 == 0 else float(stageOrder) - 1.0
                    prev_tasks = stage_task_node_dict.get(prev_stage, {})
                    if prev_tasks:
                        all_producer_consumer_pairs.append((prev_tasks, {taskName: [nodeName]}))
                
                # Handle regular task nodes as producers for stage_out nodes
                elif op_str in ['read', 'write']:  # Regular task nodes (read/write operations)
                    # Check if there are stage_out nodes for this task at stageOrder + 0.5
                    next_stage = float(stageOrder) + 0.5
                    next_tasks = stage_task_node_dict.get(next_stage, {})
                    if next_tasks:
                        # Find stage_out nodes for this task
                        stage_out_task_name = f'stage_out-{taskName}'
                        if stage_out_task_name in next_tasks:
                            # if debug:
                            #     print(f"Found regular task -> stage_out connection: {taskName} -> {stage_out_task_name}")
                            all_producer_consumer_pairs.append(({taskName: [nodeName]}, {stage_out_task_name: next_tasks[stage_out_task_name]}))

                
                # Handle final data movement (terminal nodes)
                elif str(taskName).startswith('stage_out-final'):
                    # These are terminal nodes and don't need additional processing
                    pass

    # Debug: Print all collected pairs
    if debug:
        print(f"Collected {len(all_producer_consumer_pairs)} producer-consumer pairs")

    # Note: Virtual producers are now handled by existing stage_in-{taskName} rows with operation "none"
    # These are created in workflow_data_staging.py and processed in the main loop above

    # Process all collected pairs in a single call to avoid redundant processing
    if all_producer_consumer_pairs:
        # Create a combined producer-consumer structure
        combined_prod_nodes = {}
        combined_cons_nodes = {}
        
        for prod_nodes, cons_nodes in all_producer_consumer_pairs:
            # Merge producer nodes
            for task_name, node_list in prod_nodes.items():
                if task_name not in combined_prod_nodes:
                    combined_prod_nodes[task_name] = []
                combined_prod_nodes[task_name].extend(node_list)
            
            # Merge consumer nodes
            for task_name, node_list in cons_nodes.items():
                if task_name not in combined_cons_nodes:
                    combined_cons_nodes[task_name] = []
                combined_cons_nodes[task_name].extend(node_list)
        
        # Process all edges in a single call
        add_producer_consumer_edge(WFG, combined_prod_nodes, combined_cons_nodes, debug=debug, workflow_name=workflow_name)

    all_SPM_estT_values = extract_SPM_estT_values(WFG)
    if NORMALIZE:
        all_SPM_estT_values = normalize_estT_values_g(all_SPM_estT_values)
    all_SPM_estT_values = calculate_averages_and_rank(all_SPM_estT_values, debug=debug)
    return all_SPM_estT_values 