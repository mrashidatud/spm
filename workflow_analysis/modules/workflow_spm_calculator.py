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
    
    # Case 1: Producer is stage_in-{taskName} and consumer is the corresponding taskName
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
            return False
    
    # Case 2: Producer is taskName and consumer is stage_out-{taskName}
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
            return False
    
    # Case 3: General cross-storage matching for cp/scp operations
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

def add_producer_consumer_edge(WFG, prod_nodes, cons_nodes, debug=False):
    """
    Add edges between producer and consumer nodes, including cp/scp nodes and handling fileName as a list for those.
    """
    import pandas as pd
    import re
    
    if debug:
        print(f"\n=== Processing producer-consumer edges ===")

    # Step 1: Create producer DataFrame (operation in [0, 2, 3])
    producer_data = []
    for node_name, node_data in WFG.nodes(data=True):
        op = node_data.get('operation')
        task_name = node_data.get('taskName', '')
        
        # Producers: regular tasks (op 0,1) and stage_in tasks (cp/scp)
        if (op in [0, 1, 2, 3, 'cp', 'scp', 'read', 'write'] and 
            (op in [0, 1, 'read', 'write'] or 'stage_in' in task_name)):
            producer_data.append({
                'node_name': node_name,
                'taskName': node_data.get('taskName'),
                'fileName': node_data.get('fileName', '').strip(),
                'opCount': node_data.get('opCount'),
                'aggregateFilesizeMB': node_data.get('aggregateFilesizeMB'),
                'parallelism': node_data.get('parallelism'),
                'operation': op,
                'node_data': node_data
            })
    producer_df = pd.DataFrame(producer_data)
    if debug:
        print(f"Producers: {len(producer_df)} nodes")

    # Step 2: Create consumer DataFrame (operation in [1, 2, 3])
    consumer_data = []
    for node_name, node_data in WFG.nodes(data=True):
        op = node_data.get('operation')
        task_name = node_data.get('taskName', '')
        
        # Consumers: regular tasks (op 0,1) and stage_out tasks (cp/scp)
        if (op in [0, 1, 2, 3, 'cp', 'scp', 'read', 'write'] and 
            (op in [0, 1, 'read', 'write'] or 'stage_out' in task_name)):
            consumer_data.append({
                'node_name': node_name,
                'taskName': node_data.get('taskName'),
                'prevTask': node_data.get('prevTask'),
                'fileName': node_data.get('fileName', '').strip(),
                'opCount': node_data.get('opCount'),
                'aggregateFilesizeMB': node_data.get('aggregateFilesizeMB'),
                'parallelism': node_data.get('parallelism'),
                'operation': op,
                'node_data': node_data
            })
    consumer_df = pd.DataFrame(consumer_data)
    if debug:
        print(f"Consumers: {len(consumer_df)} nodes")

    # Debug: Print producer and consumer task names
    if debug:
        print(f"Producer tasks: {sorted(producer_df['taskName'].unique())}")
        print(f"Consumer tasks: {sorted(consumer_df['taskName'].unique())}")
    
    # Debug: Check for potential missing pairs based on naming patterns
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
        if expected_consumer in consumer_tasks:
            if debug:
                print(f"✓ Found matching pair: {prod_task} -> {expected_consumer}")
        else:
            missing_pairs.append((prod_task, expected_consumer))
            if debug:
                print(f"✗ Missing consumer: {expected_consumer} for producer: {prod_task}")

    # Check for orphaned stage_out consumers (consumers without matching producers)
    for cons_task in consumer_tasks:
        if cons_task.startswith('stage_out-'):
            expected_producer = cons_task.replace('stage_out-', '')
            if expected_producer not in producer_tasks:
                if debug:
                    print(f"✗ Orphaned consumer: {cons_task} (no matching producer: {expected_producer})")

    # Check for regular task relationships via prevTask
    for _, consumer_row in consumer_df.iterrows():
        prev_task = consumer_row.get('prevTask')
        if prev_task and prev_task not in producer_tasks:
            if debug:
                print(f"✗ Consumer {consumer_row['taskName']} references missing producer: {prev_task}")

    if debug and missing_pairs:
        print(f"\nSummary: {len(missing_pairs)} potential missing stage_out pairs found")

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
        # Special handling for stage_in tasks
        if prod_task_name.startswith('stage_in-'):
            # Extract the actual task name from stage_in-{taskName}
            actual_task_name = prod_task_name.replace('stage_in-', '')
            if debug:
                print(f"Processing stage_in connection: {prod_task_name} -> {actual_task_name}")
            
            # Find consumers with the actual task name
            if actual_task_name in consumer_df['taskName'].values:
                consumer_subset = consumer_df[consumer_df['taskName'] == actual_task_name]
                
                for _, producer_row in producer_group.iterrows():
                    prod_node_name = producer_row['node_name']
                    prod_fileName = producer_row['fileName']
                    prod_node_data = producer_row['node_data']
                    prod_op = producer_row['operation']

                    # If producer is cp/scp, treat fileName as a list
                    if prod_op in [2, 3, 'cp', 'scp']:
                        prod_file_list = re.split(r",|'", prod_fileName)
                        prod_file_list = [f.strip() for f in prod_file_list if f.strip()]
                    else:
                        prod_file_list = [prod_fileName]

                    # For each consumer, check matching logic
                    for _, consumer_row in consumer_subset.iterrows():
                        cons_node_name = consumer_row['node_name']
                        cons_fileName = consumer_row['fileName']
                        cons_node_data = consumer_row['node_data']
                        cons_op = consumer_row['operation']

                        # If consumer is cp/scp, treat fileName as a list
                        if cons_op in [2, 3, 'cp', 'scp']:
                            cons_file_list = re.split(r",|'", cons_fileName)
                            cons_file_list = [f.strip() for f in cons_file_list if f.strip()]
                        else:
                            cons_file_list = [cons_fileName]

                        # Matching logic for stage_in -> regular task
                        match = False
                        if prod_op in [2, 3, 'cp', 'scp']:
                            for f in cons_file_list:
                                if f in prod_file_list:
                                    match = True
                                    break
                        elif cons_op in [2, 3, 'cp', 'scp']:
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
                        
                        # Process edge attributes (same logic as below)
                        edge_attributes = {}
                        # First, find all available storage types from the actual keys
                        all_prod_storages = set()
                        all_cons_storages = set()
                        
                        for key in prod_node_data.keys():
                            if key.startswith('estimated_trMiB_'):
                                # Extract storage type from key like 'estimated_trMiB_beegfs-ssd_37p'
                                parts = key.split('_')
                                if len(parts) >= 3:
                                    storage_type = '_'.join(parts[2:-1])  # Handle compound storage types
                                    all_prod_storages.add(storage_type)
                        
                        for key in cons_node_data.keys():
                            if key.startswith('estimated_trMiB_'):
                                # Extract storage type from key like 'estimated_trMiB_beegfs-ssd_37p'
                                parts = key.split('_')
                                if len(parts) >= 3:
                                    storage_type = '_'.join(parts[2:-1])  # Handle compound storage types
                                    all_cons_storages.add(storage_type)
                        
                        # Try all combinations of producer and consumer storage types
                        for prod_storage in all_prod_storages:
                            for cons_storage in all_cons_storages:
                                # Find all producer and consumer keys for these storage types
                                prod_keys = [
                                    key for key in prod_node_data.keys()
                                    if key.startswith(f'estimated_trMiB_{prod_storage}_')
                                ]
                                cons_keys = [
                                    key for key in cons_node_data.keys()
                                    if key.startswith(f'estimated_trMiB_{cons_storage}_')
                                ]
                                
                                # For cp/scp, allow cross-storage matching
                                allow_cross = False
                                if prod_op in [2, 3, 'cp', 'scp'] or cons_op in [2, 3, 'cp', 'scp']:
                                    if is_valid_storage_match(prod_storage, cons_storage, prod_task_name, consumer_row['taskName']):
                                        allow_cross = True
                                # For non-cp/scp, require exact match
                                if (prod_op in [2, 3, 'cp', 'scp'] or cons_op in [2, 3, 'cp', 'scp']):
                                    if not allow_cross:
                                        continue
                                else:
                                    # Strict match
                                    if prod_storage != cons_storage:
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
                                        prod_aggregateFilesizeMB = producer_row['aggregateFilesizeMB']
                                        cons_aggregateFilesizeMB = consumer_row['aggregateFilesizeMB']
                                        prod_opCount = producer_row['opCount']
                                        cons_opCount = consumer_row['opCount']
                                        
                                        # For cp/scp, allow cross-storage matching
                                        allow_cross = False
                                        if prod_op in [2, 3, 'cp', 'scp'] or cons_op in [2, 3, 'cp', 'scp']:
                                            if is_valid_storage_match(prod_storage, cons_storage, prod_task_name, consumer_row['taskName']):
                                                allow_cross = True
                                        # For non-cp/scp, require exact match
                                        if (prod_op in [2, 3, 'cp', 'scp'] or cons_op in [2, 3, 'cp', 'scp']):
                                            if not allow_cross:
                                                continue
                                        else:
                                            # Strict match
                                            if prod_storage != cons_storage:
                                                continue
                                        
                                        # # estimated time calculation with op Count skewer
                                        # if prod_ts_slope > 0:
                                        #     estT_prod = prod_opCount * prod_aggregateFilesizeMB / prod_estimated_trMiB
                                        # else:
                                        #     estT_prod = (1/prod_opCount) * prod_aggregateFilesizeMB / prod_estimated_trMiB
                                        # if cons_ts_slope > 0:
                                        #     estT_cons = cons_opCount * cons_aggregateFilesizeMB / cons_estimated_trMiB 
                                        # else:
                                        #     estT_cons = (1/cons_opCount) * cons_aggregateFilesizeMB / cons_estimated_trMiB
                                        
                                        # estimated time calculation without op Count skewer
                                        estT_prod = prod_aggregateFilesizeMB / prod_estimated_trMiB
                                        estT_cons = cons_aggregateFilesizeMB / cons_estimated_trMiB

                                        if debug:
                                            if estT_prod < 0:
                                                print(f"estT_prod: {estT_prod}, prod_estimated_trMiB: {prod_estimated_trMiB}")
                                            if estT_cons < 0:
                                                print(f"estT_prod: {estT_prod}, cons_estimated_trMiB: {cons_estimated_trMiB}")

                                        SPM = estT_prod / estT_cons if estT_cons > 0 else float('inf')
                                        # Use both parallelisms in the key
                                        edge_key = f'{cons_storage}_{n_prod.replace("p", "")}_{n_cons}'
                                        edge_attributes[f'estT_prod_{prod_storage}_{n_prod}'] = estT_prod
                                        edge_attributes[f'estT_cons_{cons_storage}_{n_cons}'] = estT_cons
                                        edge_attributes[f'SPM_{edge_key}'] = SPM
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
                        'prod_task_name': prod_task_name,
                        'cons_task_name': consumer_row['taskName'],
                        'file_name': prod_fileName,
                    })
                    if WFG.has_edge(prod_node_name, cons_node_name):
                        WFG.edges[prod_node_name, cons_node_name].update(edge_attributes)
                    else:
                        WFG.add_edge(prod_node_name, cons_node_name, **edge_attributes)
            continue  # Skip the regular prevTask matching for stage_in tasks

        # Find consumers that have this producer as their prevTask
        if prod_task_name in consumer_groups.groups:
            consumer_group = consumer_groups.get_group(prod_task_name)
            if debug:
                print(f"Processing pair: {prod_task_name} -> consumers with prevTask={prod_task_name}")

            for _, producer_row in producer_group.iterrows():
                prod_node_name = producer_row['node_name']
                prod_fileName = producer_row['fileName']
                prod_node_data = producer_row['node_data']
                prod_op = producer_row['operation']

                # If producer is cp/scp, treat fileName as a list
                if prod_op in [2, 3, 'cp', 'scp']:
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
                    if cons_op in [2, 3, 'cp', 'scp']:
                        cons_file_list = re.split(r",|'", cons_fileName)
                        cons_file_list = [f.strip() for f in cons_file_list if f.strip()]
                    else:
                        cons_file_list = [cons_fileName]

                    # Matching logic:
                    match = False
                    if prod_op in [2, 3, 'cp', 'scp']:
                        for f in cons_file_list:
                            if f in prod_file_list:
                                match = True
                                break
                    elif cons_op in [2, 3, 'cp', 'scp']:
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
                    
                    for key in prod_node_data.keys():
                        if key.startswith('estimated_trMiB_'):
                            # Extract storage type from key like 'estimated_trMiB_beegfs-ssd_37p'
                            parts = key.split('_')
                            if len(parts) >= 3:
                                storage_type = '_'.join(parts[2:-1])  # Handle compound storage types
                                all_prod_storages.add(storage_type)
                    
                    for key in cons_node_data.keys():
                        if key.startswith('estimated_trMiB_'):
                            # Extract storage type from key like 'estimated_trMiB_beegfs-ssd_37p'
                            parts = key.split('_')
                            if len(parts) >= 3:
                                storage_type = '_'.join(parts[2:-1])  # Handle compound storage types
                                all_cons_storages.add(storage_type)
                    
                    # Try all combinations of producer and consumer storage types
                    for prod_storage in all_prod_storages:
                        for cons_storage in all_cons_storages:
                            # Find all producer and consumer keys for these storage types
                            prod_keys = [
                                key for key in prod_node_data.keys()
                                if key.startswith(f'estimated_trMiB_{prod_storage}_')
                            ]
                            cons_keys = [
                                key for key in cons_node_data.keys()
                                if key.startswith(f'estimated_trMiB_{cons_storage}_')
                            ]
                            
                            # For cp/scp, allow cross-storage matching
                            allow_cross = False
                            if prod_op in [2, 3, 'cp', 'scp'] or cons_op in [2, 3, 'cp', 'scp']:
                                if is_valid_storage_match(prod_storage, cons_storage, prod_task_name, consumer_row['taskName']):
                                    allow_cross = True
                            # For non-cp/scp, require exact match
                            if (prod_op in [2, 3, 'cp', 'scp'] or cons_op in [2, 3, 'cp', 'scp']):
                                if not allow_cross:
                                    continue
                            else:
                                # Strict match
                                if prod_storage != cons_storage:
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
                                    prod_aggregateFilesizeMB = producer_row['aggregateFilesizeMB']
                                    cons_aggregateFilesizeMB = consumer_row['aggregateFilesizeMB']
                                    prod_opCount = producer_row['opCount']
                                    cons_opCount = consumer_row['opCount']
                                    
                                    # For cp/scp, allow cross-storage matching
                                    allow_cross = False
                                    if prod_op in [2, 3, 'cp', 'scp'] or cons_op in [2, 3, 'cp', 'scp']:
                                        if is_valid_storage_match(prod_storage, cons_storage, prod_task_name, consumer_row['taskName']):
                                            allow_cross = True
                                    # For non-cp/scp, require exact match
                                    if (prod_op in [2, 3, 'cp', 'scp'] or cons_op in [2, 3, 'cp', 'scp']):
                                        if not allow_cross:
                                            continue
                                    else:
                                        # Strict match
                                        if prod_storage != cons_storage:
                                            continue
                                    
                                    # # estimated time calculation with op Count skewer
                                    # if prod_ts_slope > 0:
                                    #     estT_prod = prod_opCount * prod_aggregateFilesizeMB / prod_estimated_trMiB
                                    # else:
                                    #     estT_prod = (1/prod_opCount) * prod_aggregateFilesizeMB / prod_estimated_trMiB
                                    # if cons_ts_slope > 0:
                                    #     estT_cons = cons_opCount * cons_aggregateFilesizeMB / cons_estimated_trMiB 
                                    # else:
                                    #     estT_cons = (1/cons_opCount) * cons_aggregateFilesizeMB / cons_estimated_trMiB
                                    

                                    # estimated time calculation without op Count skewer
                                    estT_prod = prod_aggregateFilesizeMB / prod_estimated_trMiB
                                    estT_cons = cons_aggregateFilesizeMB / cons_estimated_trMiB
                                    SPM = estT_prod / estT_cons if estT_cons > 0 else float('inf')

                                    if debug:
                                        if estT_prod < 0:
                                            print(f"estT_prod: {estT_prod}, prod_estimated_trMiB: {prod_estimated_trMiB}")
                                        if estT_cons < 0:
                                            print(f"estT_prod: {estT_prod}, cons_estimated_trMiB: {cons_estimated_trMiB}")

                                    # Use both parallelisms in the key
                                    edge_key = f'{cons_storage}_{n_prod.replace("p", "")}_{n_cons}'
                                    edge_attributes[f'estT_prod_{prod_storage}_{n_prod}'] = estT_prod
                                    edge_attributes[f'estT_cons_{cons_storage}_{n_cons}'] = estT_cons
                                    edge_attributes[f'SPM_{edge_key}'] = SPM
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
        print(f"Processed pairs: {processed_pairs_debug}")
    
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
    Averages list values for estT_prod, estT_cons, SPM, and dsize, and calculates rank for each storage_n.
    
    rank = ave_dsize * (estT_prod + estT_cons) * abs(SPM - 1)
    
    Parameters:
    SPM_estT_values (dict): Dictionary containing 'estT_prod', 'estT_cons', 'SPM', and 'dsize' values.
    debug (bool): Boolean to control debug output (default: False)

    Returns:
    dict: Updated dictionary with averaged values and calculated ranks.
    """
    if debug:
        print(f"Calculating averages and ranks for {len(SPM_estT_values)} producer-consumer pairs...")
        
    for pair, data in SPM_estT_values.items():
        # Initialize a dictionary for ranks if not present
        if 'rank' not in data:
            data['rank'] = {}
        
        for storage_n in data['SPM'].keys():
            # Normalize storage_n to match keys in estT_prod and estT_cons
            prod_key = f"{'_'.join(storage_n.split('_')[:2])}p"  # Example: ssd_2_1p -> ssd_2p
            cons_key = f"{storage_n.split('_')[0]}_{storage_n.split('_')[2][:-1]}p"  # Example: ssd_2_1p -> ssd_1p
            prod_par = int(prod_key.split('_')[1].replace('p',''))
            cons_par = int(cons_key.split('_')[1].replace('p',''))
            
            # Average estT_prod
            estT_prod_values = data['estT_prod'].get(prod_key, [])
            avg_estT_prod = sum(estT_prod_values) / len(estT_prod_values) if estT_prod_values else 0.0
            # Average estT_cons
            estT_cons_values = data['estT_cons'].get(cons_key, [])
            avg_estT_cons = sum(estT_cons_values) / len(estT_cons_values) if estT_cons_values else 0.0
            # Average SPM
            spm_values = data['SPM'].get(storage_n, [])
            avg_spm = sum(spm_values) / len(spm_values) if spm_values else 0.0
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
                # Normalize storage_n to match keys in estT_prod and estT_cons
                prod_key = f"{'_'.join(storage_n.split('_')[:2])}p"  # Example: ssd_2_1p -> ssd_2p
                cons_key = f"{storage_n.split('_')[0]}_{storage_n.split('_')[2][:-1]}p"  # Example: ssd_2_1p -> ssd_1p
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
        print(f"Completed sum and ranking calculations.")
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
        
        # Group rank values by storage type
        storage_groups = {}
        for storage_n, rank_values in rank_data.items():
            storage_type = storage_n.split("_")[0]  # Extract storage type (e.g., 'beegfs' or 'ssd')
            if storage_type not in storage_groups:
                storage_groups[storage_type] = []
            storage_groups[storage_type].append((storage_n, rank_values[0]))  # Add rank values

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
    including initial stage 0 nodes where the producer is 'initial_data'.
    
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

def calculate_spm_for_workflow(wf_df: pd.DataFrame, debug: bool = False) -> dict:
    """
    Calculate SPM values for the entire workflow, matching the robust logic of the original notebook.
    Now also accounts for cp and scp operation rows as producers and consumers in the workflow graph.
    
    Parameters:
    - wf_df: DataFrame containing workflow task data
    - debug: Boolean to control debug output (default: False)
    
    Returns:
    - dict: SPM values for all producer-consumer pairs
    """
    WFG, stage_task_node_dict, stage_order_list = add_workflow_graph_nodes(wf_df, verbose=debug)

    # Debug: Print stage order list and task node dictionary
    if debug:
        print(f"Stage order list: {stage_order_list}")
        print(f"Number of nodes: {len(WFG.nodes)}")
        print("First five nodes:")
        for i, (node_name, node_data) in enumerate(list(WFG.nodes(data=True))[:5]):
            print(f"({node_name}, {node_data})")

    # Collect all unique stage orders, including fractional ones (e.g., -1, 0, 0.5, 1, 1.5, ...)
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
        print(f"\nFound {len(stage_out_tasks)} stage_out tasks:")
        for stageOrder, taskName, nodeNames in stage_out_tasks:
            print(f"  Stage {stageOrder}: {taskName} -> {len(nodeNames)} nodes")

    # First pass: collect regular stage-to-stage connections
    for currOrder in all_stage_orders:
        cons_nodes = stage_task_node_dict.get(currOrder, {})
        if currOrder == 0:
            pass
            # # TODO: Handle stage 0
            # if INITIAL_STAGE:
            #     handle_initial_stage(WFG, cons_nodes)
        else:
            # Add edges between producer and consumer nodes
            prevOrder = None
            # For fractional stageOrders (e.g., 0.5, 1.5), connect to the previous integer stage
            if isinstance(currOrder, float) and currOrder % 1 != 0:
                prevOrder = float(currOrder)
            else:
                prevOrder = currOrder - 1.0
            prod_nodes = stage_task_node_dict.get(prevOrder, {})
            if prod_nodes and cons_nodes:
                all_producer_consumer_pairs.append((prod_nodes, cons_nodes))

    # Second pass: collect cp/scp connections and stage_out connections
    if debug:
        print("\n=== Processing producer-consumer edges ===")
    for stageOrder, task_dict in stage_task_node_dict.items():
        for taskName, nodeNames in task_dict.items():
            for nodeName in nodeNames:
                node_data = WFG.nodes[nodeName]
                op = node_data.get('operation')
                
                # Handle stage_in connections (cp/scp nodes as producers)
                if op in ['cp', 'scp', 2, 3] and str(taskName).startswith('stage_in-'):
                    # Extract the actual task name from stage_in-{taskName}
                    actual_task_name = taskName.replace('stage_in-', '')
                    # Find the next integer stage where the actual task should be
                    next_stage = float(stageOrder) + 0.5 if float(stageOrder) % 1 == 0 else float(stageOrder) + 1.0
                    next_tasks = stage_task_node_dict.get(next_stage, {})
                    if actual_task_name in next_tasks:
                        if debug:
                            print(f"Found stage_in connection: {taskName} -> {actual_task_name}")
                        all_producer_consumer_pairs.append(({taskName: [nodeName]}, {actual_task_name: next_tasks[actual_task_name]}))
                    else:
                        if debug:
                            print(f"DEBUG: Looking for {actual_task_name} at stage {next_stage}, available tasks: {list(next_tasks.keys())}")
                
                # Handle stage_out connections (cp/scp nodes as consumers)
                elif op in ['cp', 'scp', 2, 3] and str(taskName).startswith('stage_out-'):
                    # Find the previous stage (stageOrder-0.5 or int-1)
                    prev_stage = float(stageOrder) - 0.5 if float(stageOrder) % 1 == 0 else float(stageOrder) - 1.0
                    prev_tasks = stage_task_node_dict.get(prev_stage, {})
                    if prev_tasks:
                        all_producer_consumer_pairs.append((prev_tasks, {taskName: [nodeName]}))
                
                # Handle regular task nodes as producers for stage_out nodes
                elif op in [0, 1]:  # Regular task nodes (write/read operations)
                    # Check if there are stage_out nodes for this task at stageOrder + 0.5
                    next_stage = float(stageOrder) + 0.5
                    next_tasks = stage_task_node_dict.get(next_stage, {})
                    if next_tasks:
                        # Find stage_out nodes for this task
                        stage_out_task_name = f'stage_out-{taskName}'
                        if stage_out_task_name in next_tasks:
                            if debug:
                                print(f"Found regular task -> stage_out connection: {taskName} -> {stage_out_task_name}")
                            all_producer_consumer_pairs.append(({taskName: [nodeName]}, {stage_out_task_name: next_tasks[stage_out_task_name]}))
                        else:
                            if debug:
                                print(f"DEBUG: Looking for stage_out-{taskName} at stage {next_stage}, available tasks: {list(next_tasks.keys())}")
                
                # Handle initial/final data movement
                elif str(taskName).startswith('stage_in-0') or str(taskName).startswith('stage_out-final'):
                    # These are already handled by the main loop or are terminal nodes
                    pass

    # Debug: Print all collected pairs
    if debug:
        print(f"\nCollected {len(all_producer_consumer_pairs)} producer-consumer pairs:")
        for i, (prod_nodes, cons_nodes) in enumerate(all_producer_consumer_pairs):
            prod_tasks = list(prod_nodes.keys())
            cons_tasks = list(cons_nodes.keys())
            print(f"  Pair {i+1}: {prod_tasks} -> {cons_tasks}")

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
        add_producer_consumer_edge(WFG, combined_prod_nodes, combined_cons_nodes, debug=debug)

    all_SPM_estT_values = extract_SPM_estT_values(WFG)
    if NORMALIZE:
        all_SPM_estT_values = normalize_estT_values_g(all_SPM_estT_values)
    all_SPM_estT_values = calculate_averages_and_rank(all_SPM_estT_values, debug=debug)
    return all_SPM_estT_values 