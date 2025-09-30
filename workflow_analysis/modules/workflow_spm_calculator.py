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



def add_producer_consumer_edge(WFG, wf_df, debug=False, workflow_name=None):
    """
    Add edges between producer and consumer nodes with simplified step-by-step logic.
    
    Parameters:
    - WFG: NetworkX DiGraph object
    - wf_df: DataFrame containing workflow task data
    - debug: Boolean to control debug output
    - workflow_name: Name of the workflow for logging
    """
    import re
    import time
    
    if debug:
        print(f"\n=== Processing producer-consumer edges ===")
        start_time = time.time()

    # Step 1: Create producer and consumer DataFrame (operation in ['read', 'write', 'cp', 'scp', 'none'])
    # Since we are passing the wf_df, we can simply make a copy with filter condition
    producer_df = wf_df[wf_df['operation'].isin(['read', 'write', 'cp', 'scp', 'none'])].copy()
    consumer_df = wf_df[wf_df['operation'].isin(['read', 'write', 'cp', 'scp', 'none'])].copy()
    
    # Clean up producer_df: remove rows with taskName containing "stage_out"
    producer_df = producer_df[~producer_df['taskName'].str.contains('stage_out', na=False)]
    
    # Clean up consumer_df: remove rows with taskName containing "stage_in"
    consumer_df = consumer_df[~consumer_df['taskName'].str.contains('stage_in', na=False)]

    if producer_df.empty and debug:
        print("ERROR: No producers found!")
        return
    if consumer_df.empty and debug:
        print("ERROR: No consumers found!")
        return
    

    # Step 3: Iterate through producer_df to find matching nodes from consumer_df
    edge_count = 0
    processed_pairs = 0
    
    for _, producer_row in producer_df.iterrows():
        prod_task_name = producer_row['taskName']
        prod_fileName = producer_row['fileName']
        prod_op = producer_row['operation']
        
        # Step 3.1: First filtering to obtain a subset of consumer_df
        if prod_task_name.startswith('stage_in-'):
            # Case (1): producer has stage_in-{taskName} format
            actual_task_name = prod_task_name.replace('stage_in-', '')
            subset_consumer_df = consumer_df[consumer_df['taskName'] == actual_task_name]
        else:
            # Case (2): other rows, filter by prevTask
            subset_consumer_df = consumer_df[consumer_df['prevTask'] == prod_task_name]
        
        if subset_consumer_df.empty:
            if debug:
                print(f"[DEBUG] No matching subset_consumer_df found for {producer_row['taskName']}")
            continue
            
        # Step 3.2: Find files that can match and add edges
        if prod_task_name.startswith('stage_in-'):
            # Case (1): stage_in tasks with comma-delimited file names
            prod_file_list = prod_fileName.split(',') if ',' in prod_fileName else [prod_fileName]
            prod_file_list = [f.strip() for f in prod_file_list if f.strip()]
            
            for _, consumer_row in subset_consumer_df.iterrows():
                cons_fileName = consumer_row['fileName']
                
                # Check if consumer fileName is in producer file list
                if cons_fileName in prod_file_list:
                    if debug:
                        print(f"[DEBUG] File match found: {cons_fileName} in producer file list")
                    # Add edge with unique edge key
                    edge_attributes_dict = create_multiple_edge_attributes(producer_row, consumer_row, prod_op, consumer_row['operation'], debug)

                    if debug:
                        print(f"[DEBUG] Created {len(edge_attributes_dict)} edge attributes for file match")

                    if debug:
                        print(f"[DEBUG] Attempting to add {len(edge_attributes_dict)} edge combinations")
                    if add_edge_to_wfg(WFG, producer_row, consumer_row, edge_attributes_dict, debug):
                        edge_count += len(edge_attributes_dict)
                        processed_pairs += 1
                        if debug:
                            print(f"[DEBUG] Successfully added {len(edge_attributes_dict)} edge combinations, total edges: {edge_count}")
                    else:
                        if debug:
                            print(f"[DEBUG] Failed to add edge combinations")
                else:
                    if debug:
                        print(f"[DEBUG] No file match: {cons_fileName} not in producer file list {prod_file_list}")

        else:
            # Case (2): regular tasks
            # Split subset consumer_df into two cases
            regular_consumers = subset_consumer_df[~subset_consumer_df['taskName'].str.contains('stage_out', na=False)]
            stage_out_consumers = subset_consumer_df[subset_consumer_df['taskName'].str.contains('stage_out', na=False)]
            
            # Case (2.1): Regular tasks - exact fileName matching
            for _, consumer_row in regular_consumers.iterrows():
                if producer_row['fileName'] == consumer_row['fileName']:
                    edge_attributes_dict = create_multiple_edge_attributes(producer_row, consumer_row, prod_op, consumer_row['operation'], debug)
                    
                    if add_edge_to_wfg(WFG, producer_row, consumer_row, edge_attributes_dict, debug):
                        edge_count += len(edge_attributes_dict)
                        processed_pairs += 1

            
            # Case (2.2): Stage_out tasks - check if producer fileName is in consumer file list
            for _, consumer_row in stage_out_consumers.iterrows():
                cons_fileName = consumer_row['fileName']
                cons_file_list = cons_fileName.split(',') if ',' in cons_fileName else [cons_fileName]
                cons_file_list = [f.strip() for f in cons_file_list if f.strip()]
                
                if producer_row['fileName'] in cons_file_list:
                    edge_attributes_dict = create_multiple_edge_attributes(producer_row, consumer_row, prod_op, consumer_row['operation'], debug)
                    
                    if add_edge_to_wfg(WFG, producer_row, consumer_row, edge_attributes_dict, debug):
                        edge_count += len(edge_attributes_dict)
                        processed_pairs += 1


    if debug:
        print(f"✓ Processed {processed_pairs} producer-consumer pairs, created {edge_count} edge attributes")
        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    # Save WFG as JSON
    if workflow_name:
        save_wfg_as_json(WFG, workflow_name, debug)
    
    return WFG


def is_valid_storage_match_for_edge(prod_storage, cons_storage, debug=False):
    """
    Check if producer and consumer storage types are valid matches for edge creation.
    
    Rules:
    1. If producer is compound storage (e.g., "beegfs-ssd"), consumer should equal the second storage type
    2. If consumer is compound storage (e.g., "ssd-beegfs"), producer should equal the first storage type  
    3. If both are compound storage types, check if they share a common storage type
    4. If neither are compound, they must be equal
    
    Parameters:
    - prod_storage: Producer storage type (e.g., "beegfs", "beegfs-ssd")
    - cons_storage: Consumer storage type (e.g., "ssd", "ssd-beegfs")
    
    Returns:
    - bool: True if storage types are valid matches, False otherwise
    """
    # Check if either storage type is compound (contains "-")
    prod_is_compound = '-' in prod_storage
    cons_is_compound = '-' in cons_storage
    
    # Case 3: Both are compound storage types - check for common storage types
    if prod_is_compound and cons_is_compound:
        prod_parts = prod_storage.split('-')
        cons_parts = cons_storage.split('-')

        prod_second = prod_parts[1]
        cons_first = cons_parts[0]
        if prod_second == cons_first:
            if debug:
                print(f"[DEBUG] Compound storage match found: {prod_storage} -> {cons_storage} via common '{prod_second}'")
            return True
    
    # Case 1: Producer is compound storage
    if prod_is_compound:
        # Extract second storage type from producer (e.g., "beegfs-ssd" -> "ssd")
        second_storage = prod_storage.split('-')[1]
        return second_storage == cons_storage
    
    # Case 2: Consumer is compound storage
    if cons_is_compound:
        # Extract first storage type from consumer (e.g., "ssd-beegfs" -> "ssd")
        first_storage = cons_storage.split('-')[0]
        return prod_storage == first_storage
    
    # Case 4: Neither are compound storage types - they must be equal
    return prod_storage == cons_storage

def create_multiple_edge_attributes(producer_row, consumer_row, prod_op, cons_op, debug=False):
    """
    Create multiple edge attributes based on valid estimated_trMiB columns.
    
    This function finds all valid estimated_trMiB columns for both producer and consumer,
    and creates edge attributes for each valid combination.
    
    Parameters:
    - producer_row: Producer row data
    - consumer_row: Consumer row data  
    - prod_op: Producer operation
    - cons_op: Consumer operation
    
    Returns:
    - dict: Dictionary where each edge_key maps to its corresponding edge attributes
    """
    import re
    
    if debug:
        print(f"\n[DEBUG] Creating edge attributes for {producer_row['taskName']} -> {consumer_row['taskName']}")
        print(f"[DEBUG] Producer operation: {prod_op}, Consumer operation: {cons_op}")
    
    # Find all valid estimated_trMiB columns for producer
    prod_est_columns = []
    for col_name, value in producer_row.items():
        if col_name.startswith('estimated_trMiB_') and value is not None: # value can be zero for virtual producer and consumer
            prod_est_columns.append((col_name, value))
    
    # Find all valid estimated_trMiB columns for consumer
    cons_est_columns = []
    for col_name, value in consumer_row.items():
        if col_name.startswith('estimated_trMiB_') and value is not None: # value can be zero for virtual producer and consumer
            cons_est_columns.append((col_name, value))
    
    if debug:
        print(f"[DEBUG] Found {len(prod_est_columns)} producer estimated_trMiB columns:")
        for col_name, value in prod_est_columns:
            print(f"  - {col_name}: {value}")
        print(f"[DEBUG] Found {len(cons_est_columns)} consumer estimated_trMiB columns:")
        for col_name, value in cons_est_columns:
            print(f"  - {col_name}: {value}")
    
    # If no valid columns found, return empty dictionary
    if not prod_est_columns or not cons_est_columns:
        if debug:
            print(f"[DEBUG] No valid estimated_trMiB columns found, returning empty dict")
        return {}
    
    edge_attributes_dict = {}
    
    # Iterate through each valid producer column
    for prod_col_name, prod_est_trmib in prod_est_columns:
        # Parse producer column: estimated_trMiB_{storage}_{parallelism}p
        prod_match = re.match(r'estimated_trMiB_(.+?)_(\d+)p', prod_col_name)
        if not prod_match:
            if debug:
                print(f"[DEBUG] Failed to parse producer column: {prod_col_name}")
            continue
        
        prod_storage = prod_match.group(1)
        n_prod = prod_match.group(2)
        
        if debug:
            print(f"[DEBUG] Parsed producer: storage='{prod_storage}', parallelism='{n_prod}' from '{prod_col_name}'")
        
        # Iterate through each valid consumer column
        for cons_col_name, cons_est_trmib in cons_est_columns:
            # Parse consumer column: estimated_trMiB_{storage}_{parallelism}p
            cons_match = re.match(r'estimated_trMiB_(.+?)_(\d+)p', cons_col_name)
            if not cons_match:
                if debug:
                    print(f"[DEBUG] Failed to parse consumer column: {cons_col_name}")
                continue
                                
            cons_storage = cons_match.group(1)
            n_cons = cons_match.group(2)
        
            if debug:
                print(f"[DEBUG] Parsed consumer: storage='{cons_storage}', parallelism='{n_cons}' from '{cons_col_name}'")

            # Check if the prod_storage and cons_storage are valid matches
            is_valid = is_valid_storage_match_for_edge(prod_storage, cons_storage, debug)
            if debug:
                print(f"[DEBUG] Storage match check: {prod_storage} -> {cons_storage} = {is_valid}")
            
            if not is_valid:
                if debug:
                    print(f"[DEBUG] Skipping invalid storage match: {prod_storage} -> {cons_storage}")
                continue
                                        
            # Calculate timing values
            prod_aggregateFilesizeMB = producer_row['aggregateFilesizeMB']
            cons_aggregateFilesizeMB = consumer_row['aggregateFilesizeMB']
            
            # Calculate estimated times
            estT_prod = prod_aggregateFilesizeMB / prod_est_trmib if prod_est_trmib > 0 else 0.0
            estT_cons = cons_aggregateFilesizeMB / cons_est_trmib if cons_est_trmib > 0 else 0.0
            
            # Calculate SPM
            SPM = estT_prod + estT_cons
            
            # Create edge key
            edge_key = f'{prod_storage}_{n_prod}_{cons_storage}_{n_cons}'
            
            if debug:
                print(f"[DEBUG] Creating edge with key: {edge_key}")
                print(f"[DEBUG] estT_prod: {estT_prod:.4f}s, estT_cons: {estT_cons:.4f}s, SPM: {SPM:.4f}s")
            
            # Create edge attributes for this combination
            edge_attributes = {
                'prod_aggregateFilesizeMB': prod_aggregateFilesizeMB,
                'cons_aggregateFilesizeMB': cons_aggregateFilesizeMB,
                # 'prod_max_parallelism': producer_row['parallelism'],
                # 'cons_max_parallelism': consumer_row['parallelism'],
                'n_prod': n_prod,
                'n_cons': n_cons,
                'prod_storage': prod_storage,
                'cons_storage': cons_storage,
                'estT_prod': estT_prod,
                'estT_cons': estT_cons,
                'SPM': SPM,
                # 'prod_stage_order': producer_row['stageOrder'],
                # 'cons_stage_order': consumer_row['stageOrder'],
                'prod_task_name': producer_row['taskName'],
                'cons_task_name': consumer_row['taskName'],
                # 'file_name': producer_row['fileName'],
                'prod_op': prod_op,
                'cons_op': cons_op,
                # 'edge_key': edge_key,
            }
            
            # Add to dictionary with edge_key as the key
            edge_attributes_dict[edge_key] = edge_attributes
    
    if debug:
        print(f"[DEBUG] Created {len(edge_attributes_dict)} edge attributes")
        for edge_key in edge_attributes_dict.keys():
            print(f"  - {edge_key}")
    
    return edge_attributes_dict

def add_edge_to_wfg(WFG, producer_row, consumer_row, edge_attributes_dict, debug=False):
    """
    Add edge to WFG with sanity check for node existence.
    This function stores all edge data in the exact format requested:
    {
        'edge_key_1': {complete_attributes_dict_1},
        'edge_key_2': {complete_attributes_dict_2},
        ...
    }
    
    Parameters:
    - WFG: NetworkX DiGraph object
    - producer_row: Producer row data
    - consumer_row: Consumer row data
    - edge_attributes_dict: Dictionary where each edge_key maps to edge attributes
    - debug: Debug flag
    
    Returns:
    - bool: True if edge was added successfully, False otherwise
    """
    # Sanity check: verify nodes exist in WFG
    prod_node_name = f"{producer_row['taskName']}:{producer_row.get('taskPID', 'unknown')}:{producer_row['fileName']}"
    cons_node_name = f"{consumer_row['taskName']}:{consumer_row.get('taskPID', 'unknown')}:{consumer_row['fileName']}"
    
    if debug:
        print(f"[DEBUG] add_edge_to_wfg: Adding {len(edge_attributes_dict)} edge combinations")
        print(f"[DEBUG] Producer node: {prod_node_name}")
        print(f"[DEBUG] Consumer node: {cons_node_name}")
        print(f"[DEBUG] WFG has producer node: {WFG.has_node(prod_node_name)}")
        print(f"[DEBUG] WFG has consumer node: {WFG.has_node(cons_node_name)}")
    
    if not WFG.has_node(prod_node_name):
        if debug:
            print(f"[DEBUG] Producer node not found in WFG: {prod_node_name}")
        return False
    
    if not WFG.has_node(cons_node_name):
        if debug:
            print(f"[DEBUG] Consumer node not found in WFG: {cons_node_name}")
        return False
    
    # Add edge with all edge attributes from the dictionary
    if WFG.has_edge(prod_node_name, cons_node_name):
        if debug:
            print(f"[DEBUG] Updating existing edge between {prod_node_name} -> {cons_node_name}")
        # Get existing edge data
        existing_edge_data = WFG.edges[prod_node_name, cons_node_name]
        
        # Store all edge attributes from the dictionary in the exact format requested
        for edge_key, edge_attrs in edge_attributes_dict.items():
            # Store the complete edge attributes dictionary under the edge_key
            existing_edge_data[edge_key] = edge_attrs.copy()
            
            # Also store the edge_key in a list to track all keys for this edge
            if 'all_edge_keys' not in existing_edge_data:
                existing_edge_data['all_edge_keys'] = []
            if edge_key not in existing_edge_data['all_edge_keys']:
                existing_edge_data['all_edge_keys'].append(edge_key)
        
    else:
        if debug:
            print(f"[DEBUG] Adding new edge between {prod_node_name} -> {cons_node_name}")
        
        # Create new edge with all edge attributes from the dictionary in the exact format requested
        new_edge_attributes = {}
        
        # Store all edge attributes from the dictionary
        for edge_key, edge_attrs in edge_attributes_dict.items():
            # Store the complete edge attributes dictionary under the edge_key
            new_edge_attributes[edge_key] = edge_attrs.copy()
        
        # Add the edge_key list to track all keys for this edge
        new_edge_attributes['all_edge_keys'] = list(edge_attributes_dict.keys())
        
        # Add new edge
        WFG.add_edge(prod_node_name, cons_node_name, **new_edge_attributes)
    
    if debug:
        # Show how many edge keys are now stored for this edge
        edge_data = WFG.edges[prod_node_name, cons_node_name]
        if 'all_edge_keys' in edge_data:
            print(f"[DEBUG] Edge now contains {len(edge_data['all_edge_keys'])} edge keys: {edge_data['all_edge_keys']}")
            # Show the structure of the stored data
            for edge_key in edge_data['all_edge_keys']:
                if edge_key in edge_data:
                    print(f"[DEBUG] Edge key '{edge_key}' contains {len(edge_data[edge_key])} attributes")
                    if debug and len(edge_data[edge_key]) > 0:
                        print(f"[DEBUG] Sample attributes for '{edge_key}': {list(edge_data[edge_key].keys())[:5]}")
        print(f"[DEBUG] Successfully added/updated edge with {len(edge_attributes_dict)} edge combinations")
        print(f"[DEBUG] Edge now exists: {WFG.has_edge(prod_node_name, cons_node_name)}")
    
    return True

    
def save_wfg_as_json(WFG, workflow_name, debug=False):
    """
    Save WFG as JSON file.
    """
    try:
        import os
        import json
        # Create the output directory if it doesn't exist
        output_dir = "workflow_spm_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert NetworkX graph to JSON-serializable format
        wfg_data = {
            'nodes': {},
            'edges': []
        }
        
        # Convert nodes to JSON-serializable format
        for node_name, node_data in WFG.nodes(data=True):
            # Convert numpy types to native Python types
            serializable_node_data = {}
            for key, value in node_data.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_node_data[key] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.ndarray):
                    serializable_node_data[key] = value.tolist()
                elif value is None:
                    serializable_node_data[key] = None
                elif isinstance(value, (int, float, str, bool, list, dict)):
                    serializable_node_data[key] = value
                else:
                    serializable_node_data[key] = str(value)  # Convert other types to string
            
            wfg_data['nodes'][node_name] = serializable_node_data
        
        # Convert edges to JSON-serializable format
        for edge in WFG.edges(data=True):
            producer_node, consumer_node, edge_data = edge
            
            # Convert edge attributes to JSON-serializable format
            serializable_edge_data = {}
            for key, value in edge_data.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_edge_data[key] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.ndarray):
                    serializable_edge_data[key] = value.tolist()
                elif value is None:
                    serializable_edge_data[key] = None
                elif isinstance(value, (int, float, str, bool, list, dict)):
                    serializable_edge_data[key] = value
                else:
                    serializable_edge_data[key] = str(value)  # Convert other types to string
            
            # Get the edge_key from the edge data
            edge_key = edge_data.get('edge_key', f"{producer_node}_{consumer_node}")
            
            # Create edge entry with edge_key as the key and all attributes as the value
            wfg_data['edges'].append({
                'producer_node': producer_node,
                'consumer_node': consumer_node,
                'edge_key': edge_key,
                'edge_data': serializable_edge_data
            })
        
        # Save WFG as JSON
        json_filename = f"{workflow_name}_WFG.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(wfg_data, f, indent=4, default=str)
        
        if debug:
            print(f"   Saved WFG to: {json_path}")
            print(f"   Total nodes: {len(wfg_data['nodes'])}")
            print(f"   Total edges: {len(wfg_data['edges'])}")
    except Exception as e:
        if debug:
            print(f"   Warning: Could not save WFG: {e}")
            import traceback
            traceback.print_exc()

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


def calculate_max_and_rank(SPM_estT_values, debug=False):
    """
    Takes maximum values for estT_prod and estT_cons, calculates SPM as max_estT_prod + max_estT_cons,
    and calculates rank for each storage_n.
    
    This function now handles the new edge key format: "beegfs-ssd_15_ssd_150"
    
    Parameters:
    SPM_estT_values (dict): Dictionary containing 'estT_prod', 'estT_cons', and 'dsize' values.
    debug (bool): Boolean to control debug output (default: False)

    Returns:
    dict: Updated dictionary with maximum values, calculated SPM, and calculated ranks.
    """
    if debug:
        print(f"Calculating maximums and ranks for {len(SPM_estT_values)} producer-consumer pairs...")
        
    for pair, data in SPM_estT_values.items():
        if debug:
            print(f"Processing pair: {pair}")
            print(f"Available estT_prod keys: {list(data['estT_prod'].keys())}")
            print(f"Available estT_cons keys: {list(data['estT_cons'].keys())}")
        
        # Initialize a dictionary for ranks if not present
        if 'rank' not in data:
            data['rank'] = {}
        
        # Process each edge_key directly (new format: "beegfs-ssd_15_ssd_150")
        for edge_key in data['estT_prod'].keys():
            if debug:
                print(f"Processing edge_key: {edge_key}")
            
            # Parse the edge_key format: "beegfs-ssd_15_ssd_150"
            parts = edge_key.split('_')
            if len(parts) >= 4:
                prod_storage = parts[0]  # "beegfs-ssd"
                n_prod = parts[1]        # "15"
                cons_storage = parts[2]  # "ssd"
                n_cons = parts[3]        # "150"
                
                if debug:
                    print(f"  Parsed: prod_storage={prod_storage}, n_prod={n_prod}, cons_storage={cons_storage}, n_cons={n_cons}")
            else:
                if debug:
                    print(f"  Skipping edge_key with unexpected format: {edge_key}")
                continue
            
            # Get the values for this edge_key
            estT_prod_values = data['estT_prod'].get(edge_key, [])
            estT_cons_values = data['estT_cons'].get(edge_key, [])
            dsize_prod_values = data['dsize_prod'].get(edge_key, [])
            dsize_cons_values = data['dsize_cons'].get(edge_key, [])
            par_prod_values = data['par_prod'].get(edge_key, [])
            par_cons_values = data['par_cons'].get(edge_key, [])
            
            if debug:
                print(f"  estT_prod_values: {estT_prod_values}")
                print(f"  estT_cons_values: {estT_cons_values}")
                print(f"  dsize_prod_values: {dsize_prod_values}")
                print(f"  dsize_cons_values: {dsize_cons_values}")
            
            # Calculate maximums
            max_estT_prod = max(estT_prod_values) if estT_prod_values else 0.0
            max_estT_cons = max(estT_cons_values) if estT_cons_values else 0.0
            max_dsize_prod = max(dsize_prod_values) if dsize_prod_values else 0.0
            max_dsize_cons = max(dsize_cons_values) if dsize_cons_values else 0.0
            max_par_prod = max(par_prod_values) if par_prod_values else 0.0
            max_par_cons = max(par_cons_values) if par_cons_values else 0.0
            
            # Calculate SPM
            max_spm = max_estT_prod + max_estT_cons
            
            if debug:
                print(f"  Maximums: estT_prod={max_estT_prod:.4f}, estT_cons={max_estT_cons:.4f}, SPM={max_spm:.4f}")
                print(f"  Maximums: dsize_prod={max_dsize_prod:.4f}, dsize_cons={max_dsize_cons:.4f}")
                print(f"  Maximums: par_prod={max_par_prod:.4f}, par_cons={max_par_cons:.4f}")
            
            # Calculate rank
            if MULTI_NODES == False:
                prod_seq_tasks = max_par_prod / int(n_prod) if int(n_prod) > 0 else 0
                cons_seq_tasks = max_par_cons / int(n_cons) if int(n_cons) > 0 else 0
                rank = (prod_seq_tasks * max_dsize_prod * max_estT_prod + cons_seq_tasks * max_dsize_cons * max_estT_cons)
            else:
                rank = max_estT_prod + max_estT_cons
            
            if debug:
                print(f"  Calculated rank: {rank:.4f}")
            
            # Store maximums back in the dictionary for reference
            data['estT_prod'][edge_key] = [max_estT_prod]
            data['estT_cons'][edge_key] = [max_estT_cons]
            data['SPM'][edge_key] = [max_spm]
            data['dsize_prod'][edge_key] = [max_dsize_prod]
            data['dsize_cons'][edge_key] = [max_dsize_cons]
            data['par_prod'][edge_key] = [max_par_prod]
            data['par_cons'][edge_key] = [max_par_cons]
            data['rank'][edge_key] = [rank]

    if debug:
        print(f"Completed maximum and ranking calculations.")
    return SPM_estT_values


def calculate_averages_and_rank(SPM_estT_values, debug=False):
    """
    Averages list values for estT_prod and estT_cons, calculates SPM as ave_estT_prod + ave_estT_cons,
    and calculates rank for each storage_n.
    
    This function now handles the new edge key format: "beegfs-ssd_15_ssd_150"
    
    Parameters:
    SPM_estT_values (dict): Dictionary containing 'estT_prod', 'estT_cons', and 'dsize' values.
    debug (bool): Boolean to control debug output (default: False)

    Returns:
    dict: Updated dictionary with averaged values, calculated SPM, and calculated ranks.
    """
    if debug:
        print(f"Calculating averages and ranks for {len(SPM_estT_values)} producer-consumer pairs...")
        
    for pair, data in SPM_estT_values.items():
        if debug:
            print(f"Processing pair: {pair}")
            print(f"Available estT_prod keys: {list(data['estT_prod'].keys())}")
            print(f"Available estT_cons keys: {list(data['estT_cons'].keys())}")
        
        # Initialize a dictionary for ranks if not present
        if 'rank' not in data:
            data['rank'] = {}
        
        # Process each edge_key directly (new format: "beegfs-ssd_15_ssd_150")
        for edge_key in data['estT_prod'].keys():
            if debug:
                print(f"Processing edge_key: {edge_key}")
            
            # Parse the edge_key format: "beegfs-ssd_15_ssd_150"
            parts = edge_key.split('_')
            if len(parts) >= 4:
                prod_storage = parts[0]  # "beegfs-ssd"
                n_prod = parts[1]        # "15"
                cons_storage = parts[2]  # "ssd"
                n_cons = parts[3]        # "150"
                
                if debug:
                    print(f"  Parsed: prod_storage={prod_storage}, n_prod={n_prod}, cons_storage={cons_storage}, n_cons={n_cons}")
            else:
                if debug:
                    print(f"  Skipping edge_key with unexpected format: {edge_key}")
                continue
            
            # Get the values for this edge_key
            estT_prod_values = data['estT_prod'].get(edge_key, [])
            estT_cons_values = data['estT_cons'].get(edge_key, [])
            dsize_prod_values = data['dsize_prod'].get(edge_key, [])
            dsize_cons_values = data['dsize_cons'].get(edge_key, [])
            par_prod_values = data['par_prod'].get(edge_key, [])
            par_cons_values = data['par_cons'].get(edge_key, [])
            
            if debug:
                print(f"  estT_prod_values: {estT_prod_values}")
                print(f"  estT_cons_values: {estT_cons_values}")
                print(f"  dsize_prod_values: {dsize_prod_values}")
                print(f"  dsize_cons_values: {dsize_cons_values}")
            
            # Calculate averages
            avg_estT_prod = sum(estT_prod_values) / len(estT_prod_values) if estT_prod_values else 0.0
            avg_estT_cons = sum(estT_cons_values) / len(estT_cons_values) if estT_cons_values else 0.0
            avg_dsize_prod = sum(dsize_prod_values) / len(dsize_prod_values) if dsize_prod_values else 0.0
            avg_dsize_cons = sum(dsize_cons_values) / len(dsize_cons_values) if dsize_cons_values else 0.0
            avg_par_prod = sum(par_prod_values) / len(par_prod_values) if par_prod_values else 0.0
            avg_par_cons = sum(par_cons_values) / len(par_cons_values) if par_cons_values else 0.0
            
            # Calculate SPM
            avg_spm = avg_estT_prod + avg_estT_cons
            
            if debug:
                print(f"  Averages: estT_prod={avg_estT_prod:.4f}, estT_cons={avg_estT_cons:.4f}, SPM={avg_spm:.4f}")
                print(f"  Averages: dsize_prod={avg_dsize_prod:.4f}, dsize_cons={avg_dsize_cons:.4f}")
                print(f"  Averages: par_prod={avg_par_prod:.4f}, par_cons={avg_par_cons:.4f}")
            
            # Calculate rank
            if MULTI_NODES == False:
                prod_seq_tasks = avg_par_prod / int(n_prod) if int(n_prod) > 0 else 0
                cons_seq_tasks = avg_par_cons / int(n_cons) if int(n_cons) > 0 else 0
                rank = (prod_seq_tasks * avg_dsize_prod * avg_estT_prod + cons_seq_tasks * avg_dsize_cons * avg_estT_cons)
            else:
                rank = avg_estT_prod + avg_estT_cons
            
            if debug:
                print(f"  Calculated rank: {rank:.4f}")
            
            # Store averages back in the dictionary for reference
            data['estT_prod'][edge_key] = [avg_estT_prod]
            data['estT_cons'][edge_key] = [avg_estT_cons]
            data['SPM'][edge_key] = [avg_spm]
            data['dsize_prod'][edge_key] = [avg_dsize_prod]
            data['dsize_cons'][edge_key] = [avg_dsize_cons]
            data['par_prod'][edge_key] = [avg_par_prod]
            data['par_cons'][edge_key] = [avg_par_cons]
            data['rank'][edge_key] = [rank]

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
                    prod_par = int(prod_key.split('_')[1].replace('p',''))
                    cons_par = int(cons_key.split('_')[1].replace('p',''))
                
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
    
    This function handles the new edge data format where each edge_key contains
    the complete attributes for that storage combination.
    
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
        
        # Process the new edge data format where each edge_key contains complete attributes
        for key, value in attributes.items():
            # Skip non-edge_key attributes like 'all_edge_keys'
            if key == 'all_edge_keys':
                continue
                
            # Check if this is an edge_key with complete attributes
            if isinstance(value, dict) and 'SPM' in value and 'estT_prod' in value and 'estT_cons' in value:
                # This is an edge_key with complete attributes (new format)
                edge_key = key
                
                # Extract SPM value
                if 'SPM' in value:
                    if edge_key not in SPM_estT_values[prod_cons_pair]['SPM']:
                        SPM_estT_values[prod_cons_pair]['SPM'][edge_key] = []
                    SPM_estT_values[prod_cons_pair]['SPM'][edge_key].append(value['SPM'])
                
                # Extract estT_prod value
                if 'estT_prod' in value:
                    if edge_key not in SPM_estT_values[prod_cons_pair]['estT_prod']:
                        SPM_estT_values[prod_cons_pair]['estT_prod'][edge_key] = []
                    SPM_estT_values[prod_cons_pair]['estT_prod'][edge_key].append(value['estT_prod'])
                
                # Extract estT_cons value
                if 'estT_cons' in value:
                    if edge_key not in SPM_estT_values[prod_cons_pair]['estT_cons']:
                        SPM_estT_values[prod_cons_pair]['estT_cons'][edge_key] = []
                    SPM_estT_values[prod_cons_pair]['estT_cons'][edge_key].append(value['estT_cons'])
                
                # Extract dsize values
                if 'prod_aggregateFilesizeMB' in value:
                    if edge_key not in SPM_estT_values[prod_cons_pair]['dsize_prod']:
                        SPM_estT_values[prod_cons_pair]['dsize_prod'][edge_key] = []
                    SPM_estT_values[prod_cons_pair]['dsize_prod'][edge_key].append(value['prod_aggregateFilesizeMB'])
                
                if 'cons_aggregateFilesizeMB' in value:
                    if edge_key not in SPM_estT_values[prod_cons_pair]['dsize_cons']:
                        SPM_estT_values[prod_cons_pair]['dsize_cons'][edge_key] = []
                    SPM_estT_values[prod_cons_pair]['dsize_cons'][edge_key].append(value['cons_aggregateFilesizeMB'])
                
                # Extract parallelism values
                if 'prod_max_parallelism' in value:
                    if edge_key not in SPM_estT_values[prod_cons_pair]['par_prod']:
                        SPM_estT_values[prod_cons_pair]['par_prod'][edge_key] = []
                    SPM_estT_values[prod_cons_pair]['par_prod'][edge_key].append(value['prod_max_parallelism'])
                
                if 'cons_max_parallelism' in value:
                    if edge_key not in SPM_estT_values[prod_cons_pair]['par_cons']:
                        SPM_estT_values[prod_cons_pair]['par_cons'][edge_key] = []
                    SPM_estT_values[prod_cons_pair]['par_cons'][edge_key].append(value['cons_max_parallelism'])

    return SPM_estT_values

def calculate_spm_from_wfg(WFG, debug=False):
    """
    Calculate SPM values from an existing workflow graph (WFG).
    This function handles the SPM calculation logic after the graph has been built.
    
    Parameters:
    - WFG: NetworkX DiGraph object containing the workflow graph
    - debug: Boolean to control debug output for SPM calculation (default: False)
    
    Returns:
    - dict: SPM values for all producer-consumer pairs
    """
    if debug:
        print(f"\n=== Calculating SPM from WFG ===")
        print(f"WFG has {WFG.number_of_nodes()} nodes and {WFG.number_of_edges()} edges")
    
    # Extract SPM values and calculate averages/ranks
    all_SPM_estT_values = extract_SPM_estT_values(WFG)
    
    if debug:
        print(f"Extracted SPM values for {len(all_SPM_estT_values)} producer-consumer pairs")
    
    if NORMALIZE:
        if debug:
            print("Applying global normalization...")
        all_SPM_estT_values = normalize_estT_values_g(all_SPM_estT_values)
    
    if debug:
        print("Calculating maximums and ranks...")
    all_SPM_estT_values = calculate_max_and_rank(all_SPM_estT_values, debug=debug)
    
    if debug:
        print(f"✓ Completed SPM calculation for {len(all_SPM_estT_values)} pairs")
    
    return all_SPM_estT_values

def calculate_spm_for_edges(wf_df: pd.DataFrame, debug: bool = False, workflow_name: str = None) -> nx.DiGraph:
    """
    Convert workflow dataframe into a workflow DAG.
    
    Parameters:
    - wf_df: DataFrame containing workflow task data
    - debug: Boolean to control debug output for graph building (default: False)
    - workflow_name: Name of the workflow for logging intermediate results (default: None)
    
    Returns:
    - nx.DiGraph: The built workflow graph (WFG)
    """
    # Step 1: Convert all values in the operation column to string
    wf_df['operation'] = wf_df['operation'].astype(str)
    
    # Step 2: Add all task nodes into a DAG named WFG
    WFG, stage_task_node_dict, stage_order_list = add_workflow_graph_nodes(wf_df, verbose=debug)

    if debug:
        print(f"Stage order list: {stage_order_list}")
        print(f"Number of nodes: {len(WFG.nodes)}")
        print("First five nodes:")
        for i, (node_name, node_data) in enumerate(list(WFG.nodes(data=True))[:5]):
            print(f"({node_name}, {node_data})")

    # Step 3: Add edges into the workflow DAG
    # The workflow DAG should be a multi graph with many different edges between producer and consumer task nodes
    add_producer_consumer_edge(WFG, wf_df, debug=debug, workflow_name=workflow_name)

    return WFG 