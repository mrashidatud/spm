"""
Results exporter module for workflow analysis.
Contains functions to export producer-consumer storage selection and parallelism results to CSV format.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import os
from .workflow_config import DEFAULT_WF


def extract_producer_consumer_results(spm_results: Dict[str, Dict[str, Any]], 
                                    wf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract producer-consumer storage selection and parallelism results from SPM results.
    
    Parameters:
    - spm_results: Dictionary containing SPM calculation results
    - wf_df: Workflow DataFrame with task information
    
    Returns:
    - DataFrame: Producer-consumer results with columns:
      producer, producerStageOrder, consumer, consumerStageOrder, producerStorageType, producerTasksPerNode, 
      consumerStorageType, consumerTasksPerNode, SPM
    """
    
    results_data = []
    
    for pair, data in spm_results.items():
        # Parse producer-consumer pair (format: "producer:consumer")
        if ':' in pair:
            producer, consumer = pair.split(':', 1)
        else:
            # Handle special cases like "input:taskName"
            if pair.startswith('input:'):
                producer = 'input'
                consumer = pair.split(':', 1)[1]
            else:
                continue
        
        # Extract stageOrder for producer and consumer
        producer_stage_order = None
        consumer_stage_order = None
        
        # Get producer stageOrder
        if producer != 'input':
            # Look for producer in workflow DataFrame
            producer_data = wf_df[wf_df['taskName'] == producer]
            if not producer_data.empty:
                producer_stage_order = producer_data['stageOrder'].iloc[0]
        
        # Get consumer stageOrder
        consumer_data = wf_df[wf_df['taskName'] == consumer]
        if not consumer_data.empty:
            consumer_stage_order = consumer_data['stageOrder'].iloc[0]
        
        # Extract SPM data
        if 'SPM' in data:
            spm_data = data['SPM']
            
            for storage_key, spm_values in spm_data.items():
                # Parse storage key format: {prod_storage}_{num1}_{consumer_storage}_{num2}p
                # Example: "beegfs_2_ssd_30p" -> prod_storage="beegfs", num1="2", cons_storage="ssd", num2="30"
                
                # Split by underscore
                parts = storage_key.split('_')
                
                if len(parts) >= 4:
                    # Handle compound storage types (e.g., "beegfs-ssd")
                    # Find where the consumer storage starts
                    # Look for the pattern: {prod_storage}_{num1}_{cons_storage}_{num2}p
                    
                    # Start from the end and work backwards
                    # The last part should end with 'p' and contain the consumer parallelism
                    last_part = parts[-1]
                    if last_part.endswith('p'):
                        consumer_tasks_per_node = last_part[:-1]  # Remove 'p'
                        
                        # The second-to-last part is the consumer storage type
                        consumer_storage_type = parts[-2]
                        
                        # The third-to-last part is the producer tasks per node
                        producer_tasks_per_node = parts[-3]
                        
                        # Everything before that is the producer storage type
                        producer_storage_type = '_'.join(parts[:-3])
                        
                        # Get the SPM value (should be a list with one value after averaging)
                        if spm_values and len(spm_values) > 0:
                            spm_value = spm_values[0]  # Take the first (and only) value
                            
                            results_data.append({
                                'producer': producer,
                                'producerStageOrder': producer_stage_order,
                                'consumer': consumer,
                                'consumerStageOrder': consumer_stage_order,
                                'producerStorageType': producer_storage_type,
                                'producerTasksPerNode': producer_tasks_per_node,
                                'consumerStorageType': consumer_storage_type,
                                'consumerTasksPerNode': consumer_tasks_per_node,
                                'SPM': spm_value
                            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Sort by producer stageOrder, then consumer stageOrder, then SPM value
    if not results_df.empty:
        results_df = results_df.sort_values(['producerStageOrder', 'consumerStageOrder', 'SPM'])
    
    return results_df


def save_producer_consumer_results(spm_results: Dict[str, Dict[str, Any]], 
                                 wf_df: pd.DataFrame, 
                                 workflow_name: str = None,
                                 output_dir: str = "../analysis_data",
                                 filename: str = None) -> str:
    """
    Save producer-consumer storage selection and parallelism results to CSV file.
    
    Parameters:
    - spm_results: Dictionary containing SPM calculation results
    - wf_df: Workflow DataFrame with task information
    - workflow_name: Name of the workflow (default: DEFAULT_WF)
    - output_dir: Directory to save the CSV file (default: "../analysis_data")
    - filename: Custom filename (optional, will auto-generate if not provided)
    
    Returns:
    - str: Path to the saved CSV file
    """
    
    if workflow_name is None:
        workflow_name = DEFAULT_WF
    
    # Extract results
    results_df = extract_producer_consumer_results(spm_results, wf_df)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        filename = f"{workflow_name}_producer_consumer_results.csv"
    
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Full path to output file
    output_path = os.path.join(output_dir, filename)
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    
    print(f"✓ Producer-consumer results saved to: {output_path}")
    print(f"  - Total producer-consumer configurations: {len(results_df)}")
    print(f"  - Columns: {list(results_df.columns)}")
    
    if not results_df.empty:
        print(f"  - Unique producer-consumer pairs: {len(results_df[['producer', 'consumer']].drop_duplicates())}")
        print(f"  - Producer storage types: {sorted(results_df['producerStorageType'].unique())}")
        print(f"  - Consumer storage types: {sorted(results_df['consumerStorageType'].unique())}")
        print(f"  - SPM range: {results_df['SPM'].min():.4f} to {results_df['SPM'].max():.4f}")
    
    return output_path


def create_detailed_producer_consumer_report(spm_results: Dict[str, Dict[str, Any]], 
                                           wf_df: pd.DataFrame,
                                           workflow_name: str = None,
                                           output_dir: str = "../analysis_data") -> str:
    """
    Create a detailed producer-consumer report with additional information.
    
    Parameters:
    - spm_results: Dictionary containing SPM calculation results
    - wf_df: Workflow DataFrame with task information
    - workflow_name: Name of the workflow (default: DEFAULT_WF)
    - output_dir: Directory to save the report (default: "../analysis_data")
    
    Returns:
    - str: Path to the saved report file
    """
    
    if workflow_name is None:
        workflow_name = DEFAULT_WF
    
    # Extract basic results
    results_df = extract_producer_consumer_results(spm_results, wf_df)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report filename
    report_filename = f"{workflow_name}_producer_consumer_detailed_report.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write(f"Producer-Consumer Analysis Report\n")
        f.write(f"Workflow: {workflow_name}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Summary:\n")
        f.write(f"- Total producer-consumer pairs: {len(results_df)}\n")
        if not results_df.empty:
            f.write(f"- Producer stages: {sorted(results_df['producerStageOrder'].unique())}\n")
            f.write(f"- Consumer stages: {sorted(results_df['consumerStageOrder'].unique())}\n")
            f.write(f"- Storage types used: {sorted(results_df['producerStorageType'].unique())}\n")
            f.write(f"- Average SPM value: {results_df['SPM'].mean():.4f}\n")
            f.write(f"- Best SPM value: {results_df['SPM'].min():.4f}\n")
            f.write(f"- Worst SPM value: {results_df['SPM'].max():.4f}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 60 + "\n")
        
        if not results_df.empty:
            for _, row in results_df.iterrows():
                f.write(f"Producer: {row['producer']} (Stage {row['producerStageOrder']})\n")
                f.write(f"Consumer: {row['consumer']} (Stage {row['consumerStageOrder']})\n")
                f.write(f"Producer Parallelism: {row['producerTasksPerNode']}\n")
                f.write(f"Consumer Parallelism: {row['consumerTasksPerNode']}\n")
                f.write(f"Producer Storage: {row['producerStorageType']}\n")
                f.write(f"Consumer Storage: {row['consumerStorageType']}\n")
                f.write(f"SPM Value: {row['SPM']:.4f}\n")
                f.write("-" * 30 + "\n")
        else:
            f.write("No producer-consumer pairs found.\n")
    
    print(f"✓ Detailed report saved to: {report_path}")
    return report_path


def analyze_storage_distribution(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the distribution of storage types across producer-consumer pairs.
    
    Parameters:
    - results_df: DataFrame with producer-consumer results
    
    Returns:
    - Dict: Analysis results
    """
    
    if results_df.empty:
        return {}
    
    analysis = {}
    
    # Storage type distribution (combine producer and consumer storage types)
    all_storage_types = pd.concat([results_df['producerStorageType'], results_df['consumerStorageType']])
    storage_counts = all_storage_types.value_counts()
    analysis['storage_distribution'] = storage_counts.to_dict()
    
    # SPM statistics by producer storage type
    spm_by_prod_storage = results_df.groupby('producerStorageType')['SPM'].agg(['mean', 'min', 'max', 'std']).to_dict('index')
    analysis['spm_by_producer_storage'] = spm_by_prod_storage
    
    # SPM statistics by consumer storage type
    spm_by_cons_storage = results_df.groupby('consumerStorageType')['SPM'].agg(['mean', 'min', 'max', 'std']).to_dict('index')
    analysis['spm_by_consumer_storage'] = spm_by_cons_storage
    
    # Stage analysis
    stage_analysis = {}
    for stage in sorted(results_df['producerStageOrder'].unique()):
        stage_data = results_df[results_df['producerStageOrder'] == stage]
        stage_analysis[f'stage_{stage}'] = {
            'count': len(stage_data),
            'avg_spm': stage_data['SPM'].mean(),
            'producer_storage_types': stage_data['producerStorageType'].unique().tolist(),
            'consumer_storage_types': stage_data['consumerStorageType'].unique().tolist()
        }
    analysis['stage_analysis'] = stage_analysis
    
    return analysis


def print_storage_analysis(results_df: pd.DataFrame) -> None:
    """
    Print a summary of storage distribution analysis.
    
    Parameters:
    - results_df: DataFrame with producer-consumer results
    """
    
    if results_df.empty:
        print("No producer-consumer results to analyze.")
        return
    
    print("\n=== Producer-Consumer Storage Analysis ===")
    
    # Basic statistics
    print(f"\nSummary:")
    print(f"  Total configurations: {len(results_df)}")
    print(f"  Unique producer-consumer pairs: {len(results_df[['producer', 'consumer']].drop_duplicates())}")
    print(f"  SPM range: {results_df['SPM'].min():.4f} to {results_df['SPM'].max():.4f}")
    print(f"  Average SPM: {results_df['SPM'].mean():.4f}")
    
    # Producer storage type distribution
    print(f"\nProducer Storage Type Distribution:")
    prod_storage_counts = results_df['producerStorageType'].value_counts()
    for storage, count in prod_storage_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {storage}: {count} configurations ({percentage:.1f}%)")
    
    # Consumer storage type distribution
    print(f"\nConsumer Storage Type Distribution:")
    cons_storage_counts = results_df['consumerStorageType'].value_counts()
    for storage, count in cons_storage_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {storage}: {count} configurations ({percentage:.1f}%)")
    
    # SPM performance by producer storage type
    print(f"\nSPM Performance by Producer Storage Type:")
    prod_spm_stats = results_df.groupby('producerStorageType')['SPM'].agg(['mean', 'min', 'max', 'std'])
    for storage, stats in prod_spm_stats.iterrows():
        print(f"  {storage}:")
        print(f"    Mean SPM: {stats['mean']:.4f}")
        print(f"    Min SPM: {stats['min']:.4f}")
        print(f"    Max SPM: {stats['max']:.4f}")
        print(f"    Std Dev: {stats['std']:.4f}")
    
    # SPM performance by consumer storage type
    print(f"\nSPM Performance by Consumer Storage Type:")
    cons_spm_stats = results_df.groupby('consumerStorageType')['SPM'].agg(['mean', 'min', 'max', 'std'])
    for storage, stats in cons_spm_stats.iterrows():
        print(f"  {storage}:")
        print(f"    Mean SPM: {stats['mean']:.4f}")
        print(f"    Min SPM: {stats['min']:.4f}")
        print(f"    Max SPM: {stats['max']:.4f}")
        print(f"    Std Dev: {stats['std']:.4f}")
    
    # Best configurations for each producer-consumer pair
    print(f"\nBest Configuration for Each Producer-Consumer Pair:")
    best_configs = results_df.loc[results_df.groupby(['producer', 'consumer'])['SPM'].idxmin()]
    for _, row in best_configs.iterrows():
        print(f"  {row['producer']} -> {row['consumer']}:")
        print(f"    Producer: {row['producerStorageType']} ({row['producerTasksPerNode']} tasks/node)")
        print(f"    Consumer: {row['consumerStorageType']} ({row['consumerTasksPerNode']} tasks/node)")
        print(f"    SPM: {row['SPM']:.4f}") 