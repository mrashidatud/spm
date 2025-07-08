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
    - DataFrame: Producer-consumer results with specified columns
    """
    
    # Create a mapping from taskName to stageOrder for quick lookup
    task_stage_mapping = {}
    for _, row in wf_df.iterrows():
        task_name = row['taskName']
        stage_order = row['stageOrder']
        if task_name not in task_stage_mapping:
            task_stage_mapping[task_name] = stage_order
    
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
        
        # Get stage orders
        producer_stage = task_stage_mapping.get(producer, -1)  # -1 for input stage
        consumer_stage = task_stage_mapping.get(consumer, -1)
        
        # Extract storage and parallelism information
        if 'best_storage_type' in data and 'best_parallelism' in data:
            # Single best configuration
            storage_type = data['best_storage_type']
            parallelism_config = data['best_parallelism']
            spm_value = data.get('best_rank', np.nan)
            
            # Parse parallelism configuration (format: "storage_prodPar_consPar")
            if '_' in parallelism_config:
                parts = parallelism_config.split('_')
                if len(parts) >= 3:
                    prod_parallelism = parts[1]
                    cons_parallelism = parts[2]
                else:
                    prod_parallelism = np.nan
                    cons_parallelism = np.nan
            else:
                prod_parallelism = np.nan
                cons_parallelism = np.nan
            
            results_data.append({
                'producer': producer,
                'producerStage': producer_stage,
                'consumer': consumer,
                'consumerStage': consumer_stage,
                'prodParallelism': prod_parallelism,
                'consParallelism': cons_parallelism,
                'p-c-Storage': storage_type,
                'p-c-SPM': spm_value
            })
        
        elif 'rank' in data:
            # Multiple storage configurations - get the best one
            rank_data = data['rank']
            if rank_data:
                # Find the storage configuration with the best (lowest) rank
                best_storage = None
                best_rank = float('inf')
                
                for storage_config, rank_values in rank_data.items():
                    if rank_values and len(rank_values) > 0:
                        avg_rank = np.mean(rank_values)
                        if avg_rank < best_rank:
                            best_rank = avg_rank
                            best_storage = storage_config
                
                if best_storage:
                    # Parse storage configuration (format: "storage_prodPar_consPar")
                    if '_' in best_storage:
                        parts = best_storage.split('_')
                        if len(parts) >= 3:
                            storage_type = parts[0]
                            prod_parallelism = parts[1]
                            cons_parallelism = parts[2]
                        else:
                            storage_type = best_storage
                            prod_parallelism = np.nan
                            cons_parallelism = np.nan
                    else:
                        storage_type = best_storage
                        prod_parallelism = np.nan
                        cons_parallelism = np.nan
                    
                    results_data.append({
                        'producer': producer,
                        'producerStage': producer_stage,
                        'consumer': consumer,
                        'consumerStage': consumer_stage,
                        'prodParallelism': prod_parallelism,
                        'consParallelism': cons_parallelism,
                        'p-c-Storage': storage_type,
                        'p-c-SPM': best_rank
                    })
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Sort by producer stage, then consumer stage, then producer name
    if not results_df.empty:
        results_df = results_df.sort_values(['producerStage', 'consumerStage', 'producer', 'consumer'])
    
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
    print(f"  - Total producer-consumer pairs: {len(results_df)}")
    print(f"  - Columns: {list(results_df.columns)}")
    
    if not results_df.empty:
        print(f"  - Producer stages: {sorted(results_df['producerStage'].unique())}")
        print(f"  - Consumer stages: {sorted(results_df['consumerStage'].unique())}")
        print(f"  - Storage types: {sorted(results_df['p-c-Storage'].unique())}")
    
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
            f.write(f"- Producer stages: {sorted(results_df['producerStage'].unique())}\n")
            f.write(f"- Consumer stages: {sorted(results_df['consumerStage'].unique())}\n")
            f.write(f"- Storage types used: {sorted(results_df['p-c-Storage'].unique())}\n")
            f.write(f"- Average SPM value: {results_df['p-c-SPM'].mean():.4f}\n")
            f.write(f"- Best SPM value: {results_df['p-c-SPM'].min():.4f}\n")
            f.write(f"- Worst SPM value: {results_df['p-c-SPM'].max():.4f}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 60 + "\n")
        
        if not results_df.empty:
            for _, row in results_df.iterrows():
                f.write(f"Producer: {row['producer']} (Stage {row['producerStage']})\n")
                f.write(f"Consumer: {row['consumer']} (Stage {row['consumerStage']})\n")
                f.write(f"Producer Parallelism: {row['prodParallelism']}\n")
                f.write(f"Consumer Parallelism: {row['consParallelism']}\n")
                f.write(f"Storage: {row['p-c-Storage']}\n")
                f.write(f"SPM Value: {row['p-c-SPM']:.4f}\n")
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
    
    # Storage type distribution
    storage_counts = results_df['p-c-Storage'].value_counts()
    analysis['storage_distribution'] = storage_counts.to_dict()
    
    # SPM statistics by storage type
    spm_by_storage = results_df.groupby('p-c-Storage')['p-c-SPM'].agg(['mean', 'min', 'max', 'std']).to_dict('index')
    analysis['spm_by_storage'] = spm_by_storage
    
    # Stage analysis
    stage_analysis = {}
    for stage in sorted(results_df['producerStage'].unique()):
        stage_data = results_df[results_df['producerStage'] == stage]
        stage_analysis[f'stage_{stage}'] = {
            'count': len(stage_data),
            'avg_spm': stage_data['p-c-SPM'].mean(),
            'storage_types': stage_data['p-c-Storage'].unique().tolist()
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
    
    analysis = analyze_storage_distribution(results_df)
    
    print("\n=== Storage Distribution Analysis ===")
    
    if 'storage_distribution' in analysis:
        print("\nStorage Type Usage:")
        for storage, count in analysis['storage_distribution'].items():
            percentage = (count / len(results_df)) * 100
            print(f"  {storage}: {count} pairs ({percentage:.1f}%)")
    
    if 'spm_by_storage' in analysis:
        print("\nSPM Performance by Storage Type:")
        for storage, stats in analysis['spm_by_storage'].items():
            print(f"  {storage}:")
            print(f"    Mean SPM: {stats['mean']:.4f}")
            print(f"    Min SPM: {stats['min']:.4f}")
            print(f"    Max SPM: {stats['max']:.4f}")
            print(f"    Std Dev: {stats['std']:.4f}")
    
    if 'stage_analysis' in analysis:
        print("\nStage-wise Analysis:")
        for stage_key, stage_data in analysis['stage_analysis'].items():
            stage_num = stage_key.replace('stage_', '')
            print(f"  Stage {stage_num}:")
            print(f"    Pairs: {stage_data['count']}")
            print(f"    Avg SPM: {stage_data['avg_spm']:.4f}")
            print(f"    Storage types: {', '.join(stage_data['storage_types'])}") 