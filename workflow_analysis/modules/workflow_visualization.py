"""
Visualization module for workflow analysis.
Contains functions for plotting and visualizing workflow analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os


def plot_io_time_breakdown(task_io_times: Dict[str, float], 
                          save_path: str = "./analysis_data/io_breakdown.png") -> None:
    """
    Plot I/O time breakdown by task.
    
    Parameters:
    - task_io_times: Dictionary containing I/O times by task
    - save_path: Path to save the plot
    """
    # Extract data for plotting
    tasks = list(task_io_times.keys())
    times = list(task_io_times.values())
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tasks, times, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time:.2f}s', ha='center', va='bottom')
    
    plt.title('I/O Time Breakdown by Task', fontsize=14, fontweight='bold')
    plt.xlabel('Task Name', fontsize=12)
    plt.ylabel('I/O Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_storage_performance_comparison(spm_results: Dict[str, Dict[str, Any]], 
                                       save_path: str = "./analysis_data/storage_comparison.png") -> None:
    """
    Plot storage performance comparison across different storage types.
    
    Parameters:
    - spm_results: SPM calculation results
    - save_path: Path to save the plot
    """
    # Prepare data for plotting
    storage_types = []
    avg_ranks = []
    pair_names = []
    
    for pair, data in spm_results.items():
        if 'avg_rank_by_storage' in data:
            for storage, rank in data['avg_rank_by_storage'].items():
                storage_types.append(storage)
                avg_ranks.append(rank)
                pair_names.append(pair)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Storage_Type': storage_types,
        'Average_Rank': avg_ranks,
        'Pair': pair_names
    })
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar plot
    sns.barplot(data=plot_df, x='Pair', y='Average_Rank', hue='Storage_Type', 
                palette='viridis')
    
    plt.title('Storage Performance Comparison by Producer-Consumer Pair', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Producer-Consumer Pair', fontsize=12)
    plt.ylabel('Average Rank (Lower is Better)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Storage Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_spm_distribution(spm_results: Dict[str, Dict[str, Any]], 
                         save_path: str = "./analysis_data/spm_distribution.png") -> None:
    """
    Plot SPM distribution across different storage configurations.
    
    Parameters:
    - spm_results: SPM calculation results
    - save_path: Path to save the plot
    """
    # Prepare data for plotting
    spm_values = []
    storage_configs = []
    pair_names = []
    
    for pair, data in spm_results.items():
        if 'SPM' in data:
            for storage_n, spm_list in data['SPM'].items():
                for spm_val in spm_list:
                    spm_values.append(spm_val)
                    storage_configs.append(storage_n)
                    pair_names.append(pair)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'SPM_Value': spm_values,
        'Storage_Config': storage_configs,
        'Pair': pair_names
    })
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each pair
    unique_pairs = plot_df['Pair'].unique()
    n_pairs = len(unique_pairs)
    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    
    for i, pair in enumerate(unique_pairs):
        plt.subplot(rows, cols, i + 1)
        pair_data = plot_df[plot_df['Pair'] == pair]
        
        # Create box plot
        sns.boxplot(data=pair_data, x='Storage_Config', y='SPM_Value')
        plt.title(f'SPM Distribution: {pair}', fontsize=10)
        plt.xlabel('Storage Configuration')
        plt.ylabel('SPM Value')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_estimated_transfer_rates(wf_df: pd.DataFrame, 
                                save_path: str = "./analysis_data/estimated_transfer_rates.png") -> None:
    """
    Plot estimated transfer rates for different storage types and parallelism levels.
    
    Parameters:
    - wf_df: Workflow DataFrame with estimated transfer rates
    - save_path: Path to save the plot
    """
    # Get estimated transfer rate columns
    tr_columns = [col for col in wf_df.columns if col.startswith('estimated_trMiB_')]
    
    if not tr_columns:
        print("No estimated transfer rate columns found in the DataFrame.")
        return
    
    # Prepare data for plotting
    plot_data = []
    for col in tr_columns:
        # Extract storage type and parallelism from column name
        parts = col.split('_')
        if len(parts) >= 4:
            storage_type = parts[2]
            parallelism = parts[3].replace('p', '')
            
            # Get non-null values
            values = wf_df[col].dropna()
            for val in values:
                plot_data.append({
                    'Storage_Type': storage_type,
                    'Parallelism': parallelism,
                    'Transfer_Rate': val
                })
    
    if not plot_data:
        print("No valid transfer rate data found.")
        return
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create grouped box plot
    sns.boxplot(data=plot_df, x='Storage_Type', y='Transfer_Rate', hue='Parallelism')
    
    plt.title('Estimated Transfer Rates by Storage Type and Parallelism', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Storage Type', fontsize=12)
    plt.ylabel('Transfer Rate (MiB/s)', fontsize=12)
    plt.legend(title='Parallelism', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_workflow_stages(wf_df: pd.DataFrame, 
                        save_path: str = "./analysis_data/workflow_stages.png") -> None:
    """
    Plot workflow stages and their characteristics.
    
    Parameters:
    - wf_df: Workflow DataFrame
    - save_path: Path to save the plot
    """
    # Group by stage order and task name
    stage_summary = wf_df.groupby(['stageOrder', 'taskName']).agg({
        'aggregateFilesizeMB': 'sum',
        'totalTime': 'sum',
        'opCount': 'sum'
    }).reset_index()
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: File size by stage
    stage_summary.plot(kind='bar', x='stageOrder', y='aggregateFilesizeMB', 
                      ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Aggregate File Size by Stage')
    axes[0, 0].set_xlabel('Stage Order')
    axes[0, 0].set_ylabel('File Size (MB)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Total time by stage
    stage_summary.plot(kind='bar', x='stageOrder', y='totalTime', 
                      ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Total Time by Stage')
    axes[0, 1].set_xlabel('Stage Order')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Operation count by stage
    stage_summary.plot(kind='bar', x='stageOrder', y='opCount', 
                      ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Operation Count by Stage')
    axes[1, 0].set_xlabel('Stage Order')
    axes[1, 0].set_ylabel('Operation Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Task distribution by stage
    task_counts = wf_df.groupby('stageOrder')['taskName'].nunique()
    task_counts.plot(kind='bar', ax=axes[1, 1], color='gold')
    axes[1, 1].set_title('Number of Unique Tasks by Stage')
    axes[1, 1].set_xlabel('Stage Order')
    axes[1, 1].set_ylabel('Number of Tasks')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_report(wf_df: pd.DataFrame, spm_results: Dict[str, Dict[str, Any]], 
                         io_breakdown: Dict[str, float], 
                         save_path: str = "./analysis_data/summary_report.txt") -> None:
    """
    Create a comprehensive summary report of the workflow analysis.
    
    Parameters:
    - wf_df: Workflow DataFrame
    - spm_results: SPM calculation results
    - io_breakdown: I/O time breakdown
    - save_path: Path to save the report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("WORKFLOW ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Workflow Overview
        f.write("1. WORKFLOW OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total number of tasks: {len(wf_df)}\n")
        f.write(f"Number of stages: {wf_df['stageOrder'].nunique()}\n")
        f.write(f"Unique task names: {list(wf_df['taskName'].unique())}\n")
        f.write(f"Total data processed: {wf_df['aggregateFilesizeMB'].sum():.2f} MB\n")
        f.write(f"Total operations: {wf_df['opCount'].sum():,}\n\n")
        
        # I/O Breakdown
        f.write("2. I/O TIME BREAKDOWN\n")
        f.write("-" * 20 + "\n")
        for task, time in io_breakdown.items():
            f.write(f"{task}: {time:.2f} seconds\n")
        f.write(f"Total I/O time: {sum(io_breakdown.values()):.2f} seconds\n\n")
        
        # SPM Analysis
        f.write("3. STORAGE PERFORMANCE METRICS (SPM) ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for pair, data in spm_results.items():
            f.write(f"\nProducer-Consumer Pair: {pair}\n")
            if 'best_storage_type' in data:
                f.write(f"  Best Storage Type: {data['best_storage_type']}\n")
                f.write(f"  Best Parallelism: {data['best_parallelism']}\n")
                f.write(f"  Best Rank: {data['best_rank']:.2f}\n")
                f.write("  Average Rank by Storage Type:\n")
                for storage, rank in data['avg_rank_by_storage'].items():
                    f.write(f"    {storage}: {rank:.2f}\n")
        
        # Recommendations
        f.write("\n4. RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        f.write("Based on the SPM analysis:\n")
        for pair, data in spm_results.items():
            if 'best_storage_type' in data:
                f.write(f"- For {pair}: Use {data['best_storage_type']} with {data['best_parallelism']} parallelism\n")
        
        f.write("\n5. DATA QUALITY NOTES\n")
        f.write("-" * 20 + "\n")
        f.write(f"Missing values in key columns:\n")
        for col in ['operation', 'aggregateFilesizeMB', 'totalTime', 'trMiB']:
            missing_count = wf_df[col].isnull().sum()
            if missing_count > 0:
                f.write(f"  {col}: {missing_count} missing values\n")
    
    print(f"Summary report saved to: {save_path}")


def plot_all_visualizations(wf_df: pd.DataFrame, spm_results: Dict[str, Dict[str, Any]], 
                           io_breakdown: Dict[str, float]) -> None:
    """
    Generate all visualization plots for the workflow analysis.
    
    Parameters:
    - wf_df: Workflow DataFrame
    - spm_results: SPM calculation results
    - io_breakdown: I/O time breakdown
    """
    print("Generating visualization plots...")
    
    # Set style for all plots
    plt.style.use('seaborn-v0_8')
    
    # Generate all plots
    plot_io_time_breakdown(io_breakdown)
    plot_storage_performance_comparison(spm_results)
    plot_spm_distribution(spm_results)
    plot_estimated_transfer_rates(wf_df)
    plot_workflow_stages(wf_df)
    
    # Create summary report
    create_summary_report(wf_df, spm_results, io_breakdown)
    
    print("All visualizations completed!") 