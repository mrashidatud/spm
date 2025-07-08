"""
Workflow Analysis Modules Package

This package contains all the core modules for workflow analysis and SPM calculations.
"""

# Import main functions for easy access
from .workflow_config import (
    MULTI_NODES, NORMALIZE, DEBUG, STORAGE_LIST, DATA_SIZE_KB,
    WF_PARAMS, TARGET_PARAMS, OP_DICT, TEST_CONFIGS, DEFAULT_WF,
    TARGET_TASKS, OSCACHE_SIZE_MB
)

from .workflow_data_utils import (
    load_workflow_data, calculate_io_time_breakdown,
    transform_store_code, decode_store_code, bytes_to_mb
)

from .workflow_interpolation import (
    calculate_4d_interpolation_with_extrapolation,
    estimate_transfer_rates_for_workflow,
    calculate_aggregate_filesize_per_node
)

from .workflow_spm_calculator import (
    add_workflow_graph_nodes, add_producer_consumer_edge,
    calculate_spm_for_workflow, filter_storage_options,
    select_best_storage_and_parallelism, normalize_estT_values,
    display_top_sorted_averaged_rank
)

from .workflow_visualization import (
    plot_all_visualizations, create_summary_report,
    plot_storage_performance_comparison, plot_io_time_breakdown,
    plot_spm_distribution, plot_estimated_transfer_rates, plot_workflow_stages
)

from .workflow_results_exporter import (
    extract_producer_consumer_results, save_producer_consumer_results,
    create_detailed_producer_consumer_report, analyze_storage_distribution,
    print_storage_analysis
)

from .workflow_data_staging import insert_data_staging_rows

# Package metadata
__version__ = "1.0.0"
__author__ = "Workflow Analysis Team"
__description__ = "Modular workflow analysis and SPM calculation tools"

# Convenience function for quick setup
def setup_workflow_analysis(workflow_name=None):
    """
    Quick setup function to initialize workflow analysis with default settings.
    
    Args:
        workflow_name: Name of the workflow to analyze (default: DEFAULT_WF)
    
    Returns:
        tuple: (workflow_df, task_order_dict, all_wf_dict)
    """
    if workflow_name is None:
        workflow_name = DEFAULT_WF
    
    return load_workflow_data(workflow_name)

# Export all the main functions and classes
__all__ = [
    # Configuration
    'MULTI_NODES', 'NORMALIZE', 'DEBUG', 'STORAGE_LIST', 'DATA_SIZE_KB',
    'WF_PARAMS', 'TARGET_PARAMS', 'OP_DICT', 'TEST_CONFIGS', 'DEFAULT_WF',
    'TARGET_TASKS', 'OSCACHE_SIZE_MB',
    
    # Data utilities
    'load_workflow_data', 'calculate_io_time_breakdown',
    'transform_store_code', 'decode_store_code', 'bytes_to_mb',
    
    # Interpolation
    'calculate_4d_interpolation_with_extrapolation',
    'estimate_transfer_rates_for_workflow',
    'calculate_aggregate_filesize_per_node',
    
    # SPM calculations
    'add_workflow_graph_nodes', 'add_producer_consumer_edge',
    'calculate_spm_for_workflow', 'filter_storage_options',
    'select_best_storage_and_parallelism', 'normalize_estT_values',
    'display_top_sorted_averaged_rank',
    
    # Visualization
    'plot_all_visualizations', 'create_summary_report',
    'plot_storage_performance_comparison', 'plot_io_time_breakdown',
    'plot_spm_distribution', 'plot_estimated_transfer_rates', 'plot_workflow_stages',
    
    # Results export
    'extract_producer_consumer_results', 'save_producer_consumer_results',
    'create_detailed_producer_consumer_report', 'analyze_storage_distribution',
    'print_storage_analysis',
    
    # Convenience
    'setup_workflow_analysis',
    
    # Data staging
    'insert_data_staging_rows'
] 