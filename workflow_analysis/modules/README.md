# Workflow Analysis Modules

This directory contains the core Python modules for workflow analysis and SPM (Storage Performance Modeling) calculations. These modules provide a modular architecture for analyzing workflow performance across different storage systems and parallelism configurations.

## Module Overview

### 1. `workflow_config.py`
**Purpose**: Configuration management for workflow analysis parameters and settings.

**Key Features**:
- Global configuration flags (MULTI_NODES, NORMALIZE, DEBUG)
- Storage type definitions (STORAGE_LIST)
- Data size mappings
- Test configurations for different workflows
- Target parameters and operation dictionaries

**Inputs**: None (configuration file)
**Outputs**: Configuration constants and dictionaries used by other modules

**Usage**:
```python
from workflow_config import MULTI_NODES, STORAGE_LIST, TEST_CONFIGS, DEFAULT_WF
```

### 2. `workflow_data_utils.py`
**Purpose**: Data loading, transformation, and utility functions for workflow data processing.

**Key Functions**:
- `load_workflow_data(workflow_name)`: Load workflow data from CSV files
- `transform_store_code(storage_name)`: Convert storage names to numeric codes
- `decode_store_code(code)`: Convert numeric codes back to storage names
- `bytes_to_mb(size_string)`: Convert size strings to MB
- `calculate_io_time_breakdown()`: Calculate I/O time breakdown for tasks

**Inputs**: 
- Workflow name string
- CSV data files
- Size strings in various formats

**Outputs**: 
- Processed DataFrame
- Task order dictionaries
- I/O time breakdowns
- Transformed data structures

**Usage**:
```python
from workflow_data_utils import load_workflow_data, calculate_io_time_breakdown
wf_df, task_order, wf_dict = load_workflow_data("ddmd_4n_l")
```

### 3. `workflow_interpolation.py`
**Purpose**: 4D interpolation and extrapolation for transfer rate estimation.

**Key Functions**:
- `calculate_4d_interpolation_with_extrapolation()`: Perform 4D interpolation/extrapolation
- `estimate_transfer_rates_for_workflow()`: Estimate transfer rates for all workflow tasks
- `calculate_aggregate_filesize_per_node()`: Calculate aggregate file size per node

**Inputs**:
- IOR benchmark data DataFrame
- Workflow DataFrame with task information
- Target parameters (operation, file size, nodes, parallelism, transfer size)

**Outputs**:
- Estimated transfer rates for each storage type and parallelism level
- Transfer size slopes for performance modeling
- Updated workflow DataFrame with estimated values

**Usage**:
```python
from workflow_interpolation import estimate_transfer_rates_for_workflow
wf_df = estimate_transfer_rates_for_workflow(wf_df, ior_data, storage_list, allowed_parallelism)
```

### 4. `workflow_spm_calculator.py`
**Purpose**: SPM (Storage Performance Modeling) calculations and workflow graph construction.

**Key Functions**:
- `add_workflow_graph_nodes()`: Add nodes to workflow graph
- `add_producer_consumer_edge()`: Add edges between producer-consumer pairs
- `calculate_spm_for_workflow()`: Calculate SPM values for entire workflow
- `filter_storage_options()`: Filter storage options based on workflow constraints
- `select_best_storage_and_parallelism()`: Select optimal storage and parallelism
- `normalize_estT_values()`: Normalize estimated time values

**Inputs**:
- Workflow DataFrame with estimated transfer rates
- Workflow configuration
- Storage and parallelism constraints

**Outputs**:
- NetworkX workflow graph
- SPM values for producer-consumer pairs
- Filtered storage options
- Best storage and parallelism selections

**Usage**:
```python
from workflow_spm_calculator import calculate_spm_for_workflow, filter_storage_options
spm_results = calculate_spm_for_workflow(wf_df)
filtered_results = filter_storage_options(spm_results, "ddmd_4n_l")
```

### 5. `workflow_visualization.py`
**Purpose**: Visualization and reporting functions for workflow analysis results.

**Key Functions**:
- `plot_all_visualizations()`: Generate comprehensive visualization suite
- `create_summary_report()`: Create text-based summary reports
- `plot_spm_comparison()`: Plot SPM comparisons across storage types
- `plot_io_breakdown()`: Visualize I/O time breakdowns

**Inputs**:
- Workflow DataFrame
- SPM results dictionary
- I/O breakdown data
- Configuration parameters

**Outputs**:
- Matplotlib plots and figures
- Summary report text files
- Performance comparison visualizations

**Usage**:
```python
from workflow_visualization import plot_all_visualizations, create_summary_report
plot_all_visualizations(wf_df, spm_results, io_breakdown)
create_summary_report(wf_df, spm_results, io_breakdown, "report.txt")
```

### 6. `workflow_analysis_main.py`
**Purpose**: Main orchestration module that coordinates the entire workflow analysis pipeline.

**Key Functions**:
- `run_workflow_analysis()`: Execute complete workflow analysis pipeline
- `main()`: Command-line interface for workflow analysis

**Inputs**:
- Workflow name
- Configuration parameters
- Data file paths

**Outputs**:
- Complete analysis results
- Generated reports and visualizations
- Performance metrics and recommendations

**Usage**:
```python
from workflow_analysis_main import run_workflow_analysis
results = run_workflow_analysis("ddmd_4n_l", save_results=True)
```

### 7. `workflow_results_exporter.py`
**Purpose**: Export producer-consumer storage selection and parallelism results to CSV format and generate detailed reports.

**Key Functions**:
- `extract_producer_consumer_results()`: Extract results from SPM calculations into structured DataFrame
- `save_producer_consumer_results()`: Save results to CSV file with specified columns
- `create_detailed_producer_consumer_report()`: Generate detailed text report
- `analyze_storage_distribution()`: Analyze storage type distribution across pairs
- `print_storage_analysis()`: Print storage distribution analysis

**Inputs**:
- SPM results dictionary from workflow analysis
- Workflow DataFrame with task information
- Output directory and filename preferences

**Outputs**:
- CSV file with columns: producer, producerStage, consumer, consumerStage, prodParallelism, consParallelism, p-c-Storage, p-c-SPM
- Detailed text report with analysis summary
- Storage distribution analysis

**Usage**:
```python
from workflow_results_exporter import save_producer_consumer_results, print_storage_analysis
csv_path = save_producer_consumer_results(spm_results, wf_df, "my_workflow")
print_storage_analysis(results_df)
```

## Data Flow

```
workflow_config.py (Configuration)
         ↓
workflow_data_utils.py (Data Loading)
         ↓
workflow_interpolation.py (Transfer Rate Estimation)
         ↓
workflow_spm_calculator.py (SPM Calculations)
         ↓
workflow_visualization.py (Results Visualization)
         ↓
workflow_results_exporter.py (Results Export)
         ↓
workflow_analysis_main.py (Orchestration)
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **networkx**: Graph construction and analysis
- **matplotlib**: Visualization and plotting
- **scipy**: Scientific computing (if needed)

## File Structure

```
modules/
├── README.md                    # This file
├── workflow_config.py           # Configuration management
├── workflow_data_utils.py       # Data loading and utilities
├── workflow_interpolation.py    # Transfer rate estimation
├── workflow_spm_calculator.py   # SPM calculations
├── workflow_visualization.py    # Visualization and reporting
├── workflow_results_exporter.py # Results export to CSV
└── workflow_analysis_main.py    # Main orchestration
```

## Usage Example

```python
# Import modules
from modules.workflow_config import DEFAULT_WF, TEST_CONFIGS
from modules.workflow_data_utils import load_workflow_data
from modules.workflow_interpolation import estimate_transfer_rates_for_workflow
from modules.workflow_spm_calculator import calculate_spm_for_workflow
from modules.workflow_visualization import plot_all_visualizations
from modules.workflow_results_exporter import save_producer_consumer_results

# Run complete analysis
workflow_name = "ddmd_4n_l"
wf_df, task_order, wf_dict = load_workflow_data(workflow_name)
# ... additional processing steps
spm_results = calculate_spm_for_workflow(wf_df)
plot_all_visualizations(wf_df, spm_results, io_breakdown)

# Export results to CSV
csv_path = save_producer_consumer_results(spm_results, wf_df, workflow_name)
```

## Notes

- All modules are designed to work together in a pipeline
- Configuration is centralized in `workflow_config.py`
- Data flows from loading → processing → analysis → visualization
- Each module can be used independently or as part of the complete pipeline
- Error handling and logging are built into each module 