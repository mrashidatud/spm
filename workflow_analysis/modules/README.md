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
- `standardize_operation(operation)`: Convert any operation format to standardized string format
- `transform_store_code(storage_name)`: Convert storage names to numeric codes
- `decode_store_code(code)`: Convert numeric codes back to storage names
- `bytes_to_mb(size_string)`: Convert size strings to MB
- `calculate_io_time_breakdown()`: Calculate I/O time breakdown for tasks

**Key Features**:
- **Operation Standardization**: Automatically converts all operations to standardized string format ('write', 'read', 'cp', 'scp', 'none')
- **Flexible Input Handling**: Accepts operations as integers, strings, or numeric strings
- **Data Consistency**: Ensures all operations are consistently formatted throughout the system

**Inputs**: 
- Workflow name string
- CSV data files
- Size strings in various formats
- Operations in any format (integer, string, or numeric string)

**Outputs**: 
- Processed DataFrame with standardized string operations
- Task order dictionaries
- I/O time breakdowns
- Transformed data structures

**Usage**:
```python
from workflow_data_utils import load_workflow_data, calculate_io_time_breakdown, standardize_operation
wf_df, task_order, wf_dict = load_workflow_data("ddmd_4n_l")

# Standardize operations manually if needed
operation_str = standardize_operation(0)  # Returns 'write'
```

### 3. `workflow_interpolation.py`
**Purpose**: 4D interpolation and extrapolation for transfer rate estimation with robust error handling and validation.

**Key Functions**:
- `calculate_4d_interpolation_with_extrapolation()`: Perform 4D interpolation/extrapolation with negative value validation
- `estimate_transfer_rates_for_workflow()`: Estimate transfer rates for all workflow tasks
- `calculate_aggregate_filesize_per_node()`: Calculate aggregate file size per node

**Key Features**:
- **Negative Value Prevention**: Comprehensive validation to prevent negative transfer rates
- **Robust Extrapolation**: Smart fallback strategies when extrapolation produces negative values
- **Enhanced Debugging**: Detailed debugging output to identify data quality issues
- **Data Quality Checks**: Validation of input data for negative values and NaN entries
- **Improved Interpolation Logic**: Uses median of positive values instead of simple averaging
- **Flexible Operation Handling**: Supports both string and numeric operations with automatic conversion

**Inputs**:
- IOR benchmark data DataFrame (with string operations: 'write', 'read', 'cp', 'scp')
- Workflow DataFrame with task information (with standardized string operations)
- Target parameters (operation, file size, nodes, parallelism, transfer size)
- Debug flag for detailed output

**Outputs**:
- Estimated transfer rates for each storage type and parallelism level
- Transfer size slopes for performance modeling
- Updated workflow DataFrame with estimated values
- Debug information about data quality and interpolation process

**Operation Handling**:
The function handles operations flexibly:
- Accepts both string operations ('write', 'read', 'cp', 'scp') and numeric operations (0, 1)
- Automatically converts numeric operations to strings internally
- Maps workflow operations to IOR benchmark operations consistently

**Error Handling**:
- Validates that all input data values are positive
- Prevents negative transfer rates through robust fallback strategies
- Handles NaN values gracefully
- Provides detailed warnings for data quality issues

**Debug Features**:
- Shows original IOR data analysis and negative value detection
- Displays step-by-step filtering process
- Reveals bounds calculation details and extrapolation formulas
- Identifies data quality issues in source data

**Usage**:
```python
from workflow_interpolation import estimate_transfer_rates_for_workflow
# Basic usage
wf_df = estimate_transfer_rates_for_workflow(wf_df, ior_data, storage_list, allowed_parallelism)

# With debugging enabled
wf_df = estimate_transfer_rates_for_workflow(wf_df, ior_data, storage_list, allowed_parallelism, debug=True)
```

### 4. `workflow_spm_calculator.py`
**Purpose**: SPM (Storage Performance Modeling) calculations and workflow graph construction with robust validation.

**Key Functions**:
- `add_workflow_graph_nodes()`: Add nodes to workflow graph
- `add_producer_consumer_edge()`: Add edges between producer-consumer pairs
- `calculate_spm_for_workflow()`: Calculate SPM values for entire workflow
- `convert_operation_to_string()`: Convert numeric operations to string operations
- `filter_storage_options()`: Filter storage options based on workflow constraints
- `select_best_storage_and_parallelism()`: Select optimal storage and parallelism
- `normalize_estT_values()`: Normalize estimated time values

**Key Features**:
- **Negative SPM Prevention**: Comprehensive validation to prevent negative SPM values
- **Data Quality Validation**: Checks for negative or zero aggregateFilesizeMB values
- **Robust Error Handling**: Graceful handling of edge cases and invalid data
- **Enhanced Debugging**: Detailed warnings and error messages for troubleshooting
- **Improved Edge Detection**: Better handling of stage_in and stage_out operations
- **Operation Standardization**: Converts numeric operations to strings for consistent processing

**Inputs**:
- Workflow DataFrame with estimated transfer rates and standardized string operations
- Workflow configuration
- Storage and parallelism constraints
- Debug flag for detailed output

**Outputs**:
- NetworkX workflow graph
- SPM values for producer-consumer pairs
- Filtered storage options
- Best storage and parallelism selections
- Validation warnings and error messages

**Operation Handling**:
- Uses `convert_operation_to_string()` to ensure consistent string operations
- Supports both numeric and string operations with automatic conversion
- Handles cp/scp operations for storage type transitions

**Validation Features**:
- Validates that aggregateFilesizeMB values are positive
- Ensures estimated transfer rates are positive
- Prevents negative SPM calculations
- Handles edge cases with zero or invalid values

**Usage**:
```python
from workflow_spm_calculator import calculate_spm_for_workflow, filter_storage_options, convert_operation_to_string
# Basic usage
spm_results = calculate_spm_for_workflow(wf_df)

# With debugging enabled
spm_results = calculate_spm_for_workflow(wf_df, debug=True)

filtered_results = filter_storage_options(spm_results, "ddmd_4n_l")

# Convert operations manually if needed
op_str = convert_operation_to_string(0)  # Returns 'write'
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

### 7. `workflow_data_staging.py`
**Purpose**: Insert data staging (I/O) rows into workflow DataFrame to simulate data stage_in and stage_out operations.

**Key Functions**:
- `insert_data_staging_rows()`: Insert staging rows for data movement between storage types
- `get_file_groups()`: Group files for parallel processing with max parallelism limits

**Key Features**:
- **Storage Type Transitions**: Handles transitions between different storage types (beegfs-ssd, beegfs-tmpfs, etc.)
- **Operation Standardization**: Uses `standardize_operation()` for consistent string operations
- **Parallelism Management**: Splits operations by max parallelism of 60 files per row
- **Stage Simulation**: Simulates stage_in and stage_out operations for workflow analysis

**Inputs**:
- Workflow DataFrame with standardized string operations
- Debug flag for detailed output

**Outputs**:
- Enhanced workflow DataFrame with staging rows
- Data movement operations between storage types
- Stage_in and stage_out task simulations

**Usage**:
```python
from workflow_data_staging import insert_data_staging_rows
# Add staging rows to workflow DataFrame
enhanced_wf_df = insert_data_staging_rows(wf_df, debug=True)
```

### 8. `workflow_results_exporter.py`
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
workflow_data_utils.py (Data Loading & Operation Standardization)
         ↓
workflow_data_staging.py (Data Staging Operations)
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

## Operation Code Handling

The modules now use a standardized string-based approach for all operations throughout the system:

### Standardized Operation Strings
- **'write'**: Write operations
- **'read'**: Read operations
- **'cp'**: Copy operations  
- **'scp'**: Secure copy operations
- **'none'**: No operation (for staging tasks)

### Operation Standardization Functions

#### `standardize_operation()` in `workflow_data_utils.py`
Converts any operation format (integer, string, or numeric string) to standardized string format:
- `0` or `'0'` → `'write'`
- `1` or `'1'` → `'read'`
- `2` or `'2'` → `'cp'`
- `3` or `'3'` → `'scp'`
- `-1` or `'-1'` → `'none'`

#### `convert_operation_to_string()` in `workflow_spm_calculator.py`
Converts numeric operations to string operations for SPM calculations:
- `0` → `'write'`
- `1` → `'read'`

### Automatic Standardization
- **Data Loading**: All operations are automatically standardized to strings when loading workflow data
- **Interpolation**: The system handles both string and numeric operations, converting them internally
- **SPM Calculations**: Operations are converted to strings for consistent processing
- **Staging Operations**: Uses standardized string operations for stage_in and stage_out tasks

**Important**: The system now consistently uses string operations throughout all modules for better clarity and consistency.

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
├── workflow_data_staging.py     # Data staging operations
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

## Stage Ordering

### 1-Based Stage Numbering
All workflows in this system use 1-based stage numbering:
- **First stage**: Always stage 1 (not stage 0)
- **Subsequent stages**: Stage 2, 3, 4, etc.
- **Automatic normalization**: If legacy workflows with stage 0 are detected, all stages are automatically shifted by +1

### Stage Order Normalization
The system automatically handles stage order normalization:
- **Detection**: Checks if any workflow starts from stage 0
- **Normalization**: If stage 0 is found, all stages are shifted by +1
- **Result**: Ensures all workflows consistently start from stage 1

## Recent Improvements and Bug Fixes

### String Operation Standardization (Latest Update)
**Issue**: Inconsistent operation representation across modules, mixing integers and strings for operations.

**Solutions Implemented**:

1. **Operation Standardization Functions**:
   - **`standardize_operation()`** in `workflow_data_utils.py`: Converts any operation format to standardized strings
   - **`convert_operation_to_string()`** in `workflow_spm_calculator.py`: Converts numeric operations to strings
   - **Automatic Standardization**: All operations are now consistently represented as strings throughout the system

2. **Enhanced Module Integration**:
   - **`workflow_data_utils.py`**: Automatically standardizes operations during data loading
   - **`workflow_interpolation.py`**: Handles both string and numeric operations with automatic conversion
   - **`workflow_spm_calculator.py`**: Uses standardized string operations for consistent processing
   - **`workflow_data_staging.py`**: Uses standardized operations for stage_in and stage_out tasks

3. **Improved Data Consistency**:
   - All CSV files now use consistent string operations ('write', 'read', 'cp', 'scp', 'none')
   - Better clarity and consistency across the entire system
   - Support for storage type transitions using cp/scp operations

**Standardized Operation Mapping**:
- `0` or `'0'` → `'write'`
- `1` or `'1'` → `'read'`
- `2` or `'2'` → `'cp'`
- `3` or `'3'` → `'scp'`
- `-1` or `'-1'` → `'none'`

### Negative Value Prevention (Previous Update)
**Issue**: Negative transfer rates and SPM values were being calculated due to extrapolation beyond data bounds and data quality issues.

**Solutions Implemented**:

1. **Enhanced Interpolation Validation** (`workflow_interpolation.py`):
   - Added comprehensive validation to prevent negative transfer rates
   - Implemented robust extrapolation logic with smart fallback strategies
   - Added detailed debugging to identify data quality issues
   - Improved interpolation logic using median of positive values
   - Added step-by-step data filtering validation

2. **SPM Calculation Validation** (`workflow_spm_calculator.py`):
   - Added validation for negative or zero aggregateFilesizeMB values
   - Implemented checks for negative estimated transfer rates
   - Enhanced error handling for edge cases
   - Added detailed warning messages for troubleshooting

3. **Data Quality Checks**:
   - Validation of input data for negative values and NaN entries
   - Comprehensive debugging output to identify problematic data
   - Graceful handling of edge cases with invalid values

**Debug Features Added**:
- Original IOR data analysis and negative value detection
- Step-by-step filtering process visualization
- Bounds calculation details and extrapolation formulas
- Data quality issue identification in source data

**Usage with Debugging**:
```python
# Enable debugging for detailed output
wf_df = estimate_transfer_rates_for_workflow(wf_df, ior_data, storage_list, debug=True)
spm_results = calculate_spm_for_workflow(wf_df, debug=True)
```

These improvements ensure that the workflow analysis produces physically meaningful, positive transfer rates and SPM values while providing detailed debugging information to identify and resolve data quality issues. 