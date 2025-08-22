# Workflow Analysis Modules

This directory contains the core Python modules for workflow analysis and SPM (Storage Performance Modeling) calculations. These modules provide a modular architecture for analyzing workflow performance across different storage systems and parallelism configurations.

## Scripts Overview

The workflow analysis pipeline has been split into two separate scripts for better modularity:

### 1. `workflow_data_loader.py` - Data Loading Phase

**Purpose**: Loads all datalife JSON files for a given workflow and collects them into a CSV dataframe.

**What it does**:
- Loads workflow data from JSON files using the `load_workflow_data` function
- Processes and transforms the raw data
- Saves the processed workflow data to CSV format
- This corresponds to Step 1 of the workflow analysis pipeline

**Usage**:
```bash
# Load data for a specific workflow
python workflow_data_loader.py --workflow <workflow_name>

# List available workflows
python workflow_data_loader.py --list-workflows

# Specify output directory
python workflow_data_loader.py --workflow <workflow_name> --output-dir ./my_data

# Use a different CSV filename
python workflow_data_loader.py --workflow <workflow_name> --csv-filename my_workflow.csv
```

**Output**: 
- `./analysis_data/{workflow_name}_workflow_data.csv` - The processed workflow data

### 2. `workflow_analyzer.py` - Analysis Phase

**Purpose**: Performs complete workflow analysis on a CSV dataframe.

**What it does**:
- Takes a CSV file as input (created by the loader)
- Creates a working copy of the data (original CSV file remains unchanged)
- Performs all analysis steps (I/O breakdown, staging rows, SPM calculations, etc.)
- Generates results and visualizations
- This corresponds to Steps 2+ of the workflow analysis pipeline

**Important**: The script **does not modify** the input CSV file, so you can safely run it multiple times with the same input.

**Usage**:
```bash
# Analyze workflow from CSV file (workflow name auto-extracted from filename)
python workflow_analyzer.py path/to/my_workflow_workflow_data.csv

# Specify workflow name explicitly (overrides auto-extraction)
python workflow_analyzer.py path/to/workflow_data.csv --workflow my_workflow

# Use different IOR data path
python workflow_analyzer.py path/to/workflow_data.csv --ior-data path/to/ior_data.csv

# Don't save results to files
python workflow_analyzer.py path/to/workflow_data.csv --no-save
```

**Workflow Name Auto-Extraction**:
The script automatically extracts the workflow name from CSV filenames following the pattern:
- `{workflow_name}_workflow_data*.csv` → extracts `{workflow_name}`
- Examples:
  - `my_workflow_workflow_data.csv` → `my_workflow`
  - `ddmd_4n_l_workflow_data.csv` → `ddmd_4n_l`
  - `test_workflow_workflow_data_v2.csv` → `test_workflow`
- Fallback: If the pattern doesn't match, removes `.csv` extension

**Output**:
- `./analysis_data/{workflow_name}_original_workflow_data.csv` - Original data before modifications
- `./analysis_data/{workflow_name}_processed_workflow_data.csv` - Modified data with staging rows and transfer rates
- `./analysis_data/{workflow_name}_spm_results.json` - SPM calculation results
- `./workflow_spm_results/{workflow_name}_filtered_spm_results.csv` - Filtered SPM results
- `./workflow_spm_results/{workflow_name}_intermediate_estT_results.csv` - Intermediate DAG edges with estT values (clean format)

### Typical Workflow

The typical workflow is:

1. **Load Data**: Use `workflow_data_loader.py` to load and process JSON files
2. **Analyze Data**: Use `workflow_analyzer.py` to analyze the CSV file

**Example**:
```bash
# Step 1: Load data for a workflow
python workflow_data_loader.py --workflow my_workflow

# Step 2: Analyze the loaded data
python workflow_analyzer.py ./analysis_data/my_workflow_workflow_data.csv
```

### Benefits of This Split

1. **Separation of Concerns**: Data loading and analysis are now separate
2. **Reusability**: You can analyze the same CSV file multiple times without reloading JSON data
3. **Flexibility**: You can modify the CSV file manually and re-analyze
4. **Debugging**: Easier to debug issues in either the loading or analysis phase
5. **Performance**: Avoid reloading JSON files when you only need to re-run analysis

### Intermediate Results CSV Generation

The `workflow_spm_calculator.py` module now automatically generates detailed intermediate results in CSV format during the SPM calculation process. This provides granular insights into the workflow graph structure and per-task time estimations.

**Generation Process**:
- **Automatic Creation**: The intermediate CSV is generated automatically when `calculate_spm_for_workflow()` is called with a `workflow_name` parameter
- **Location**: Saved to `./workflow_spm_results/{workflow_name}_intermediate_estT_results.csv`
- **Content**: Contains detailed information about each edge in the workflow graph
- **Structure**: One row per estT key combination for maximum granularity

**Key Features**:
- **Per-Task Time Estimation**: Detailed time estimates for each producer-consumer pair without SPM values
- **Stage Order Tracking**: Preserves producer and consumer stage order information
- **Edge Key Generation**: Creates unique edge keys for traceability
- **Complete Edge Information**: All edge attributes preserved for analysis
- **Granular Analysis**: Each storage/parallelism combination gets its own row

**Use Cases**:
- **Debugging**: Detailed inspection of workflow graph structure
- **Analysis**: Granular performance analysis of individual edges
- **Machine Learning**: Complete data points for ML model training
- **Validation**: Verification of edge creation and attribute assignment
- **Research**: Detailed workflow performance research and analysis

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
**Purpose**: SPM (Storage Performance Modeling) calculations and workflow graph construction with robust validation and intermediate results logging.

**Key Functions**:
- `add_workflow_graph_nodes()`: Add nodes to workflow graph
- `add_producer_consumer_edge()`: Add edges between producer-consumer pairs with intermediate CSV logging
- `calculate_spm_for_workflow()`: Calculate SPM values for entire workflow
- `convert_operation_to_string()`: Convert numeric operations to string operations
- `filter_storage_options()`: Filter storage options based on workflow constraints
- `select_best_storage_and_parallelism()`: Select optimal storage and parallelism
- `normalize_estT_values()`: Normalize estimated time values
- `extract_SPM_estT_values()`: Extract SPM and estT values from workflow graph

**Key Features**:
- **Negative SPM Prevention**: Comprehensive validation to prevent negative SPM values
- **Data Quality Validation**: Checks for negative or zero aggregateFilesizeMB values
- **Robust Error Handling**: Graceful handling of edge cases and invalid data
- **Enhanced Debugging**: Detailed warnings and error messages for troubleshooting
- **Improved Edge Detection**: Better handling of stage_in and stage_out operations
- **Operation Standardization**: Converts numeric operations to strings for consistent processing
- **Intermediate Results Logging**: Saves detailed DAG edge information to CSV for analysis
- **Stage Order Tracking**: Preserves and logs producer/consumer stage order information
- **Edge Key Generation**: Creates unique edge keys for traceability and analysis

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
- **Intermediate CSV Results**: `{workflow_name}_intermediate_estT_results.csv` with detailed edge information

**Operation Handling**:
- Uses `convert_operation_to_string()` to ensure consistent string operations
- Supports both numeric and string operations with automatic conversion
- Handles cp/scp operations for storage type transitions

**Validation Features**:
- Validates that aggregateFilesizeMB values are positive
- Ensures estimated transfer rates are positive
- Prevents negative SPM calculations
- Handles edge cases with zero or invalid values

**Intermediate Results CSV Structure**:
The `add_producer_consumer_edge()` function generates a detailed CSV file with one row per estT key combination:

**Main Columns** (15): Basic producer-consumer information
- `producer_node`, `consumer_node`, `producer_task`, `consumer_task`
- `producer_stage_order`, `consumer_stage_order`
- `producer_operation`, `consumer_operation`, `producer_storage`, `consumer_storage`
- `producer_parallelism`, `consumer_parallelism`, `producer_filesize_mb`, `consumer_filesize_mb`
- `file_name`

**Edge Columns** (8): Edge-specific attributes
- `edge_key`, `edge_prod_aggregateFilesizeMB`, `edge_cons_aggregateFilesizeMB`
- `edge_prod_max_parallelism`, `edge_cons_max_parallelism`
- `edge_prod_stage_order`, `edge_cons_stage_order`
- `edge_prod_task_name`, `edge_cons_task_name`, `edge_file_name`

**estT Columns** (2): Individual estT keys
- `estT_prod_key`: Producer estT key (e.g., `estT_prod_beegfs_8p`)
- `estT_cons_key`: Consumer estT key (e.g., `estT_cons_ssd_16p`)

**Structure**:
- **One row per estT combination**: Each row represents a specific producer-consumer-storage-parallelism combination
- **Individual keys**: Each estT key combination gets its own row, making analysis easier
- **Complete traceability**: All edge information preserved for each combination

**Benefits**:
- **Individual analysis**: Each estT key combination can be analyzed separately
- **Easier filtering**: Can filter by specific storage types or parallelism levels
- **Better for ML**: Each row is a complete data point for machine learning
- **Complete data**: All estT key combinations preserved without loss
- **Key-based analysis**: Focus on storage/parallelism combinations rather than specific values

**Usage**:
```python
from workflow_spm_calculator import calculate_spm_for_workflow, filter_storage_options, convert_operation_to_string

# Basic usage (generates intermediate CSV automatically)
spm_results = calculate_spm_for_workflow(wf_df, workflow_name="my_workflow")

# With debugging enabled
spm_results = calculate_spm_for_workflow(wf_df, debug=True, workflow_name="my_workflow")

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
- **Universal Virtual Producers**: Creates `stage_in-{taskName}` operations with `operation: 'none'` for ALL tasks (not just first stage)
- **Universal Virtual Consumers**: Creates `stage_out-{taskName}` operations with `operation: 'none'` for ALL tasks
- **Storage Type Transitions**: Handles transitions between different storage types (beegfs-ssd, beegfs-tmpfs, etc.)
- **Operation Standardization**: Uses `standardize_operation()` for consistent string operations
- **Parallelism Management**: Splits operations by max parallelism of 60 files per row
- **Stage Simulation**: Simulates stage_in and stage_out operations for workflow analysis
- **Virtual Operation Placement**: 
  - Virtual "none" operations for stage_in are placed at `stageOrder - 0.5` (before the task)
  - Virtual "none" operations for stage_out are placed at `stageOrder + 0.5` (after the task)

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

### Universal Virtual Producer and Consumer Operations (Latest Update)
**Issue**: Virtual "none" operations were only being created for the first stage (stageOrder == 1) and limited stage_out operations, limiting the SPM analysis coverage.

**Solutions Implemented**:

1. **Universal Virtual Producer Creation**:
   - **All Tasks Coverage**: Now creates `stage_in-{taskName}` operations with `operation: 'none'` for ALL tasks in the workflow
   - **Consistent Stage Placement**: Virtual operations are placed at `stageOrder - 0.5` for each task
   - **Virtual Storage Type**: All virtual producers use `storageType: 'beegfs'` for consistent SPM calculations

2. **Universal Virtual Consumer Creation**:
   - **All Tasks Coverage**: Now creates `stage_out-{taskName}` operations with `operation: 'none'` for ALL tasks in the workflow
   - **Consumer Stage Placement**: Virtual operations are placed at `stageOrder + 0.5` for each task (following consumer pattern)
   - **Virtual Storage Type**: All virtual consumers use `storageType: 'beegfs'` for consistent SPM calculations

3. **Enhanced Data Staging Logic** (`workflow_data_staging.py`):
   - **New Section 1b**: Added dedicated section for creating virtual "none" operations for all stage_in tasks
   - **New Section 1c**: Added dedicated section for creating virtual "none" operations for all stage_out tasks
   - **Universal Task Processing**: Processes all tasks regardless of stage order
   - **Proper Stage Ordering**: 
     - Virtual stage_in operations are correctly positioned before their corresponding tasks
     - Virtual stage_out operations are correctly positioned after their corresponding tasks
   - **Consistent Naming**: Uses `stage_in-{taskName}` and `stage_out-{taskName}` patterns for all virtual operations

4. **Improved SPM Analysis Coverage**:
   - **Complete Virtual Producer Coverage**: Every task now has a virtual producer that can be matched in SPM calculations
   - **Complete Virtual Consumer Coverage**: Every task now has a virtual consumer that can be matched in SPM calculations
   - **Enhanced `handle_stage_in_none_producers()` Function**: Can now process virtual producers for all tasks, not just the first stage
   - **Better Workflow Modeling**: All tasks can have virtual data staging operations represented for both input and output

**Example Output**:
For a workflow with tasks `task1`, `task2`, `task3`, the system now creates:
- `stage_in-task1` with operation "none" at stage 0.5
- `stage_out-task1` with operation "none" at stage 1.5
- `stage_in-task2` with operation "none" at stage 1.5  
- `stage_out-task2` with operation "none" at stage 2.5
- `stage_in-task3` with operation "none" at stage 2.5
- `stage_out-task3` with operation "none" at stage 3.5

**Virtual Operation Characteristics**:
- `operation: 'none'` (virtual operation)
- `storageType: 'beegfs'` (virtual storage)
- `totalTime: 0` (no actual time cost)
- `trMiB: 1.0` (dummy transfer rate)
- **Stage_in**: `stageOrder: taskStage - 0.5` (positioned before actual task)
- **Stage_out**: `stageOrder: taskStage + 0.5` (positioned after actual task)

### String Operation Standardization (Previous Update)
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