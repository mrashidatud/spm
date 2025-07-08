# SPM (Storage Performance Matching) Analysis System

A comprehensive system for analyzing scientific workflow performance and optimizing storage configurations using Storage Performance Matching (SPM).

## 🎯 Overview

This project provides tools and analysis capabilities for understanding and optimizing the performance of scientific workflows across different storage systems. It processes workflow I/O pattern profiles, estimates transfer rates using IOR benchmark data, and calculates Storage Performance Matching (SPM) to recommend optimal storage configurations.

**Main User Interface**: The `workflow_analysis/` directory contains the primary analysis tools that users will interact with.

## 📁 Project Structure

```
spm/
├── workflow_analysis/                 # 🎯 Main analysis system (Primary user interface)
│   ├── workflow_analysis_main.py     # 🆕 Main orchestration script (standalone)
│   ├── modules/                      # Core Python modules
│   │   ├── workflow_config.py        # Configuration management
│   │   ├── workflow_data_utils.py    # Data loading and utilities
│   │   ├── workflow_interpolation.py # Transfer rate estimation
│   │   ├── workflow_spm_calculator.py # SPM calculations
│   │   ├── workflow_visualization.py # Visualization and reporting
│   │   ├── workflow_data_staging.py  # Data staging operations
│   │   ├── workflow_results_exporter.py # Results export utilities
│   │   ├── README.md                 # Detailed module documentation
│   │   └── __init__.py              # Module initialization
│   ├── python_tests/                 # Comprehensive test suite
│   ├── ddmd/                        # DDMD workflow data
│   │   ├── ddmd_script_order.json   # Workflow configuration
│   │   └── ddmd_4n_pfs_large/      # Workflow execution data
│   │       └── 4n_pfs_t1/          # Trial run data
│   ├── 1kgenome/                    # 1K Genome workflow data
│   │   ├── 1kg_script_order.json   # Workflow configuration
│   │   └── par_6000_10n_nfs_ps300/ # Workflow execution data
│   ├── workflow_analysis.ipynb       # Main analysis notebook
│   ├── example_debug_usage.py        # Debug parameter example
│   ├── README.md                     # Detailed documentation
│   └── TODO.md                       # Development tasks
├── perf_profiles/                    # Storage performance benchmark data (Building blocks)
│   ├── updated_master_ior_df.csv     # ⭐ Required: IOR benchmark results for transfer rate prediction
│   ├── cp_data/                      # Copy operation benchmarks (ignored)
│   ├── ior_data/                     # IOR operation benchmarks (ignored)
│   └── plot/                         # Benchmark visualizations (ignored)
└── README.md                         # This file
```

## 🚀 Quick Start

### Option 1: Using the Jupyter Notebook (Recommended)

```bash
# Navigate to the workflow analysis directory
cd workflow_analysis

# Start Jupyter
jupyter notebook

# Open workflow_analysis.ipynb and run all cells
```

### Option 2: Using the Command-Line Interface

```bash
# Navigate to the workflow analysis directory
cd workflow_analysis

# Analyze a specific workflow
python3 workflow_analysis_main.py --workflow ddmd_4n_l

# Analyze all available workflows
python3 workflow_analysis_main.py --all

# Use custom IOR data path
python3 workflow_analysis_main.py --workflow ddmd_4n_l --ior-data ../perf_profiles/updated_master_ior_df.csv

# Run without saving results
python3 workflow_analysis_main.py --workflow ddmd_4n_l --no-save
```

### Option 3: Using Individual Modules

```python
from workflow_analysis.modules import (
    load_workflow_data,
    estimate_transfer_rates_for_workflow,
    calculate_spm_for_workflow
)

# Load workflow data
wf_df, task_order, wf_dict = load_workflow_data("ddmd_4n_l")

# Estimate transfer rates
wf_df = estimate_transfer_rates_for_workflow(wf_df, ior_data, storage_list)

# Calculate SPM values
spm_results = calculate_spm_for_workflow(wf_df)
```

## 📋 Available Workflows

The system supports analysis of multiple scientific workflows:

| Workflow | Description | Data Size | Nodes | Script Order File |
|----------|-------------|-----------|-------|-------------------|
| `ddmd_4n_l` | DDMD workflow | Large | 4 | `ddmd/ddmd_script_order.json` |
| `1kg` | 1K Genome workflow | Standard | Variable | `1kgenome/1kg_script_order.json` |


## 💾 Supported Storage Types

- **`localssd`** - Local SSD storage
- **`beegfs`** - BeeGFS/PFS storage
- **`tmpfs`** - Temporary file system
- **`nfs`** - Network File System
- **`pfs`** - Parallel File System (uses beegfs as proxy)

## 📊 Workflow Data Structure

### Required Directory Structure

Each workflow must follow this structure:
```
workflow_analysis/
├── {workflow_name}/
│   ├── {workflow_name}_script_order.json    # Required: Workflow configuration
│   └── {workflow_config_run}/
│       └── run_trial1/
│           └── {datalife_trace_files}      # Workflow execution traces
```

### Script Order JSON File Structure

The `{workflow_name}_script_order.json` file is required and must contain:

```json
{
    "task_name": {
        "stage_order": 0,                    # Integer: Stage execution order
        "parallelism": 12,                   # Integer: Number of parallel tasks
        "num_tasks": 12,                     # Integer: Total number of tasks
        "predecessors": {                    # Object: Input dependencies
            "predecessor_task": {
                "inputs": [                  # Array: Regex patterns for input files
                    "stage\\d{4}_task\\d{4}\\.h5"
                ]
            }
        },
        "outputs": [                         # Array: Regex patterns for output files
            "stage\\d{4}_task\\d{4}\\.dcd",
            "stage\\d{4}_task\\d{4}\\.h5"
        ]
    }
}
```

**Key Fields:**
- `stage_order`: Execution order (0, 1, 2, etc.)
- `parallelism`: Number of parallel tasks for this stage
- `num_tasks`: Total number of tasks in this stage
- `predecessors`: Defines input dependencies from previous stages
- `outputs`: Regex patterns matching output files from this stage

## 🔧 Key Features

### 📊 4D Interpolation System
- **Multi-dimensional analysis**: Interpolates transfer rates based on aggregate file size, number of nodes, parallelism, and transfer size
- **Storage and parallelism optimization**: Provides recommendations for both storage options and task I/O node selection (e.g., how many tasks per node to use)
- **Extrapolation support**: Handles values outside the benchmark data range
- **Multi-node support**: Proper handling of tasksPerNode calculations

### 🎯 SPM (Storage Performance Matching) Calculation
- **Producer-consumer analysis**: Calculates metrics for workflow stage transitions
- **Storage optimization**: Ranks storage configurations by performance
- **Stage-aware processing**: Handles stage_in and stage_out operations correctly

### 🏗️ Modular Architecture
- **Separation of concerns**: Each module handles specific functionality
- **Reusable components**: Functions can be used across different workflows
- **Easy extension**: Simple to add new workflows or storage types
- **Comprehensive testing**: Full test suite for all components

## ⚙️ Configuration

### Global Configuration Flags

```python
# In workflow_analysis/modules/workflow_config.py
MULTI_NODES = True    # Use tasksPerNode vs parallelism
NORMALIZE = True       # Enable normalization of SPM values
```

## 📊 Output Files

The analysis generates comprehensive output files:

### Data Files
- **`{workflow_name}_workflow_data.csv`** - Processed workflow data with estimated transfer rates
- **`{workflow_name}_spm_results.json`** - SPM calculation results and best configurations

### Reports
- **`summary_report.txt`** - Comprehensive analysis summary

## 📚 Requirements

### Python Dependencies

```bash
pip install pandas numpy matplotlib seaborn networkx scikit-learn scipy jupyter
```

### Core Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **networkx**: Graph analysis for workflow stages
- **scikit-learn**: Machine learning utilities
- **scipy**: Scientific computing
- **jupyter**: Interactive notebooks

### Required Data Files

**Essential File for Analysis:**
- **`perf_profiles/updated_master_ior_df.csv`** - IOR benchmark results for transfer rate prediction

**Note**: The `perf_profiles/` directory contains building blocks for storage performance profiles. Only the `updated_master_ior_df.csv` file is required for workflow analysis. Other files in this directory are used for generating the benchmark data and are not needed for end-user analysis.

## 🔧 Module Usage Guide

**📚 Detailed Module Documentation**: For comprehensive information about each module's functions, parameters, and usage examples, see [`workflow_analysis/modules/README.md`](workflow_analysis/modules/README.md).

### Core Modules and Their Functions

#### 1. `workflow_config.py` - Configuration Management
**Purpose**: Global configuration and constants
**Key Functions**:
- `STORAGE_LIST`: Available storage types
- `TEST_CONFIGS`: Workflow configurations
- `MULTI_NODES`, `NORMALIZE`: Global flags

#### 2. `workflow_data_utils.py` - Data Loading and Processing
**Purpose**: Load and process workflow data from datalife statistics
**Key Functions**:
```python
load_workflow_data(workflow_name)
# Input: workflow_name (str)
# Output: (wf_df, task_order, wf_dict)
# - wf_df: DataFrame with workflow data
# - task_order: List of task execution order
# - wf_dict: Dictionary with workflow configuration
```

#### 3. `workflow_interpolation.py` - Transfer Rate Estimation
**Purpose**: 4D interpolation for transfer rate estimation
**Key Functions**:
```python
estimate_transfer_rates_for_workflow(wf_df, ior_data, storage_list)
# Input: 
# - wf_df: Workflow DataFrame
# - ior_data: IOR benchmark data
# - storage_list: List of storage types
# Output: wf_df with estimated transfer rates added
```

#### 4. `workflow_spm_calculator.py` - SPM Calculations
**Purpose**: Calculate Storage Performance Matching metrics
**Key Functions**:
```python
calculate_spm_for_workflow(wf_df)
# Input: wf_df (DataFrame with workflow data)
# Output: Dictionary with SPM results for all producer-consumer pairs
```

#### 5. `workflow_visualization.py` - Visualization
**Purpose**: Generate plots and visualizations
**Key Functions**:
```python
plot_all_visualizations(wf_df, spm_results, io_breakdown)
# Input: 
# - wf_df: Workflow DataFrame
# - spm_results: SPM calculation results
# - io_breakdown: I/O time breakdown
# Output: Saves visualization files
```

#### 6. `workflow_analysis_main.py` - Main Orchestration
**Purpose**: Complete analysis pipeline orchestration
**Key Functions**:
```python
run_workflow_analysis(workflow_name, ior_data_path, save_results)
# Input:
# - workflow_name: Name of workflow to analyze
# - ior_data_path: Path to IOR benchmark data
# - save_results: Boolean to save results
# Output: Dictionary with all analysis results
```

### Input Data Requirements

#### Workflow Data
- **Format**: JSON datalife trace files
- **Location**: `workflow_analysis/{workflow_name}/{config_run}/run_trial1/`
- **Content**: File I/O operations with timestamps, file sizes, operations

#### IOR Benchmark Data (Required)
- **Format**: CSV file (`updated_master_ior_df.csv`)
- **Location**: `perf_profiles/updated_master_ior_df.csv`
- **Content**: Transfer rates for different storage types, file sizes, parallelism
- **Note**: This is the only file needed from the `perf_profiles/` directory

#### Script Order Configuration
- **Format**: JSON file (`{workflow_name}_script_order.json`)
- **Location**: `workflow_analysis/{workflow_name}/`
- **Content**: Workflow stage definitions, dependencies, file patterns

### Expected Outputs

#### SPM Results
```python
{
    "producer:consumer": {
        "SPM": {
            "storage_config": [spm_values]
        },
        "estT_prod": {
            "storage_config": [producer_times]
        },
        "estT_cons": {
            "storage_config": [consumer_times]
        },
        "rank": {
            "storage_config": [rank_values]
        }
    }
}
```

#### Workflow Data
- **CSV**: Processed workflow data with estimated transfer rates
- **JSON**: SPM calculation results and best configurations
- **Text**: Summary reports with recommendations

## 🧪 Testing

### Running the Test Suite

```bash
cd workflow_analysis/python_tests

# Run all tests
python3 test_simple_workflow.py
python3 test_interpolation.py
python3 test_complete_workflow.py
python3 test_notebook_sections.py
python3 test_modular_structure.py
```

### Test Coverage

- ✅ Data loading and validation
- ✅ Configuration management
- ✅ Interpolation and estimation
- ✅ SPM calculations
- ✅ File I/O operations
- ✅ Error handling
- ✅ Module integration

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the correct directory
2. **IOR data not found**: Check that the benchmark data file exists
3. **Missing workflow data**: Verify workflow configuration exists
4. **Zero transfer rates**: Check IOR data contains required storage types

### Getting Help

1. **Check the test suite**: Run tests to verify your setup
2. **Review documentation**: See `workflow_analysis/README.md` for detailed information
3. **Check file paths**: Ensure all data files are in expected locations

## 🤝 Contributing

### Adding New Workflows

1. Add workflow configuration to `TEST_CONFIGS` in `workflow_config.py`
2. Ensure the corresponding script order JSON file exists
3. Update the data path and test folders as needed

### Adding New Storage Types

1. Add the storage type to `STORAGE_LIST` in `workflow_config.py`
2. Update the storage code mapping in `workflow_data_utils.py`
3. Ensure IOR benchmark data exists for the new storage type

### Custom Analysis

The modular structure makes it easy to add custom analysis functions:

```python
# Create new functions in the appropriate module
from workflow_analysis.modules import your_custom_function

# Use them in your analysis
results = your_custom_function(wf_df)
```

## 📄 License

This project is designed for research and educational purposes. Please ensure proper attribution when using or modifying the code.

## 📞 Support

For questions, issues, or contributions:

1. **Check the documentation**: Start with `workflow_analysis/README.md`
2. **Run the test suite**: Verify your setup with the provided tests
3. **Review examples**: Check `example_debug_usage.py` for usage patterns

---

**🎯 Quick Navigation:**
- **Main Analysis**: `workflow_analysis/workflow_analysis.ipynb`
- **Documentation**: `workflow_analysis/README.md`
- **Examples**: `workflow_analysis/example_debug_usage.py`
- **Tests**: `workflow_analysis/python_tests/` 