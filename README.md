# SPM (Storage Performance Matching) Analysis System

A comprehensive system for analyzing scientific workflow performance and optimizing storage configurations using Storage Performance Matching (SPM).

## 🎯 Overview

This project provides tools and analysis capabilities for understanding and optimizing the performance of scientific workflows across different storage systems. It processes workflow I/O pattern profiles, estimates transfer rates using IOR benchmark data, and calculates Storage Performance Matching (SPM) to recommend optimal storage configurations.

**Main User Interface**: The `workflow_analysis/` directory contains the primary analysis tools that users will interact with.

## 🚀 Quick Start

### Option 1: Using the Split Architecture (Recommended)
The workflow analysis is now split into two phases for better modularity and reusability:

**Phase 1: Data Loading**
```bash
cd workflow_analysis

# Load workflow data from JSON files and save to CSV
python3 workflow_data_loader.py --workflow ddmd_4n_l

# With custom output directory and filename
python3 workflow_data_loader.py --workflow ddmd_4n_l --output-dir ./my_data --csv-filename my_workflow.csv
```

**Phase 2: Analysis**
```bash
# Analyze workflow from CSV file (workflow name auto-extracted from filename)
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv

# Specify workflow name explicitly
python3 workflow_analyzer.py analysis_data/workflow_data.csv --workflow ddmd_4n_l

# Use different IOR data path
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv --ior-data path/to/ior_data.csv

# Don't save results to files
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv --no-save
```

### Option 2: Complete Workflow (Both Phases)
```bash
# Step 1: Load data
python3 workflow_data_loader.py --workflow ddmd_4n_l

# Step 2: Analyze the loaded data
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv
```

### Option 3: Using the Jupyter Notebook (For Debugging)
```bash
cd workflow_analysis
jupyter notebook workflow_analysis.ipynb
```

### Option 4: Using Individual Modules
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

## 🐍 Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd /path/to/spm
   ```

2. **Create a Python virtual environment:**
   ```bash
   python3 -m venv spm_env
   ```

3. **Activate the virtual environment:**
   ```bash
   source spm_env/bin/activate  # On Linux/macOS
   # or
   spm_env\Scripts\activate     # On Windows
   ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation:**
   ```bash
   cd workflow_analysis
   python3 workflow_data_loader.py --help
   python3 workflow_analyzer.py --help
   ```

### Required Packages
The following packages are automatically installed via `requirements.txt`:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical visualizations
- **networkx** - Graph construction and analysis
- **scipy** - Scientific computing and interpolation

### Deactivating the Environment
When you're done working with the project:
```bash
deactivate
```

## 📁 Project Structure

```
spm/
├── workflow_analysis/                 # Main analysis system
│   ├── workflow_data_loader.py       # Phase 1: Data loading from JSON to CSV
│   ├── workflow_analyzer.py          # Phase 2: Analysis from CSV
│   ├── workflow_analysis.ipynb       # Analysis notebook
│   ├── modules/                      # Core modules
│   ├── analysis_data/                # Generated data files
│   ├── workflow_spm_results/         # Analysis results
│   ├── template_workflow/            # Template for testing
│   └── python_tests/                 # Test suite
└── perf_profiles/                    # Benchmark data
    └── updated_master_ior_df.csv     # IOR benchmark results
```

## 📋 Available Workflows

| Workflow | Description | Data Size | Nodes |
|----------|-------------|-----------|-------|
| `ddmd_4n_l` | DDMD workflow | Large | 4 |
| `1kg` | 1K Genome workflow | Standard | Variable |
| `pyflex_s9_48f` | PyFlex workflow | S9, 48 files | Variable |
| `template_workflow` | Template workflow for testing | Artificial | 4 |

## 💾 Supported Storage Types

The system supports storage types based on I/O performance profiles collected from benchmark data:

- **`localssd`** - Local SSD storage (high bandwidth, low latency)
- **`beegfs`** - BeeGFS/PFS storage (distributed parallel file system)
- **`tmpfs`** - Temporary file system (memory-based, fastest access)
- **`nfs`** - Network File System (network-attached storage)

Performance characteristics are derived from IOR benchmark data in `perf_profiles/updated_master_ior_df.csv`, which contains transfer rates for different file sizes, parallelism levels, and I/O patterns (read/write operations, cp/scp operations).

### Storage Type Transitions
The system now supports storage type transitions using cp/scp operations:
- **`beegfs-ssd`**, **`ssd-beegfs`** - Transitions between BeeGFS and SSD storage
- **`beegfs-tmpfs`**, **`tmpfs-beegfs`** - Transitions between BeeGFS and tmpfs storage
- **`ssd-ssd`**, **`tmpfs-tmpfs`** - Same-storage operations using cp/scp

## 🔧 Key Features

### 📊 4D Interpolation System
- Multi-dimensional analysis based on aggregate file size, nodes, parallelism, and transfer size
- Storage and parallelism optimization recommendations
- Extrapolation support for values outside benchmark range
- Multi-node support with tasksPerNode calculations

### 🎯 SPM (Storage Performance Matching) Calculation
- Producer-consumer analysis for workflow stage transitions
- Storage configuration ranking by performance
- Stage-aware processing for stage_in and stage_out operations
- **Intermediate Results**: Detailed per-task time estimation (without SPM values) stored in CSV format

### 🏗️ Modular Architecture
- **Split Architecture**: Data loading and analysis phases separated for better modularity
- **Input File Preservation**: Original CSV files are never modified, enabling multiple analysis runs
- **Reusable Components**: CSV files can be analyzed multiple times without reloading JSON data
- **Separation of Concerns**: Dedicated modules for different analysis phases
- **Easy Extension**: New workflows or storage types can be easily added
- **Comprehensive Test Suite**: Thorough testing of all components

## 📊 Workflow Data Structure

### Required Directory Structure
```
workflow_analysis/
├── {workflow_name}/
│   ├── {workflow_name}_script_order.json    # Required: Workflow configuration
│   └── {workflow_config_run}/
│       └── run_trial1/
│           └── {datalife_trace_files}      # Workflow execution traces
```

### Script Order JSON File Structure
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

## 📊 Operation Code Handling

**Workflow Data**: Uses integer operations (0=write, 1=read)
**IOR Benchmark Data**: Uses string operations ('write', 'read', 'cp', 'scp')

The interpolation function automatically maps workflow integers to IOR strings internally. The system now supports cp/scp operations for modeling storage type transitions between workflow stages.

**Note**: All operations in the benchmark data CSV files are now consistently represented as strings for better clarity and consistency.

## 📊 Output Files

### Data Files
- **`analysis_data/{workflow_name}_original_workflow_data.csv`** - Original data before modifications
- **`analysis_data/{workflow_name}_processed_workflow_data.csv`** - Modified data with staging rows and transfer rates
- **`analysis_data/{workflow_name}_spm_results.json`** - SPM calculation results and best configurations

### Analysis Results
- **`workflow_spm_results/{workflow_name}_filtered_spm_results.csv`** - Filtered SPM results with storage analysis
- **`workflow_spm_results/{workflow_name}_intermediate_estT_results.csv`** - Intermediate DAG edges with per-task time estimation (without SPM values)

### Reports
- **`{workflow_name}_spm.txt`** - Top-ranked storage configurations
- **`{workflow_name}_io_breakdown.txt`** - I/O time breakdown analysis

## 🚧 Development Status

- ✅ **Core Analysis**: Complete and tested
- ✅ **SPM Calculation**: Complete and tested
- ✅ **Transfer Rate Estimation**: Complete and tested
- ✅ **Storage Type Transitions**: Complete and tested (CP/SCP operations)
- ✅ **Split Architecture**: Complete and tested (Data loading + Analysis phases)
- ✅ **Input File Preservation**: Complete and tested (Original files never modified)
- ✅ **Intermediate Results**: Complete and tested (Per-task time estimation CSV)
- 🚧 **Visualization**: Under construction
- ✅ **Template Generation**: Complete and tested
- ✅ **Command-Line Interface**: Complete and tested

## 🏗️ Split Architecture Benefits

The workflow analysis system has been redesigned with a split architecture for better modularity and reusability:

### **Phase 1: Data Loading (`workflow_data_loader.py`)**
- **Purpose**: Loads JSON files and processes them into CSV format
- **Input**: JSON datalife trace files from workflow execution
- **Output**: Clean CSV file ready for analysis
- **Benefits**: 
  - Separates data loading from analysis
  - CSV files can be manually inspected and modified
  - Data loading only needs to be done once per workflow

### **Phase 2: Analysis (`workflow_analyzer.py`)**
- **Purpose**: Performs complete workflow analysis on CSV data
- **Input**: CSV file from Phase 1
- **Output**: SPM results, filtered configurations, and intermediate analysis
- **Benefits**:
  - Original CSV file is never modified (creates working copy)
  - Can be run multiple times on the same input
  - Supports different analysis parameters without reloading data
  - Generates intermediate results for detailed inspection

### **Key Advantages**
- **Reproducibility**: Multiple analysis runs with identical results
- **Flexibility**: Modify CSV data between analysis runs
- **Debugging**: Easier to isolate issues in data loading vs. analysis
- **Performance**: Avoid reloading JSON files for repeated analysis
- **Data Integrity**: Original input files are preserved

## 📚 Documentation

- **Workflow Analysis**: `workflow_analysis/README.md` - Main analysis system documentation
- **Template Workflow**: `