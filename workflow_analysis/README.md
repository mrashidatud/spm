# Workflow Analysis System

A comprehensive system for analyzing scientific workflow performance and optimizing storage configurations using Storage Performance Matching (SPM).

## 🎯 Overview

The workflow analysis system processes datalife statistics from scientific workflows, estimates transfer rates using IOR benchmark data, and calculates Storage Performance Matching (SPM) to recommend optimal storage configurations.

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
from modules import (
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

## 📁 Project Structure

```
workflow_analysis/
├── workflow_data_loader.py           # Phase 1: Data loading from JSON to CSV
├── workflow_analyzer.py              # Phase 2: Analysis from CSV
├── workflow_analysis.ipynb           # Analysis notebook (debugging)
├── modules/                          # Core analysis modules
│   ├── workflow_config.py            # Configuration
│   ├── workflow_data_utils.py        # Data loading utilities
│   ├── workflow_interpolation.py     # Transfer rate estimation
│   ├── workflow_spm_calculator.py    # SPM calculations
│   ├── workflow_data_staging.py      # Data staging operations
│   ├── workflow_results_exporter.py  # Results export utilities
│   ├── workflow_visualization.py     # Visualization (under construction)
│   └── workflow_template_generator.py # Template generator
├── template_workflow/                 # Template for testing
├── python_tests/                     # Test suite
├── analysis_data/                    # Generated data files
└── workflow_spm_results/             # Analysis results
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

Performance characteristics are derived from IOR benchmark data in `../perf_profiles/updated_master_ior_df.csv`, which contains transfer rates for different file sizes, parallelism levels, and I/O patterns (read/write operations, cp/scp operations).

### Storage Type Transitions
The system now supports storage type transitions using cp/scp operations:
- **`beegfs-ssd`**, **`ssd-beegfs`** - Transitions between BeeGFS and SSD storage
- **`beegfs-tmpfs`**, **`tmpfs-beegfs`** - Transitions between BeeGFS and tmpfs storage
- **`ssd-ssd`**, **`tmpfs-tmpfs`** - Same-storage operations using cp/scp

## 🔧 Key Features

### 📊 Stage Numbering
All workflows use 1-based stage numbering:
- **First stage**: Always stage 1 (not stage 0)
- **Subsequent stages**: Stage 2, 3, 4, etc.
- **Automatic normalization**: Legacy workflows with stage 0 are automatically normalized

### 📊 4D Interpolation System
- Multi-dimensional analysis based on aggregate file size, nodes, parallelism, and transfer size
- Storage and parallelism optimization recommendations
- Extrapolation support for values outside benchmark range
- Multi-node support with tasksPerNode calculations

### 🎯 SPM (Storage Performance Matching) Calculation
- Producer-consumer analysis for workflow stage transitions
- Storage configuration ranking by performance
- Stage-aware processing for stage_in and stage_out operations

### 🏗️ Modular Architecture
- Separation of concerns with dedicated modules
- Reusable components across different workflows
- Easy extension for new workflows or storage types
- Comprehensive test suite

## 📊 Operation Code Handling

**Workflow Data**: Uses integer operations (0=write, 1=read)
**IOR Benchmark Data**: Uses string operations ('write', 'read', 'cp', 'scp')

The interpolation function automatically maps workflow integers to IOR strings internally. The system now supports cp/scp operations for modeling storage type transitions between workflow stages.

**Note**: All operations in the benchmark data CSV files are now consistently represented as strings for better clarity and consistency.

## 📊 Output Files

### Data Files
- **`{workflow_name}_workflow_data.csv`** - Processed workflow data with estimated transfer rates
- **`{workflow_name}_spm_results.json`** - SPM calculation results and best configurations

### Reports
- **`{workflow_name}_spm.txt`** - Top-ranked storage configurations
- **`{workflow_name}_io_breakdown.txt`** - I/O time breakdown analysis

## 📊 Intermediate Results Format

The system generates detailed intermediate results that provide insights into the workflow analysis process. These files contain comprehensive information about the workflow graph structure, transfer rate calculations, and SPM computations.

### Intermediate Results Files

#### 1. **`{workflow_name}_WFG.json`**
This file contains the complete workflow graph structure in JSON format, including all nodes, edges, and their calculated transfer time estimates.

**File Location**: `workflow_spm_results/{workflow_name}_WFG.json`

**Structure**:
```json
{
    "nodes": {
        "node_identifier": {
            "taskName": "task_name",
            "operation": "read|write|cp|scp|none",
            "parallelism": 24,
            "aggregateFilesizeMB": 1024.5,
            "storageType": "beegfs|ssd|tmpfs|nfs",
            "stageOrder": 1.0,
            "numNodes": 4,
            "tasksPerNode": 6,
            "fileName": "file1.h5,file2.h5",
            "estimated_trMiB_beegfs_24p": 45.2,
            "estimated_trMiB_ssd_12p": 67.8,
            // ... other transfer rate estimates
        }
    },
    "edges": [
        {
            "producer_node": "task1:pid1:file1.h5",
            "consumer_node": "task2:pid2:file2.h5",
            "attributes": {
                "estT_prod_beegfs_24p": 0.045,
                "estT_cons_ssd_12p": 0.023,
                "prod_aggregateFilesizeMB": 1024.5,
                "cons_aggregateFilesizeMB": 512.0,
                "prod_max_parallelism": 24,
                "cons_max_parallelism": 12,
                "prod_stage_order": 1.0,
                "cons_stage_order": 1.5,
                "prod_task_name": "task1",
                "cons_task_name": "task2",
                "file_name": "file1.h5,file2.h5"
            }
        }
    ]
}
```

**Key Parameters in Edge Attributes**:
- **`estT_prod_{storage}_{parallelism}p`**: Producer transfer time in seconds for specific storage and parallelism
- **`estT_cons_{storage}_{parallelism}p`**: Consumer transfer time in seconds for specific storage and parallelism
- **`prod_aggregateFilesizeMB`**: Producer aggregate file size in MB
- **`cons_aggregateFilesizeMB`**: Consumer aggregate file size in MB
- **`prod_max_parallelism`**: Maximum parallelism for producer
- **`cons_max_parallelism`**: Maximum parallelism for consumer
- **`prod_stage_order`**: Producer stage order
- **`cons_stage_order`**: Consumer stage order
- **`prod_task_name`**: Producer task name
- **`cons_task_name`**: Consumer task name
- **`file_name`**: File name(s) associated with this edge

**Key Format Examples**:
- `estT_prod_beegfs_24p`: Producer transfer time for beegfs storage with 24 parallel tasks
- `estT_cons_ssd_12p`: Consumer transfer time for ssd storage with 12 parallel tasks
- `estT_prod_tmpfs_8p`: Producer transfer time for tmpfs storage with 8 parallel tasks



### Understanding the Parameters

#### **Transfer Time Calculations**
- **`estT_prod_value`**: Time in seconds for the producer to complete its I/O operation
- **`estT_cons_value`**: Time in seconds for the consumer to complete its I/O operation
- **Calculation**: `estT = aggregateFilesizeMB / estimated_transfer_rate_MiB_per_sec`

#### **Stage Order System**
- **Integer stages** (1.0, 2.0, 3.0): Main workflow stages
- **Fractional stages** (1.5, 2.5): Data staging operations (stage_in, stage_out)
- **Stage_in**: `stageOrder - 0.5` (before the main task)
- **Stage_out**: `stageOrder + 0.5` (after the main task)

#### **Storage Type Combinations**
- **Single storage**: `beegfs`, `ssd`, `tmpfs`, `nfs`
- **Transition storage**: `beegfs-ssd`, `ssd-beegfs`, `beegfs-tmpfs`, `tmpfs-beegfs`
- **Same-storage operations**: `ssd-ssd`, `tmpfs-tmpfs`

#### **Parallelism Levels**
- **Common values**: 1p, 4p, 8p, 12p, 16p, 24p, 32p, 48p, 64p
- **Maximum**: 60 files per staging operation (configurable)
- **Calculation**: Based on number of files and system capabilities

### Data Flow Analysis

#### **Producer-Consumer Relationships**
1. **Regular tasks**: `task1` → `task2` (stage 1 → stage 2)
2. **Stage_in operations**: `stage_in-task1` → `task1` (stage 0.5 → stage 1)
3. **Stage_out operations**: `task1` → `stage_out-task1` (stage 1 → stage 1.5)

#### **Transfer Time Patterns**
- **Virtual producers** (`operation: "none"`): `estT_prod_value = 0.0`
- **Regular operations**: Calculated based on file size and transfer rate
- **Staging operations**: Use cp/scp transfer rates for cross-storage movements

### Using Intermediate Results

#### **For Debugging**
- Examine specific edge calculations and transfer time estimates
- Verify transfer rate estimations for different storage configurations
- Check stage ordering and task relationships in the workflow graph

#### **For Analysis**
- Identify bottlenecks in the workflow by analyzing transfer times
- Compare different storage configurations and their performance
- Analyze the impact of parallelism changes on workflow performance

#### **For Custom Processing**
- Load JSON data for custom analysis and visualization
- Extract specific storage/parallelism combinations for detailed study
- Parse node and edge data for statistical analysis and reporting

### Example Analysis Queries

For detailed code examples on how to analyze the intermediate results, see:

- **JSON Data Loading**: [`workflow_spm_calculator.py:1010-1067`](modules/workflow_spm_calculator.py#L1010-L1067) - JSON serialization logic
- **Edge Analysis**: [`workflow_spm_calculator.py:1010-1067`](modules/workflow_spm_calculator.py#L1010-L1067) - Edge attribute extraction
- **Node Analysis**: [`workflow_spm_calculator.py:1010-1067`](modules/workflow_spm_calculator.py#L1010-L1067) - Node data structure

Basic usage pattern:
```python
import json
import pandas as pd

# Load intermediate results
with open('workflow_spm_results/{workflow_name}_WFG.json', 'r') as f:
    wfg_data = json.load(f)

# Access nodes and edges
nodes = wfg_data['nodes']
edges = wfg_data['edges']
```

## 🔧 How to Use the Scripts

### Using `workflow_data_loader.py`

The `workflow_data_loader.py` script loads workflow data from JSON files and converts it to CSV format for analysis.

**Main Function**: [`workflow_data_loader.py:82`](workflow_data_loader.py#L82) - Command-line interface

#### Basic Usage
```bash
# Load workflow data from JSON files and save to CSV
python3 workflow_data_loader.py --workflow ddmd_4n_l

# With custom output directory and filename
python3 workflow_data_loader.py --workflow ddmd_4n_l --output-dir ./my_data --csv-filename my_workflow.csv
```

#### Command Line Arguments
| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--workflow, -w` | Workflow name to load | - | Yes |
| `--output-dir, -o` | Output directory for CSV files | `./analysis_data` | No |
| `--csv-filename, -f` | Custom CSV filename | `{workflow_name}_workflow_data.csv` | No |
| `--help, -h` | Show help message | - | No |

### Using `workflow_analyzer.py`

The `workflow_analyzer.py` script performs complete workflow analysis on CSV data, including transfer rate estimation, SPM calculations, and storage optimization.

**Main Function**: [`workflow_analyzer.py:29`](workflow_analyzer.py#L29) - Complete analysis pipeline

#### Basic Usage
```bash
# Analyze workflow from CSV file (workflow name auto-extracted from filename)
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv

# Specify workflow name explicitly
python3 workflow_analyzer.py analysis_data/workflow_data.csv --workflow ddmd_4n_l

# Use different IOR data path
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv --ior-data path/to/ior_data.csv

# Run analysis without saving results to files (for testing)
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv --no-save
```

#### Command Line Arguments
| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `csv_file` | Path to the CSV file containing workflow data | - | Yes |
| `--workflow, -w` | Workflow name (auto-extracted from filename if not provided) | None | No |
| `--ior-data, -i` | Path to IOR benchmark data CSV file | `../perf_profiles/updated_master_ior_df.csv` | No |
| `--no-save` | Do not save results to files (for testing/debugging) | False | No |
| `--help, -h` | Show help message | - | No |

#### Analysis Steps
See [`workflow_analyzer.py:29`](workflow_analyzer.py#L29) for the complete analysis pipeline:

1. **Data Loading**: Loads workflow data from CSV file and creates a working copy
2. **I/O Time Breakdown**: Calculates I/O time breakdown for each task
3. **Data Staging**: Inserts data staging rows for workflow transitions
4. **Aggregate File Size**: Calculates aggregate file size per node
5. **Transfer Rate Estimation**: Estimates transfer rates using IOR benchmark data
6. **SPM Calculation**: Calculates Storage Performance Matching values
7. **Storage Filtering**: Filters storage options based on workflow configuration
8. **Best Configuration Selection**: Identifies optimal storage and parallelism settings
9. **Results Export**: Saves analysis results to various output files

### Advanced Usage Examples

#### Custom IOR Data Analysis
```bash
# Use a different IOR benchmark dataset
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv \
    --ior-data /path/to/custom_ior_benchmarks.csv
```

#### Testing Without File Output
```bash
# Run analysis for testing without generating output files
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv --no-save
```

#### Workflow Name Extraction
```bash
# Auto-extract from filename pattern: {workflow_name}_workflow_data*.csv
python3 workflow_analyzer.py analysis_data/ddmd_4n_l_workflow_data.csv

# Or specify explicitly
python3 workflow_analyzer.py analysis_data/generic_workflow.csv --workflow ddmd_4n_l
```

### Troubleshooting

#### Common Issues

1. **CSV file not found**: Ensure the CSV file exists and the path is correct
2. **Workflow not in configuration**: The script will still work but may skip some filtering steps
3. **IOR data file not found**: Provide correct path to IOR benchmark data or skip transfer rate estimation

#### Debug Mode
For detailed debugging, see [`workflow_analyzer.py:29`](workflow_analyzer.py#L29) and set `debug=True` in relevant function calls.

### Integration with Other Tools
The scripts are designed to work seamlessly with:
- **Jupyter notebooks**: Results can be loaded for further analysis
- **Custom scripts**: JSON results can be parsed for custom processing
- **Visualization tools**: Output data can be used for plotting and charts

See [`workflow_spm_calculator.py:1010-1067`](modules/workflow_spm_calculator.py#L1010-L1067) for JSON data structure details.

## 🚧 Development Status

- ✅ **Core Analysis**: Complete and tested
- ✅ **SPM Calculation**: Complete and tested  
- ✅ **Transfer Rate Estimation**: Complete and tested
- ✅ **Storage Type Transitions**: Complete and tested (CP/SCP operations)
- 🚧 **Visualization**: Under construction
- ✅ **Template Generation**: Complete and tested
- ✅ **Command-Line Interface**: Complete and tested

## 📚 Related Documentation

- **Template Workflow**: `template_workflow/README.md` - Template workflow documentation
- **Module Documentation**: `modules/README.md` - Detailed module descriptions
- **Test Documentation**: `python_tests/README.md` - Test suite documentation
- **Root Documentation**: `../README.md` - Project overview

---

The workflow analysis system provides comprehensive tools for optimizing scientific workflow performance across different storage configurations. 