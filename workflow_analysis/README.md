# Workflow Analysis System

A comprehensive system for analyzing scientific workflow performance and optimizing storage configurations using Storage Performance Matching (SPM).

## 🎯 Overview

The workflow analysis system processes datalife statistics from scientific workflows, estimates transfer rates using IOR benchmark data, and calculates Storage Performance Matching (SPM) to recommend optimal storage configurations.

## 🚀 Quick Start

### Option 1: Using the Command-Line Script (Recommended)
```bash
cd workflow_analysis

# Analyze a specific workflow
python3 workflow_analysis_main.py --workflow ddmd_4n_l

# Analyze all available workflows
python3 workflow_analysis_main.py --all

# With custom CSV filename
python3 workflow_analysis_main.py --workflow template_workflow --csv-filename my_workflow.csv
```

### Option 2: Using the Main Notebook (For Debugging)
```bash
cd workflow_analysis
jupyter notebook workflow_analysis.ipynb
```

### Option 3: Using Individual Modules
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
├── workflow_analysis_main.py         # Main orchestration script
├── workflow_analysis.ipynb           # Analysis notebook (debugging)
├── modules/                          # Core modules
│   ├── workflow_config.py            # Configuration
│   ├── workflow_data_utils.py        # Data loading
│   ├── workflow_interpolation.py     # Transfer rate estimation
│   ├── workflow_spm_calculator.py    # SPM calculations
│   ├── workflow_visualization.py     # Visualization (under construction)
│   └── workflow_template_generator.py # Template generator
├── template_workflow/                 # Template for testing
├── python_tests/                     # Test suite
└── analysis_data/                    # Output results
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