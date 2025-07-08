# SPM (Storage Performance Matching) Analysis System

A comprehensive system for analyzing scientific workflow performance and optimizing storage configurations using Storage Performance Matching (SPM).

## 🎯 Overview

This project provides tools and analysis capabilities for understanding and optimizing the performance of scientific workflows across different storage systems. It processes datalife statistics, estimates transfer rates using IOR benchmark data, and calculates Storage Performance Metrics (SPM) to recommend optimal storage configurations.

## 📁 Project Structure

```
spm/
├── workflow_analysis/                 # 🎯 Main analysis system
│   ├── modules/                      # Core Python modules
│   │   ├── workflow_config.py        # Configuration management
│   │   ├── workflow_data_utils.py    # Data loading and utilities
│   │   ├── workflow_interpolation.py # Transfer rate estimation
│   │   ├── workflow_spm_calculator.py # SPM calculations
│   │   ├── workflow_visualization.py # Visualization and reporting
│   │   └── workflow_analysis_main.py # Main orchestration
│   ├── python_tests/                 # Comprehensive test suite
│   ├── analysis_data/                # Output directory for results
│   ├── workflow_analysis.ipynb       # Main analysis notebook
│   ├── example_debug_usage.py        # Debug parameter example
│   └── README.md                     # Detailed documentation
├── perf_profiles/                    # Performance benchmark data
│   └── updated_master_ior_df.csv     # IOR benchmark results
├── workflow_data/                    # Workflow datalife statistics
│   ├── ddmd/                        # DDMD workflow data
│   ├── 1kgenome/                    # 1K Genome workflow data
│   └── ...                          # Other workflow data
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
# Navigate to the modules directory
cd workflow_analysis/modules

# Analyze a specific workflow
python workflow_analysis_main.py --workflow ddmd_4n_l

# Analyze all available workflows
python workflow_analysis_main.py --all
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
spm_results = calculate_spm_for_workflow(wf_df, debug=False)
```

## 🔧 Key Features

### 📊 4D Interpolation System
- **Multi-dimensional analysis**: Interpolates transfer rates based on aggregate file size, number of nodes, parallelism, and transfer size
- **Direct parallelism handling**: Uses target parallelism values directly for interpolation/extrapolation
- **Extrapolation support**: Handles values outside the benchmark data range
- **Slope analysis**: Provides transfer size sensitivity information
- **Multi-node support**: Proper handling of tasksPerNode calculations

### 🎯 SPM (Storage Performance Matching) Calculation
- **Producer-consumer analysis**: Calculates metrics for workflow stage transitions
- **Storage optimization**: Ranks storage configurations by performance
- **Stage-aware processing**: Handles stage_in and stage_out operations correctly
- **Debug support**: Optional verbose output for troubleshooting

### 📈 Comprehensive Visualization
- **I/O breakdown plots**: Detailed analysis of read/write operations
- **Storage comparison charts**: Performance across different storage types
- **SPM distribution**: Statistical analysis of performance metrics
- **Workflow stage analysis**: Stage-by-stage performance breakdown

### 🏗️ Modular Architecture
- **Separation of concerns**: Each module handles specific functionality
- **Reusable components**: Functions can be used across different workflows
- **Easy extension**: Simple to add new workflows or storage types
- **Comprehensive testing**: Full test suite for all components

## 📋 Available Workflows

The system supports analysis of multiple scientific workflows:

| Workflow | Description | Data Size | Nodes |
|----------|-------------|-----------|-------|
| `ddmd_2n_s` | DDMD workflow | Small | 2 |
| `ddmd_4n_l` | DDMD workflow | Large | 4 |
| `1kg` | 1K Genome workflow | Standard | Variable |
| `1kg_2` | 1K Genome workflow (alt) | Standard | Variable |
| `pyflex_240f` | PyFlex workflow | 240 files | Variable |
| `pyflex_s9_48f` | PyFlex workflow (S9) | 48 files | Variable |
| `ptychonn` | PtychoNN workflow | Standard | Variable |
| `montage` | Montage workflow | Standard | Variable |
| `seismology` | Seismology workflow | Standard | Variable |
| `llm_wf` | LLM workflow | Standard | Variable |

## 💾 Supported Storage Types

- **`localssd`** - Local SSD storage
- **`beegfs`** - BeeGFS/PFS storage
- **`tmpfs`** - Temporary file system
- **`nfs`** - Network File System
- **`pfs`** - Parallel File System (uses beegfs as proxy)

## ⚙️ Configuration

### Global Configuration Flags

```python
# In workflow_analysis/modules/workflow_config.py
MULTI_NODES = True    # Use tasksPerNode vs parallelism
NORMALIZE = True       # Enable normalization of SPM values
DEBUG = False          # Enable debug output
```

### Debug Parameter

The `calculate_spm_for_workflow()` function supports a `debug` parameter:

```python
# Minimal output (default)
spm_results = calculate_spm_for_workflow(wf_df, debug=False)

# Verbose output for debugging
spm_results = calculate_spm_for_workflow(wf_df, debug=True)
```

## 📊 Output Files

The analysis generates comprehensive output files:

### Data Files
- **`{workflow_name}_workflow_data.csv`** - Processed workflow data with estimated transfer rates
- **`{workflow_name}_spm_results.json`** - SPM calculation results and best configurations

### Visualizations
- **`io_breakdown.png`** - I/O time breakdown visualization
- **`storage_comparison.png`** - Storage performance comparison
- **`spm_distribution.png`** - SPM distribution across configurations
- **`estimated_transfer_rates.png`** - Estimated transfer rates visualization
- **`workflow_stages.png`** - Workflow stage analysis

### Reports
- **`summary_report.txt`** - Comprehensive analysis summary

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
- ✅ Visualization generation
- ✅ File I/O operations
- ✅ Error handling
- ✅ Module integration

## 🔍 Debug and Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the correct directory
2. **IOR data not found**: Check that the benchmark data file exists
3. **Missing workflow data**: Verify workflow configuration exists
4. **Zero transfer rates**: Check IOR data contains required storage types

### Debug Mode

Enable debug output for detailed troubleshooting:

```python
# Set debug=True for verbose output
spm_results = calculate_spm_for_workflow(wf_df, debug=True)
```

### Getting Help

1. **Check the test suite**: Run tests to verify your setup
2. **Review documentation**: See `workflow_analysis/README.md` for detailed information
3. **Check file paths**: Ensure all data files are in expected locations
4. **Enable debug mode**: Set `debug=True` for verbose output

## 🚀 Recent Improvements

### Transfer Rate Estimation Enhancements
- **Fixed parallelism handling**: Uses target parallelism values directly
- **Improved accuracy**: Calculates transfer rates for exact parallelism values
- **Multi-node support**: Proper handling of tasksPerNode calculations
- **Better column naming**: Estimated transfer rate columns reflect actual target values

### SPM (Storage Performance Matching) Calculation Improvements
- **Stage-aware processing**: Correct handling of stage_in and stage_out operations
- **Debug parameter**: Optional verbose output for troubleshooting
- **Enhanced producer-consumer matching**: Improved logic for workflow stage connections

### Data Processing Enhancements
- **Enhanced DataFrame expansion**: Proper expansion for different node configurations
- **Corrected tasksPerNode calculation**: Uses `ceil(parallelism / numNodes)`
- **Improved storage type mapping**: Added support for additional storage types

## 🔮 Future Enhancements

Potential improvements for the system:

- **Parallel processing**: For large datasets
- **Machine learning**: Based transfer rate prediction
- **Real-time analysis**: Capabilities
- **Workflow management**: System integration
- **Interactive visualizations**: Advanced plotting capabilities
- **Cloud storage support**: Additional storage types
- **Performance optimization**: Faster processing for large workflows

## 📚 Dependencies

### Required Python Packages

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
3. **Enable debug mode**: Use `debug=True` for detailed output
4. **Review examples**: Check `example_debug_usage.py` for usage patterns

---

**🎯 Quick Navigation:**
- **Main Analysis**: `workflow_analysis/workflow_analysis.ipynb`
- **Documentation**: `workflow_analysis/README.md`
- **Examples**: `workflow_analysis/example_debug_usage.py`
- **Tests**: `workflow_analysis/python_tests/` 