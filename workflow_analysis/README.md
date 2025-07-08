# Workflow Analysis - Modular Structure

This directory contains a modular implementation of the workflow analysis system, reorganized from the original `workflow_analysis.ipynb` notebook into separate Python modules for better maintainability and reusability.

## Overview

The workflow analysis system processes datalife statistics from scientific workflows, estimates transfer rates using IOR benchmark data, and calculates Storage Performance Matching (SPM) to recommend optimal storage configurations.

## Project Structure

```
workflow_analysis/
├── workflow_analysis_main.py         # 🆕 Main orchestration script (standalone)
├── modules/                          # Core Python modules
│   ├── __init__.py                   # Package initialization
│   ├── README.md                     # Detailed module documentation
│   ├── workflow_config.py            # Configuration management
│   ├── workflow_data_utils.py        # Data loading and utilities
│   ├── workflow_interpolation.py     # Transfer rate estimation
│   ├── workflow_spm_calculator.py    # SPM calculations
│   ├── workflow_visualization.py     # Visualization and reporting
│   ├── workflow_data_staging.py      # Data staging operations
│   ├── workflow_results_exporter.py  # Results export utilities
│   └── __init__.py                   # Module initialization
├── python_tests/                     # Comprehensive test suite
│   ├── README.md                     # Test documentation
│   ├── test_simple_workflow.py       # Basic workflow loading test
│   ├── test_interpolation.py         # Interpolation function test
│   ├── test_complete_workflow.py     # End-to-end pipeline test
│   ├── test_notebook_sections.py     # Complete notebook test
│   └── test_modular_structure.py     # Module architecture test
├── analysis_data/                    # Output directory for results
├── workflow_analysis.ipynb            # 🎯 Main analysis notebook
├── README.md                         # This file
└── TODO.md                           # Development tasks
```

## Module Structure

### Core Modules (in `modules/` directory)

1. **`workflow_config.py`** - Configuration and constants
   - Global configuration flags (MULTI_NODES, NORMALIZE, DEBUG)
   - Storage types and data size mappings
   - Test configurations for different workflows
   - Parameter definitions

2. **`workflow_data_utils.py`** - Data loading and processing utilities
   - Functions to load workflow data from datalife statistics
   - Data transformation and cleaning functions
   - I/O time breakdown calculations
   - Task name assignment and workflow graph construction
   - Multi-node DataFrame expansion with tasksPerNode calculation

3. **`workflow_interpolation.py`** - 4D interpolation functions
   - 4D interpolation for transfer rate estimation using target parallelism directly
   - Functions to estimate transfer rates for different storage configurations
   - Aggregate file size calculations
   - Direct interpolation/extrapolation for exact parallelism values

4. **`workflow_spm_calculator.py`** - SPM (Storage Performance Matching) calculation and analysis
   - Workflow graph construction using NetworkX
   - SPM calculation for producer-consumer pairs
   - Storage option filtering and ranking
   - Best configuration selection

5. **`workflow_visualization.py`** - Visualization functions
   - I/O time breakdown plots
   - Storage performance comparison charts
   - SPM distribution visualizations
   - Workflow stage analysis plots
   - Summary report generation

6. **`workflow_data_staging.py`** - Data staging operations
   - Functions to stage workflow data for analysis

7. **`workflow_results_exporter.py`** - Results export utilities
   - Functions to export analysis results to various formats

### Test Suite (in `python_tests/` directory)

- **`test_simple_workflow.py`** - Basic workflow data loading test
- **`test_interpolation.py`** - 4D interpolation and transfer rate estimation test
- **`test_complete_workflow.py`** - End-to-end workflow analysis test
- **`test_notebook_sections.py`** - Comprehensive test of all notebook sections
- **`test_modular_structure.py`** - Module architecture and import testing

### Notebooks

- **`workflow_analysis.ipynb`** - 🎯 **Main analysis notebook** (your primary tool)

## Usage

### 1. Main Analysis Script

The main entry point for workflow analysis is now:

```
python3 workflow_analysis/workflow_analysis_main.py --help
```

This script orchestrates the full workflow analysis pipeline, including loading workflow data, estimating transfer rates, calculating SPM, filtering storage options, selecting best storage/parallelism, and displaying results. See command-line options with `--help`.

### 🎯 Option 1: Using the Main Notebook (Recommended)

1. Open `workflow_analysis.ipynb` in Jupyter
2. Modify the `WORKFLOW_NAME` variable to analyze different workflows
3. Run all cells sequentially
4. View results in the notebook and generated files in `analysis_data/`

### Option 2: Using the Command-Line Script

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

### Option 3: Using the Main Script Programmatically

```python
# Import the main script directly
import sys
sys.path.append('workflow_analysis')
from workflow_analysis_main import run_workflow_analysis

# Run analysis for a specific workflow
results = run_workflow_analysis(
    workflow_name="ddmd_4n_l",
    ior_data_path="../perf_profiles/updated_master_ior_df.csv",
    save_results=True
)

# Access results
workflow_df = results['workflow_df']
spm_results = results['spm_results']
best_results = results['best_results']
```

### Option 4: Using Individual Modules

```python
# Import specific modules as needed
from modules import (
    load_workflow_data,
    estimate_transfer_rates_for_workflow,
    calculate_spm_for_workflow,
    plot_all_visualizations
)

# Load workflow data
wf_df, task_order, wf_dict = load_workflow_data("ddmd_4n_l")

# Estimate transfer rates
wf_df = estimate_transfer_rates_for_workflow(wf_df, ior_data, storage_list)

# Calculate SPM values
spm_results = calculate_spm_for_workflow(wf_df)

# Generate visualizations
plot_all_visualizations(wf_df, spm_results, io_breakdown)
```

## Available Workflows

The system supports analysis of multiple workflows:

- `ddmd_2n_s` - DDMD workflow (small data, 2 nodes)
- `ddmd_4n_l` - DDMD workflow (large data, 4 nodes)
- `1kg` - 1K Genome workflow
- `1kg_2` - 1K Genome workflow (alternative)
- `pyflex_240f` - PyFlex workflow (240 files)
- `pyflex_s9_48f` - PyFlex workflow (S9, 48 files)
- `ptychonn` - PtychoNN workflow
- `montage` - Montage workflow
- `seismology` - Seismology workflow
- `llm_wf` - LLM workflow

## Configuration

### Global Configuration Flags

- `MULTI_NODES` - Use tasksPerNode vs parallelism
- `NORMALIZE` - Enable normalization of SPM values
- `DEBUG` - Enable debug output

### Debug Parameter

The `calculate_spm_for_workflow()` function now supports a `debug` parameter to control verbose output:

```python
# Minimal output (default)
spm_results = calculate_spm_for_workflow(wf_df, debug=False)

# Verbose output for debugging
spm_results = calculate_spm_for_workflow(wf_df, debug=True)
```

**Debug Output Includes:**
- Stage order list and node details
- Producer-consumer pair processing
- Edge creation and attribute details
- Missing pair diagnostics
- Processing statistics

**Usage:**
- `debug=False` (default): Clean output, suitable for production use
- `debug=True`: Verbose output, useful for debugging and development

### Storage Types

- `localssd` - Local SSD storage
- `beegfs` - BeeGFS/PFS storage
- `tmpfs` - Temporary file system
- `nfs` - Network File System
- `pfs` - Parallel File System (uses beegfs as proxy)

## Output Files

The analysis generates several output files in the `analysis_data/` directory:

1. **`{workflow_name}_workflow_data.csv`** - Processed workflow data with estimated transfer rates
2. **`{workflow_name}_spm_results.json`** - SPM calculation results and best configurations
3. **`io_breakdown.png`** - I/O time breakdown visualization
4. **`storage_comparison.png`** - Storage performance comparison
5. **`spm_distribution.png`** - SPM distribution across configurations
6. **`estimated_transfer_rates.png`** - Estimated transfer rates visualization
7. **`workflow_stages.png`** - Workflow stage analysis
8. **`summary_report.txt`** - Comprehensive analysis summary

## Key Features

### 4D Interpolation
- Interpolates transfer rates based on aggregate file size, number of nodes, parallelism, and transfer size
- Uses target parallelism values directly for interpolation/extrapolation (not closest available)
- Handles extrapolation for values outside the benchmark data range
- Provides slope information for transfer size sensitivity
- Supports multi-node configurations with tasksPerNode calculation

### SPM (Storage Performance Matching) Calculation
- Calculates Storage Performance Matching for producer-consumer pairs
- Considers both producer and consumer I/O characteristics
- Ranks storage configurations by performance

### Visualization
- Comprehensive plotting of analysis results
- Multiple chart types for different aspects of the analysis
- Automatic saving of plots with high resolution

### Modular Design
- Separated concerns for better maintainability
- Reusable functions across different workflows
- Easy to extend and modify

## Dependencies

Required Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- networkx
- scikit-learn
- scipy

## Testing

### Running the Test Suite

The project includes a comprehensive test suite in the `python_tests/` directory:

```bash
# Navigate to the test directory
cd python_tests

# Run all tests in sequence
python3 test_simple_workflow.py
python3 test_interpolation.py
python3 test_complete_workflow.py
python3 test_notebook_sections.py
python3 test_modular_structure.py

# Or run a specific test
python3 test_[test_name].py
```

### Test Coverage

The test suite covers:
- ✅ Data loading and validation
- ✅ Configuration management
- ✅ Interpolation and estimation
- ✅ SPM calculations
- ✅ Visualization generation
- ✅ File I/O operations
- ✅ Error handling
- ✅ Module integration

For detailed test documentation, see `python_tests/README.md`.

## Troubleshooting

### Common Issues

1. **Import errors**: 
   ```
   ModuleNotFoundError: No module named 'modules'
   ```
   **Solution**: Ensure you're importing from the correct path or running from the right directory

2. **IOR data not found**: 
   ```
   FileNotFoundError: [Errno 2] No such file or directory: '../../perf_profiles/updated_master_ior_df.csv'
   ```
   **Solution**: Ensure the IOR benchmark data file exists at the specified path

3. **Missing workflow data**: Check that the workflow configuration exists in `TEST_CONFIGS`
4. **Zero transfer rates**: Check that IOR data contains the required storage types and operation types
5. **Missing parallelism values**: Ensure IOR data covers the parallelism range needed for your workflow

### Debug Mode

Enable debug output by setting `DEBUG = True` in `modules/workflow_config.py` to get more detailed information during analysis.

### Getting Help

1. **Check the test suite**: Run the tests to verify your setup
2. **Review module documentation**: See `modules/README.md` for detailed module information
3. **Check file paths**: Ensure all data files are in the expected locations
4. **Enable debug mode**: Set `DEBUG = True` for verbose output

## Extending the System

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
1. Create new functions in the appropriate module
2. Import and use them in the main analysis script
3. Add visualization functions if needed

## Performance Notes

- The 4D interpolation can be computationally intensive for large datasets
- Consider using smaller subsets of IOR data for faster analysis during development
- The SPM calculation scales with the number of producer-consumer pairs
- Multi-node expansion increases the dataset size by the number of node configurations
- Transfer rate estimation now uses target parallelism directly, improving accuracy

## Recent Improvements

### Transfer Rate Estimation Enhancements (Latest)
- **Fixed parallelism handling**: Now uses target parallelism values directly instead of finding closest available
- **Improved accuracy**: Transfer rates are calculated for exact parallelism values (1, 3, 6, 12) as specified in workflow configuration
- **Multi-node support**: Added `expand_df` function to properly handle multi-node configurations with correct `tasksPerNode` calculation
- **Better column naming**: Estimated transfer rate columns now reflect the actual target parallelism values

### Data Processing Improvements
- **Enhanced DataFrame expansion**: Proper expansion for different node configurations (1, 2, 4 nodes)
- **Corrected tasksPerNode calculation**: Uses `ceil(parallelism / numNodes)` as in the original notebook
- **Improved storage type mapping**: Added support for `pfs` storage type with beegfs proxy

## Future Enhancements

Potential improvements for the system:
- Parallel processing for large datasets
- Machine learning-based transfer rate prediction
- Real-time analysis capabilities
- Integration with workflow management systems
- Advanced visualization with interactive plots 