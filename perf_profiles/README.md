# IOR Benchmark Analysis Tools

This directory contains tools for collecting, analyzing, and visualizing IOR benchmark data from different storage systems, including both IOR and cp (copy) benchmark data.

## Overview

The tools are designed to process IOR benchmark JSON files and cp benchmark data, creating a master DataFrame with the following key parameters:

- **operation**: write or read operation
- **randomOffset**: 0 (all benchmarks are sequential)
- **transferSize**: transfer size in bytes
- **aggregateFilesizeMB**: total file size in MB
- **numTasks**: number of tasks used
- **totalTime**: total time for the operation
- **numNodes**: number of nodes used
- **tasksPerNode**: tasks per node
- **parallelism**: total parallelism (numTasks * numNodes)
- **trMiB**: throughput in MiB/s
- **storageType**: storage system type (beegfs, localssd, nfs, tmpfs, or cp types like beegfs-ssd, ssd-ssd, etc.)

## Files

- `ior_utils.py`: Main utility functions for data collection and analysis
- `ior_analysis_new.ipynb`: Jupyter notebook for IOR visualization and analysis
- `move_data_bench_analysis.ipynb`: Jupyter notebook for cp benchmark analysis
- `test_ior_utils.py`: Test script to verify IOR functionality
- `test_averaged_statistics.py`: Test script for averaged statistics functionality
- `concat_csv_files.py`: Script to concatenate IOR and cp data with averaging
- `add_merge_cell.py`: Script to add merge functionality to notebooks
- `clean_csv_files.py`: Script to clean existing CSV files
- `README.md`: This documentation file
- `plot/`: Directory containing all generated plots (created automatically)

## Data Structure

### IOR Benchmark Data
The IOR benchmark data is organized in the following directory structure:

```
ior_data/
├── beegfs_pior_buffered_1n/
│   ├── pior_1k_1gb_n2_1.json
│   ├── pior_64m_1gb_n8_1.json
│   └── ...
├── beegfs_pior_buffered_2n/
│   └── ...
├── localssd_pior_buffered_1n/
│   └── ...
├── nfs_pior_buffered_1n/
│   └── ...
└── tmpfs_pior_buffered_1n/
    └── ...
```

### CP Benchmark Data
The cp benchmark data is stored in:
```
cp_data/
├── beegfs_cp_1n/
├── localssd_cp_1n/
├── nfs_cp_1n/
└── tmpfs_cp_1n/
```

Where:
- Directory names indicate storage type and number of nodes (e.g., `beegfs_pior_buffered_1n` = beegfs with 1 node)
- JSON files contain benchmark results with names like `pior_64m_1gb_n8_1.json` (64MB transfer, 1GB file, 8 tasks, run 1)

## Usage

### 1. Quick Start

```python
from ior_utils import *

# Collect all IOR data
df = collect_ior_data("ior_data")

# Save to CSV
save_master_ior_df(df, "master_ior_df.csv")

# Print overview
print_data_overview(df)
```

### 2. Data Cleaning

The tools automatically clean data by removing rows with zero throughput (trMiB = 0):

```python
# Clean data by removing rows with zero throughput
cleaned_df = clean_data_by_throughput(df, min_throughput=0.0)

# Export individual storage data (automatically cleaned)
export_storage_data(df, clean_data=True, min_throughput=0.0)
```

### 3. Averaged Statistics

Calculate averaged statistics for each storage type, combining multiple trials:

```python
# Calculate averaged statistics (only averaged or single-trial rows)
df_averaged = calculate_averaged_statistics(df)

# Save averaged-only data
save_master_ior_df(df_averaged, "master_ior_df_averaged.csv")
```

**Averaging Logic:**
- If a benchmark has multiple trials, calculate the mean of `trMiB` and `totalTime`
- If a benchmark has only one trial, keep that row as-is
- All averaged rows are marked with `ave_` prefix (e.g., `ave_beegfs`, `ave_localssd`)

### 4. Combined Data Analysis

The system now supports combined analysis of IOR and cp benchmark data:

- `master_ior_df.csv`: Original IOR benchmark data
- `master_ior_df_averaged.csv`: Averaged-only IOR benchmark data (no original non-averaged data)
- `updated_master_ior_df.csv`: Combined averaged IOR and cp benchmark data
- `master_move_data.csv`: CP benchmark data only

### 5. Concatenation with Averaging

Use the concatenation script to combine IOR and cp data with automatic averaging:

```bash
python concat_csv_files.py
```

This script will:
1. Load `master_ior_df.csv` and calculate averaged statistics
2. Filter to keep only averaged IOR data (remove original non-averaged data)
3. Load `master_move_data.csv` (cp data)
4. Concatenate averaged IOR data with cp data
5. Save result to `updated_master_ior_df.csv`

**Output files:**
- `master_ior_df_averaged.csv`: Only averaged IOR data
- `updated_master_ior_df.csv`: Averaged IOR data + cp data

### 6. Filter by Storage Types

```python
# Collect data for specific storage types only
selected_storage_types = ['beegfs', 'localssd']  # Exclude nfs and tmpfs
df_filtered = collect_ior_data(data_dir, storage_types=selected_storage_types)
save_master_ior_df(df_filtered, "master_ior_df_filtered.csv")
```

### 7. Data Filtering

```python
# Filter data by various conditions
beegfs_data = filter_data_by_conditions(df, storage_type='beegfs')
one_node_data = filter_data_by_conditions(df, num_nodes=1)
large_transfer_data = filter_data_by_conditions(df, transfer_size=64*1024*1024)  # 64MB
write_operations = filter_data_by_conditions(df, operation='write')
```

### 8. Visualization

```python
# Compare storage types for fixed transfer size and nodes
plot_storage_comparison(df, 
                       transfer_size=64*1024*1024,  # 64MB
                       num_nodes=1,
                       title="Storage Comparison")

# Analyze transfer size impact for specific storage
plot_transfer_size_analysis(df,
                           storage_type='beegfs',
                           num_nodes=1,
                           title="Transfer Size Analysis")
```

### 9. Jupyter Notebooks

Run the Jupyter notebooks for interactive analysis:

```bash
# IOR benchmark analysis
jupyter notebook ior_analysis_new.ipynb

# CP benchmark analysis
jupyter notebook move_data_bench_analysis.ipynb
```

The notebooks provide:
- Data collection and overview
- Storage comparison plots
- Transfer size analysis
- Scaling analysis across nodes
- Summary statistics
- Custom analysis functions
- **All plots are automatically saved to the `plot/` directory**
- **Automatic data cleaning (removes rows with trMiB = 0)**

## Key Features

### 1. Modular Design
- Core functions in `ior_utils.py`
- Visualization and analysis in Jupyter notebooks
- Easy to extend and modify

### 2. Flexible Data Selection
- Choose which storage types to include
- Filter by transfer size, number of nodes, operations
- Combine multiple filters

### 3. Data Cleaning
- **Automatic removal of rows with zero throughput**
- Configurable minimum throughput threshold
- Detailed cleaning statistics

### 4. Averaged Statistics
- **Automatic averaging of multiple trials**
- Single-trial benchmarks preserved as-is
- New storage types with `ave_` prefix
- Configurable averaging parameters
- **Option to keep only averaged data** (no original non-averaged data)

### 5. Multi-Benchmark Support
- **IOR benchmark data processing**
- **CP benchmark data integration**
- Combined analysis capabilities
- Separate analysis workflows

### 6. Concatenation Pipeline
- **Automatic averaging of IOR data**
- **Filtering to keep only averaged values**
- **Seamless concatenation with cp data**
- **Clean output with no duplicate data**

### 7. Comprehensive Analysis
- Storage type comparisons
- Transfer size impact analysis
- Scaling behavior across nodes
- Summary statistics and heatmaps

### 8. Easy Visualization
- Pre-built plotting functions
- Customizable plots
- **Automatic plot directory creation and management**
- All plots saved to `plot/` directory with descriptive names

## Data Cleaning

The tools automatically clean the data by removing rows where throughput (trMiB) is 0 or below a specified threshold:

- **Default behavior**: Removes all rows where `trMiB = 0`
- **Configurable threshold**: Can set minimum throughput threshold
- **Individual storage files**: Each storage type CSV is cleaned separately
- **Statistics provided**: Shows how many records were removed

### Cleaning Statistics (Example)
```
Cleaned ior_data_beegfs.csv:
  Original records: 6412
  Cleaned records: 4748
  Removed records: 1664

Cleaned ior_data_localssd.csv:
  Original records: 7516
  Cleaned records: 5596
  Removed records: 1920
```

## Plot Directory

The `plot/` directory is automatically created when running the notebooks and contains all generated plots:

- `storage_comparison_64mb_1node.png`: Storage comparison for 64MB transfer, 1 node
- `storage_comparison_multiple_transfer_sizes.png`: Comparison across different transfer sizes
- `transfer_size_analysis_all_storage.png`: Transfer size analysis for all storage types
- `transfer_size_analysis_beegfs.png`: Detailed transfer size analysis for beegfs
- `scaling_analysis_all_storage.png`: Scaling analysis across nodes
- `throughput_heatmap.png`: Heatmap of average throughput
- `detailed_analysis_*.png`: Detailed analysis for specific conditions
- `test_averaged_*.png`: Averaged statistics comparison plots

## Testing

Run the test scripts to verify functionality:

```bash
# Test IOR utilities
python test_ior_utils.py

# Test averaged statistics
python test_averaged_statistics.py
```

The test scripts will:
- Test data collection
- Test filtering functions
- Test save/load functionality
- Test summary statistics
- Test averaged statistics calculation
- Verify path extraction

## Cleaning Existing Files

To clean existing CSV files:

```bash
python clean_csv_files.py
```

This script will:
- Clean all individual storage CSV files
- Remove rows where trMiB = 0
- Provide cleaning statistics
- Save cleaned data back to the same files

## Concatenation Workflow

The complete workflow for combining IOR and cp data:

1. **Run IOR analysis**: Execute `ior_analysis_new.ipynb` to generate `master_ior_df.csv`
2. **Run CP analysis**: Execute `move_data_bench_analysis.ipynb` to generate `master_move_data.csv`
3. **Concatenate with averaging**: Run `python concat_csv_files.py`

**What the concatenation script does:**
- Loads IOR data and calculates averaged statistics
- Filters to keep only averaged IOR data (removes original non-averaged data)
- Loads cp data
- Concatenates averaged IOR data with cp data
- Saves clean combined dataset to `updated_master_ior_df.csv`

## Example Output

The tools will generate:
- `master_ior_df.csv`: Master DataFrame with all IOR data
- `master_ior_df_averaged.csv`: Averaged-only IOR benchmark data
- `updated_master_ior_df.csv`: Combined averaged IOR and cp benchmark data
- `master_move_data.csv`: CP benchmark data only
- `plot/`: Directory containing all PNG plots for visualization
- `ior_summary_statistics.csv`: Summary statistics CSV file
- `ior_data_*.csv`: Individual storage type CSV files (cleaned)

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- jupyter (for notebooks)

Install requirements:
```bash
pip install pandas numpy matplotlib seaborn jupyter
```

## Troubleshooting

1. **Data directory not found**: Make sure you're running from the `perf_profiles` directory
2. **No JSON files found**: Check that the `ior_data` directory contains the expected structure
3. **Import errors**: Ensure all required packages are installed
4. **Empty results**: Verify that the JSON files contain the expected data structure
5. **Plot directory issues**: The `plot/` directory is created automatically, but ensure you have write permissions
6. **Zero throughput data**: Use the cleaning functions to remove invalid data points
7. **Averaged statistics issues**: Check that the input DataFrame contains multiple trials for averaging
8. **Concatenation errors**: Ensure both `master_ior_df.csv` and `master_move_data.csv` exist before running concatenation

## Extending the Tools

To add new analysis features:

1. Add new functions to `ior_utils.py`
2. Import and use them in the Jupyter notebooks
3. Update the test scripts if needed
4. Document new features in this README
5. Use `os.path.join(plot_dir, 'filename.png')` for saving new plots
6. Consider data cleaning for new data sources
7. Update averaged statistics logic for new benchmark types
8. Update concatenation script for new data formats

## Data Format

### IOR JSON Files
The JSON files should contain:
- `summary` array with operation results
- Each summary entry should have: `operation`, `transferSize`, `numTasks`, `tasksPerNode`, `MeanTime`, `bwMeanMIB`
- File names should follow the pattern: `pior_<transfer>m_<size>gb_n<tasks>_<run>.json`

### CP Benchmark Data
The cp benchmark data should be structured to match the IOR data format for combined analysis.

### Storage Type Naming Convention
- **IOR data**: `beegfs`, `localssd`, `nfs`, `tmpfs`
- **Averaged IOR data**: `ave_beegfs`, `ave_localssd`, `ave_nfs`, `ave_tmpfs`
- **CP data**: `beegfs-ssd`, `ssd-beegfs`, `beegfs-tmpfs`, `tmpfs-beegfs`, `ssd-ssd`, `tmpfs-tmpfs` 