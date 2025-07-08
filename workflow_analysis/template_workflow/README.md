# Template Workflow - Detailed Documentation

This directory contains a template workflow system designed for testing, development, and learning the workflow analysis pipeline. The template demonstrates a simple producer-consumer relationship with artificial data that mimics real workflow patterns.

## 📁 Directory Structure

```
template_workflow/
├── README.md                        # This detailed documentation
├── template_script_order.json       # Workflow configuration
└── template_run/                    # Workflow execution data
    ├── workflow_data.csv            # Artificial workflow data
    ├── input_data_1.txt            # Example input files
    ├── input_data_2.txt
    ├── input_data_3.txt
    ├── input_data_4.txt
    ├── task1_output_1.dat          # Example task1 output files
    ├── task1_output_2.dat
    ├── task1_output_3.dat
    ├── task1_output_4.dat
    ├── final_result_1.out          # Example final output files
    └── final_result_2.out
```

## 🔧 Workflow Configuration

### Script Order JSON Structure

The `template_script_order.json` file defines the workflow structure:

```json
{
    "task1": {
        "stage_order": 0,
        "parallelism": 4,
        "num_tasks": 4,
        "predecessors": {
            "initial_data": {
                "inputs": ["input_data_\\d+\\.txt"]
            }
        },
        "outputs": [
            "task1_output_\\d+\\.dat",
            "task1_results_\\d+\\.json"
        ]
    },
    "task2": {
        "stage_order": 1,
        "parallelism": 2,
        "num_tasks": 2,
        "predecessors": {
            "task1": {
                "inputs": ["task1_output_\\d+\\.dat"]
            }
        },
        "outputs": [
            "final_result_\\d+\\.out",
            "summary_\\d+\\.txt"
        ]
    }
}
```

### Configuration Fields Explained

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `stage_order` | Integer | Execution order of the task (0, 1, 2, ...) | `0` |
| `parallelism` | Integer | Number of parallel tasks for this stage | `4` |
| `num_tasks` | Integer | Total number of tasks in this stage | `4` |
| `predecessors` | Object | Input dependencies from previous stages | `{"task1": {"inputs": [...]}}` |
| `inputs` | Array | Regex patterns matching input files | `["input_data_\\d+\\.txt"]` |
| `outputs` | Array | Regex patterns matching output files | `["task1_output_\\d+\\.dat"]` |

## 📊 Workflow Data Metrics

### CSV Data Structure

The `workflow_data.csv` file contains artificial workflow data with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `operation` | Integer | 0=write, 1=read | `1` |
| `randomOffset` | Integer | 0=sequential, 1=random | `0` |
| `transferSize` | Integer | Block size in bytes | `4096` |
| `aggregateFilesizeMB` | Float | Total file size in MB | `94.51` |
| `numTasks` | Integer | Number of tasks | `4` |
| `parallelism` | Integer | Parallelism level | `4` |
| `totalTime` | Float | Processing time in seconds | `8.53` |
| `numNodesList` | String | List of node counts | `"[4]"` |
| `numNodes` | Integer | Number of nodes | `4` |
| `tasksPerNode` | Integer | Tasks per node | `1` |
| `trMiB` | Float | Transfer rate in MB/s | `11.08` |
| `storageType` | String | Storage type | `"beegfs"` |
| `opCount` | Integer | Number of operations | `4` |
| `taskName` | String | Task name | `"task1"` |
| `taskPID` | Integer | Process ID | `1000` |
| `fileName` | String | File name | `"input_data_1.txt"` |
| `stageOrder` | Integer | Stage execution order | `0` |
| `prevTask` | String | Previous task name | `"initial_data"` |
| `aggregateFilesizeMBtask` | Float | File size per task | `23.63` |

### Key Metrics Explained

#### 1. **Operation Types**
- **`operation = 0`**: Write operations (producing data)
- **`operation = 1`**: Read operations (consuming data)

#### 2. **File Access Patterns**
- **`randomOffset = 0`**: Sequential access pattern
- **`randomOffset = 1`**: Random access pattern

#### 3. **Performance Metrics**
- **`totalTime`**: Actual processing time in seconds
- **`trMiB`**: Transfer rate in MB/s (calculated as file_size / total_time)
- **`aggregateFilesizeMB`**: Total file size being processed

#### 4. **Parallelism Configuration**
- **`parallelism`**: Number of parallel tasks for this operation
- **`numTasks`**: Total number of tasks in this stage
- **`tasksPerNode`**: Tasks distributed per node (calculated as ceil(parallelism / numNodes))

#### 5. **Storage and Infrastructure**
- **`storageType`**: Storage system type (beegfs, localssd, tmpfs, etc.)
- **`numNodes`**: Number of compute nodes
- **`numNodesList`**: List of node configurations for multi-node analysis

## 🔄 Workflow Execution Flow

### Stage 0: Task1 (Producer)
```
Input Files: input_data_1.txt, input_data_2.txt, input_data_3.txt, input_data_4.txt
Operation: Read (operation=1)
Parallelism: 4 tasks
Output Files: task1_output_1.dat, task1_output_2.dat, task1_output_3.dat, task1_output_4.dat
```

### Stage 1: Task2 (Consumer)
```
Input Files: task1_output_1.dat, task1_output_2.dat
Operation: Read (operation=1)
Parallelism: 2 tasks
Output Files: final_result_1.out, final_result_2.out
```

## 📈 Data Generation Parameters

The template generator creates artificial data with configurable parameters:

### Default Parameters
- **Base file size**: 100 MB
- **Time variance**: 20% (adds realistic variability)
- **Number of nodes**: 4
- **Storage type**: beegfs (default)

### Customizable Parameters
```python
wf_df = generate_template_workflow_data(
    workflow_name="custom_workflow",
    num_nodes=8,                    # Custom node count
    base_file_size_mb=200.0,        # Custom file size
    time_variance=0.3,              # Custom timing variance
    debug=True
)
```

## 🎯 Producer-Consumer Relationships

### Task1 → Task2 Flow
1. **Task1 reads** from initial data files (`input_data_*.txt`)
2. **Task1 writes** output files (`task1_output_*.dat`)
3. **Task2 reads** from Task1's output files (`task1_output_*.dat`)
4. **Task2 writes** final results (`final_result_*.out`)

### Data Dependencies
- **Stage 0**: Task1 depends on `initial_data`
- **Stage 1**: Task2 depends on `task1`

### File Pattern Matching
- **Input patterns**: Regex patterns that match input files
- **Output patterns**: Regex patterns that match output files
- **Dependency resolution**: System matches file patterns to establish producer-consumer relationships

## 🔍 Analysis Pipeline Integration

### 1. **Data Loading**
The template workflow data is loaded directly from `workflow_data.csv` when present in the test folder.

### 2. **I/O Time Breakdown**
- Calculates total I/O time per task
- Separates read vs. write operations
- Adjusts for parallelism

### 3. **Transfer Rate Estimation**
- Uses IOR benchmark data to estimate transfer rates
- Interpolates for different storage configurations
- Handles multi-node scenarios

### 4. **SPM Calculation**
- Identifies producer-consumer pairs
- Calculates Storage Performance Matching (SPM) values
- Ranks storage configurations

### 5. **Results Display**
- Shows top-ranked storage configurations
- Displays SPM analysis
- Generates visualizations

## 🛠️ Usage Examples

### Running Template Analysis
```bash
# Basic analysis
python3 workflow_analysis_main.py --workflow template_workflow --no-save

# With custom IOR data
python3 workflow_analysis_main.py --workflow template_workflow --ior-data ../perf_profiles/updated_master_ior_df.csv

# With debug output
python3 workflow_analysis_main.py --workflow template_workflow --debug
```

### Programmatic Usage
```python
from modules.workflow_template_generator import generate_complete_template

# Generate new template
result = generate_complete_template(
    workflow_name="my_test_workflow",
    debug=True
)

# Analyze the generated template
from workflow_analysis_main import run_workflow_analysis
results = run_workflow_analysis("my_test_workflow")
```

## 📋 Template Customization

### Creating Custom Workflows
1. **Define task structure** in script order JSON
2. **Specify file patterns** for inputs and outputs
3. **Set parallelism** and node configurations
4. **Generate artificial data** using the template generator
5. **Test with analysis pipeline**

### Example Custom Configuration
```json
{
    "preprocessing": {
        "stage_order": 0,
        "parallelism": 8,
        "num_tasks": 8,
        "predecessors": {
            "raw_data": {
                "inputs": ["raw_\\d+\\.csv"]
            }
        },
        "outputs": ["processed_\\d+\\.h5"]
    },
    "training": {
        "stage_order": 1,
        "parallelism": 4,
        "num_tasks": 4,
        "predecessors": {
            "preprocessing": {
                "inputs": ["processed_\\d+\\.h5"]
            }
        },
        "outputs": ["model_\\d+\\.pkl"]
    }
}
```

## 🔧 Troubleshooting

### Common Issues
1. **Missing columns**: Ensure all `WF_PARAMS` columns are present
2. **Data type mismatches**: Check numeric vs. string values
3. **File path issues**: Verify relative paths in configuration
4. **Import errors**: Ensure modules are in Python path

### Debug Mode
Enable debug output to see detailed processing information:
```python
results = calculate_spm_for_workflow(wf_df, debug=True)
```

## 📚 Related Documentation

- **Main README**: `../README.md` - Overview of the workflow analysis system
- **Module Documentation**: `../modules/README.md` - Detailed module descriptions
- **API Reference**: `../modules/workflow_template_generator.py` - Template generator API
- **Configuration**: `../modules/workflow_config.py` - System configuration

---

This template workflow provides a foundation for understanding and testing the workflow analysis system. Use it as a starting point for creating custom workflows or learning the analysis pipeline. 