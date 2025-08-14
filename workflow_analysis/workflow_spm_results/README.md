# Workflow SPM Results

This directory contains the Storage Performance Matching (SPM) analysis results for different workflows. The files contain comprehensive SPM calculations for producer-consumer task pairs across various storage configurations and parallelism levels.

## File Types

### CSV Files (`*_filtered_spm_results.csv`)
These files contain detailed SPM results in tabular format with the following columns:

| Column | Description |
|--------|-------------|
| `producer` | Name of the producer task |
| `consumer` | Name of the consumer task |
| `producerStorageType` | Storage type for the producer task (e.g., beegfs, ssd, tmpfs, beegfs-ssd, beegfs-tmpfs) |
| `producerTasksPerNode` | Number of tasks per node for the producer |
| `consumerStorageType` | Storage type for the consumer task (e.g., beegfs, ssd, tmpfs, beegfs-ssd, beegfs-tmpfs) |
| `consumerTasksPerNode` | Number of tasks per node for the consumer |
| `SPM` | Storage Performance Matching value (lower is better) |



## Available Workflows

- **`ddmd_4n_l_filtered_spm_results.csv`** - DDMD workflow results
- **`1kg_filtered_spm_results.csv`** - 1K Genome workflow results  
- **`pyflex_s9_48f_filtered_spm_results.csv`** - PyFlex workflow results
- **`template_workflow_filtered_spm_results.csv`** - Template workflow results

## Important Notes on Filtering

⚠️ **User Selection Required**: These results files contain **unfiltered** SPM calculations. Users must manually select appropriate configurations based on their specific requirements.

### Unfiltered Configurations Include:

1. **Unreasonable Storage Transitions**:
   - Producer in `beegfs` and consumer in `ssd` or `tmpfs` without proper staging
   - Direct transitions between incompatible storage types
   - Missing intermediate staging operations

2. **Unachievable Parallelism**:
   - Tasks per node values that exceed hardware capabilities
   - Parallelism levels that don't match available resources
   - Configurations that violate workflow manager or system constraints

3. **Storage Type Combinations**:
   - All possible combinations of storage types, including impractical ones
   - Both single storage types (`beegfs`, `ssd`, `tmpfs`) and compound types (`beegfs-ssd`, `beegfs-tmpfs`, etc.) for both producers and consumers
   - Stage_in and stage_out operations with various storage transitions
   - Complex scenarios where both producer and consumer use compound storage types

## How to Use These Results

### 1. Understand Your System Constraints
Before selecting configurations, consider:
- Available storage types and their capacities
- Maximum tasks per node your system can handle
- Network bandwidth between storage systems
- Memory constraints for tmpfs operations

### 2. Filter Results Manually
Use the CSV files to filter results based on:
- **Storage Availability**: Only select storage types you have access to
- **Parallelism Limits**: Choose tasks per node values within your system's capabilities
- **Practical Transitions**: Select storage transitions that make sense for your workflow
- **Performance Requirements**: Focus on configurations with lower SPM values

### 3. Example Filtering Criteria
```python
# Example: Filter for practical configurations
filtered_results = df[
    (df['producerTasksPerNode'] <= max_tasks_per_node) &
    (df['consumerTasksPerNode'] <= max_tasks_per_node) &
    (df['SPM'] < acceptable_spm_threshold) &
    # Add your specific storage availability constraints
]
```

### 4. Consider Workflow Context
- **Stage_in Operations**: Usually require transitions from persistent storage (beegfs) to fast storage (ssd/tmpfs)
- **Stage_out Operations**: Usually require transitions from fast storage back to persistent storage
- **Intermediate Operations**: May use same storage type or practical transitions

## Storage Type Meanings

### Storage Types (Both Producer and Consumer)
Both producer and consumer storage types can use the same set of storage configurations:

#### Single Storage Types
- **`beegfs`**: Distributed parallel file system (persistent, slower)
- **`ssd`**: Local SSD storage (faster, limited capacity)
- **`tmpfs`**: Memory-based temporary storage (fastest, volatile)

#### Compound Storage Types (for transitions)
- **`beegfs-ssd`**: Transition from beegfs to ssd (stage_in)
- **`beegfs-tmpfs`**: Transition from beegfs to tmpfs (stage_in)
- **`ssd-beegfs`**: Transition from ssd to beegfs (stage_out)
- **`tmpfs-beegfs`**: Transition from tmpfs to beegfs (stage_out)
- **`ssd-ssd`**: Same-storage operation using cp/scp
- **`tmpfs-tmpfs`**: Same-storage operation using cp/scp

## SPM Value Interpretation

- **Lower SPM values** indicate better performance
- **SPM = 0** represents optimal performance (baseline)
- **Higher SPM values** indicate performance degradation
- **Negative SPM values** may indicate calculation errors or invalid configurations

## Recommendations

1. **Use CSV files** for detailed analysis and custom filtering
2. **Validate configurations** against your system's actual capabilities
3. **Consider the complete workflow** when selecting storage configurations
4. **Test selected configurations** in your actual environment before deployment

## Example Usage

```python
import pandas as pd

# Load results
df = pd.read_csv('ddmd_4n_l_filtered_spm_results.csv')

# Filter for practical configurations
practical_results = df[
    (df['producerTasksPerNode'] <= 8) &  # Your system limit
    (df['consumerTasksPerNode'] <= 8) &  # Your system limit
    (df['SPM'] < 1.0) &  # Acceptable performance threshold
    # Add storage availability filters
]

# Sort by SPM value (best first)
best_configs = practical_results.sort_values('SPM').head(10)
print(best_configs)
```

Remember: These results provide a comprehensive view of all possible configurations, but **manual selection and validation are essential** for practical deployment. 