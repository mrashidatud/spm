# Python Tests for Workflow Analysis

This directory contains comprehensive test scripts for the workflow analysis modules. These tests verify the functionality of each module and ensure the complete pipeline works correctly.

## Test Scripts Overview

### 1. `test_simple_workflow.py`
**Purpose**: Basic workflow data loading test

**What it tests**:
- Workflow data loading functionality
- Task definition parsing
- Basic data structure validation

**Key Features**:
- Tests `load_workflow_data()` function
- Validates workflow configuration
- Checks data integrity and structure

**Usage**:
```bash
cd python_tests
python3 test_simple_workflow.py
```

**Expected Output**:
```
=== Testing workflow data loading ===
✓ Workflow data loaded successfully
  - Records: 267
  - Task definitions: 4
  - Workflow dict entries: 4
  - Unique tasks: ['openmm', 'aggregate', 'training', 'inference']
  - Operations: [0, 1] (workflow data format - integers)
  - Sample data shape: (267, 55)
✓ All steps completed successfully!
```

---

### 2. `test_interpolation.py`
**Purpose**: Test 4D interpolation and transfer rate estimation

**What it tests**:
- 4D interpolation function
- Transfer rate estimation for workflow tasks
- IOR data integration

**Key Features**:
- Tests `estimate_transfer_rates_for_workflow()` function
- Validates interpolation results
- Checks estimated transfer rate columns

**Usage**:
```bash
cd python_tests
python3 test_interpolation.py
```

**Expected Output**:
```
Loading IOR data...
IOR data shape: (1000, 15)
Test workflow data shape: (2, 4)
Testing interpolation with limited parallelism...
Estimated columns: 12
Sample estimated values:
estimated_trMiB_localssd_1p: 1234.56
estimated_trMiB_beegfs_1p: 2345.67
...
Non-zero estimated values: 10/12 (83.3%)
```

---

### 3. `test_complete_workflow.py`
**Purpose**: End-to-end workflow analysis test

**What it tests**:
- Complete workflow analysis pipeline
- All major processing steps
- SPM calculation workflow

**Key Features**:
- Tests the entire analysis pipeline
- Validates data flow between modules
- Checks final SPM results

**Usage**:
```bash
cd python_tests
python3 test_complete_workflow.py
```

**Expected Output**:
```
=== Step 1: Loading workflow data ===
Workflow data loaded: 267 records
Task definitions: 4
Unique tasks: ['openmm', 'aggregate', 'training', 'inference']

=== Step 2: Calculating I/O time breakdown ===
I/O breakdown completed: 45.23 seconds total

=== Step 3: Calculating aggregate file size per node ===
Aggregate file size calculation completed

=== Step 4: Estimating transfer rates ===
Loaded 1000 IOR benchmark records
Estimated transfer rate columns: 12
Non-zero estimated values: 2000/2400 (83.3%)

=== Step 5: Calculating SPM values ===
SPM calculation completed: 3 producer-consumer pairs
  - openmm:aggregate
  - openmm:training
  - aggregate:training

=== Workflow analysis completed successfully! ===
```

---

### 4. `test_notebook_sections.py`
**Purpose**: Comprehensive test of all notebook sections

**What it tests**:
- All major notebook functionality
- Complete data processing pipeline
- Results generation and saving

**Key Features**:
- Tests all 7 major processing steps
- Validates data saving functionality
- Checks visualization generation
- Comprehensive error handling

**Usage**:
```bash
cd python_tests
python3 test_notebook_sections.py
```

**Expected Output**:
```
=== Testing Notebook Sections ===

Step 1: Loading workflow data...
✓ Workflow data loaded: 267 records
✓ Task definitions: 4
✓ Unique tasks: ['openmm', 'aggregate', 'training', 'inference']
✓ Stages: [0, 1, 2]

Step 2: Calculating I/O time breakdown...
✓ I/O breakdown completed: 45.23 seconds total

Step 3: Calculating aggregate file size per node...
✓ Aggregate file size calculation completed

Step 4: Estimating transfer rates...
✓ Loaded 1000 IOR benchmark records
✓ Estimated transfer rate columns: 12
✓ Non-zero estimated values: 2000/2400 (83.3%)

Step 5: Calculating SPM values...
✓ SPM calculation completed: 3 producer-consumer pairs

Step 6: Filtering storage options...
✓ Storage options filtered
✓ Best configurations selected

Step 7: Saving results...
✓ Saved workflow data to: ../analysis_data/ddmd_4n_l_workflow_data.csv
✓ Saved SPM results to: ../analysis_data/ddmd_4n_l_spm_results.json

=== All notebook sections completed successfully! ===
```

---

### 5. `test_modular_structure.py`
**Purpose**: Test modular architecture and imports

**What it tests**:
- Module import functionality
- Configuration validation
- Data utility functions
- Interpolation functions
- SPM calculator functions
- Visualization functions

**Key Features**:
- Comprehensive import testing
- Configuration validation
- Functionality testing for each module
- Error handling verification

**Usage**:
```bash
cd python_tests
python3 test_modular_structure.py
```

**Expected Output**:
```
Testing module imports...
✓ workflow_config imported successfully
✓ workflow_data_utils imported successfully
✓ workflow_interpolation imported successfully
✓ workflow_spm_calculator imported successfully
✓ workflow_visualization imported successfully
✓ workflow_analysis_main imported successfully

Testing configuration...
Default workflow: ddmd_4n_l
Available workflows: ['ddmd_2n_s', 'ddmd_4n_l', '1kg', '1kg_2', ...]
Storage types: ['localssd', 'beegfs', 'tmpfs']
✓ Default workflow configuration exists

Testing data utilities...
✓ Storage code transformation works
✓ Storage code decoding works
✓ Bytes to MB conversion works

Testing interpolation functions...
✓ Interpolation result: 125.50, slope: 0.25

Testing SPM calculator...
✓ SPM normalization works

Testing visualization functions...
✓ Summary report generation works

=== All tests passed! ===
```

## Running All Tests

To run all tests in sequence:

```bash
cd python_tests

# Run tests in order of complexity
python3 test_simple_workflow.py
python3 test_interpolation.py
python3 test_complete_workflow.py
python3 test_notebook_sections.py
python3 test_modular_structure.py
```

Or run a specific test:

```bash
python3 test_[test_name].py
```

## Test Dependencies

### Required Files
- `../../perf_profiles/updated_master_ior_df.csv` - IOR benchmark data
- `../modules/` - All workflow analysis modules
- `../analysis_data/` - Output directory (created automatically)

### Python Dependencies
- pandas
- numpy
- networkx
- matplotlib
- scipy

## Test Output

### Success Indicators
- ✓ Green checkmarks for successful operations
- Detailed progress information
- Final success messages

### Error Indicators
- ✗ Red X marks for failed operations
- Detailed error messages with tracebacks
- Warning messages for non-critical issues

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'modules'
   ```
   **Solution**: Ensure you're running from the `python_tests` directory

2. **File Not Found**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory: '../../perf_profiles/updated_master_ior_df.csv'
   ```
   **Solution**: Check that the IOR data file exists in the correct location

3. **Permission Errors**:
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   **Solution**: Ensure write permissions for the `../analysis_data/` directory

### Debug Mode

To run tests with additional debugging information:

```python
# Add to any test script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Test Coverage

These tests cover:
- ✅ Data loading and validation
- ✅ Configuration management
- ✅ Interpolation and estimation
- ✅ SPM calculations
- ✅ Visualization generation
- ✅ File I/O operations
- ✅ Error handling
- ✅ Module integration

## Contributing

When adding new functionality:
1. Create a corresponding test in this directory
2. Follow the naming convention: `test_[functionality].py`
3. Include comprehensive error handling
4. Add documentation to this README
5. Ensure tests pass before committing changes 