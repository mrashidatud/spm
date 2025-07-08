# TODO - Workflow Analysis Fixes and Improvements

This document tracks the current issues and planned improvements for the workflow analysis system.

## New TODOs

### 1. Add Plotting for Producer-Consumer Tasks for Paper
- [ ] Add plotting functions that directly create publication-quality plots for producer-consumer task pairs, suitable for inclusion in papers.
- [ ] Integrate with workflow_spm_calculator and workflow_visualization modules.

### 2. Add Custom Filter Code for Workflow Storage Selection Plan
- [ ] Implement custom filter logic for each workflow to select the optimal storage selection plan.
- [ ] Allow per-workflow customization in storage selection and filtering.

---

## Testing Strategy

For each fix:
1. Create test cases to reproduce the issue
2. Implement the fix
3. Verify the fix resolves the issue
4. Add regression tests to prevent future issues
5. Update documentation

## Notes

- All fixes should maintain backward compatibility
- Add configuration options to enable/disable new features
- Update test scripts to cover new functionality
- Ensure performance impact is minimal
- Add appropriate error handling and logging

---

## Completed Issues (Previously in Priority Order)

### 1. Make "Calculate Aggregate File Size per Node" Optional ✅
- [x] The aggregate file size calculation is now optional via a configuration flag.

### 2. Step 4 Transfer Rate Estimation - Zero Values Issue ✅
- [x] Fixed data alignment and zero value issues in transfer rate estimation.

### 3. Add CP/SCP Operations for Storage Type Changes ✅
- [x] System now accounts for cp/scp operations when storage changes between producer and consumer.

### 4. Add Performance Modeling for Entire Workflow Stages ✅
- [x] Added workflow-wide performance modeling, critical path analysis, and end-to-end timing predictions.

