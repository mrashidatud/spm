#!/usr/bin/env python3
"""
Workflow Runtime Comparison Visualization
Compares actual runtime vs SPM-pythonruntime for different workflows
"""

import matplotlib.pyplot as plt
import numpy as np

# 1000 Genomes I/O Data
genome_io_data = {
  "indiv": {
    "read": {
      "op_count": "200 random",
      "avg_io_size": [8192, 8192],  # 8K
      "total_io_size": "24G"
    },
    "write": {
      "op_count": "100 sequential",
      "avg_io_size": [81920, 102400],  # 80K - 100K range
      "total_io_size": "23M"
    }
  },
  "merge": {
    "read": {
      "op_count": "300 random",
      "avg_io_size": [1433.6, 1433.6],  # 1.4K
      "total_io_size": "750M"
    },
    "write": {
      "op_count": "10 sequential",
      "avg_io_size": [778240, 778240],  # 760K
      "total_io_size": "16M"
    }
  },
  "sift": {
    "read": {
      "op_count": "20 random",
      "avg_io_size": [32768, 8192],  # 32K / 8K (two values)
      "total_io_size": "24G"
    },
    "write": {
      "op_count": "10 sequential",
      "avg_io_size": [696320, 696320],  # 680K
      "total_io_size": "8M"
    }
  },
  "mutation": {
    "read": {
      "op_count": "140 random, 70 sequential",
      "avg_io_size": [16384, 43008],  # 16K - 42K range
      "total_io_size": "1G"
    },
    "write": {
      "op_count": "70 sequential",
      "avg_io_size": [251904, 251904],  # 246K
      "total_io_size": "230M"
    }
  },
  "freq": {
    "read": {
      "op_count": "140 random, 70 sequential",
      "avg_io_size": [1024, 1024],  # 1K
      "total_io_size": "750M"
    },
    "write": {
      "op_count": "140 random, 70 sequential",
      "avg_io_size": [600, 4096],  # 600B - 4K range
      "total_io_size": "360M"
    }
  }
}

# DeepDriveMD I/O Data
ddmd_io_data = {
  "openmm": {
    "read": {
      "op_count": "--",
      "avg_io_size": "--",
      "total_io_size": "--"
    },
    "write": {
      "op_count": "24 random",
      "avg_io_size": [1024, 1024],  # 1K
      "total_io_size": "175M"
    }
  },
  "aggregate": {
    "read": {
      "op_count": "13 random",
      "avg_io_size": [354, 354],  # 354B
      "total_io_size": "222M"
    },
    "write": {
      "op_count": "1 random",
      "avg_io_size": [328, 328],  # 328B
      "total_io_size": "16M"
    }
  },
  "train": {
    "read": {
      "op_count": "12 sequential, 1 random",
      "avg_io_size": [157, 157],  # 157B
      "total_io_size": "555M"
    },
    "write": {
      "op_count": "23 sequential, 11 random",
      "avg_io_size": [657408, 657408],  # 642K
      "total_io_size": "4.4G"
    }
  },
  "inference": {
    "read": {
      "op_count": "12 sequential, 1 random",
      "avg_io_size": [209, 209],  # 209B
      "total_io_size": "190M"
    },
    "write": {
      "op_count": "1 random",
      "avg_io_size": [4096, 4096],  # 4K
      "total_io_size": "50K"
    }
  }
}

# PyFLEXTRKR I/O Data
pyflex_io_data = {
  "idfea": {
    "read": {
      "op_count": "522049 random",
      "avg_io_size": [64512, 64512],  # 63K
      "total_io_size": "3.2G"
    },
    "write": {
      "op_count": "524900 random",
      "avg_io_size": [51200, 51200],  # 50K
      "total_io_size": "2.6G"
    }
  },
  "single": {
    "read": {
      "op_count": "2415754 random",
      "avg_io_size": [5120, 5120],  # 5K
      "total_io_size": "1087M"
    },
    "write": {
      "op_count": "13254 random",
      "avg_io_size": [12288, 12288],  # 12K
      "total_io_size": "16M"
    }
  }
#   // ... (continuing with the same pattern for all other tasks)
}

def plot_workflow_runtime_comparison():
    """Plot comparison between actual and SPM-predicted workflow runtimes"""
    
    # Data
    dfman_runtime = {
        "workflow_name": ["1kgenome", "pyflextrkr", "ddmd"],
        "runtime_sec": [508, 877, 249]
    }
    
    spm_runtime = {
        "workflow_name": ["1kgenome", "pyflextrkr", "ddmd"],
        "runtime_sec": [211.03, 439.14, 214.00]
    }

    baseline_runtime = {
        "workflow_name": ["1kgenome", "pyflextrkr", "ddmd"],
        "runtime_sec": [973.11, 474.86, 212.00]
    }
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create bar positions
    x = np.arange(len(dfman_runtime["workflow_name"]))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, dfman_runtime["runtime_sec"], width, 
                   label='DFMan', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, spm_runtime["runtime_sec"], width,
                   label='SPM', color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, baseline_runtime["runtime_sec"], width,
                   label='Baseline', color='#2ca02c', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=14, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Customize the plot
    ax.set_xlabel('Workflow', fontsize=18, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=18, fontweight='bold')
    # ax.set_title('Workflow Time: DFMan vs SPM-Predicted vs Baseline', 
    #              fontsize=20, fontweight='bold', pad=30)
    ax.set_xticks(x)
    ax.set_xticklabels(dfman_runtime["workflow_name"], fontsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(max(dfman_runtime["runtime_sec"]), max(spm_runtime["runtime_sec"]), max(baseline_runtime["runtime_sec"])) * 1.1)
    
    # Add speedup annotations
    for i, (actual, spm, baseline) in enumerate(zip(dfman_runtime["runtime_sec"], spm_runtime["runtime_sec"], baseline_runtime["runtime_sec"])):
        speedup_vs_dfman = actual / spm
        speedup_vs_baseline = baseline / spm
        # Position at bottom of the bars
        ax.annotate(f'{speedup_vs_dfman:.1f}x faster than DFMan\n{speedup_vs_baseline:.1f}x faster than Baseline',
                   xy=(i, 0),
                   xytext=(0, 10),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffcc99', alpha=0.7))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("workflow_runtime_comparison.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig("workflow_runtime_comparison.png", format='png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("WORKFLOW RUNTIME COMPARISON SUMMARY")
    print("="*60)
    
    total_actual = sum(dfman_runtime["runtime_sec"])
    total_spm = sum(spm_runtime["runtime_sec"])
    total_improvement = ((total_actual - total_spm) / total_actual) * 100
    
    print(f"Total DFMan Runtime: {total_actual:.2f} seconds")
    print(f"Total SPM-Predicted Runtime: {total_spm:.2f} seconds")
    print(f"Total Improvement: {total_improvement:.1f}% faster")
    print()
    
    for i, workflow in enumerate(dfman_runtime["workflow_name"]):
        actual = dfman_runtime["runtime_sec"][i]
        spm = spm_runtime["runtime_sec"][i]
        improvement = ((actual - spm) / actual) * 100
        print(f"{workflow}:")
        print(f"  DFMan: {actual:.2f} seconds")
        print(f"  SPM: {spm:.2f} seconds")
        print(f"  Improvement: {improvement:.1f}% faster")
        print()
    
    # Print SPM prediction error rates
    print("\n" + "="*60)
    print("SPM PREDICTION ERROR RATES")
    print("="*60)
    
    # Error rate data for each workflow
    error_rates = {
        "1kgenome": {
            "total_entries": 45,
            "global_10": 0,
            "local_10": 12,
            "global_5": 2,
            "local_5": 21
        },
        "pyflextrkr": {
            "total_entries": 72,
            "global_10": 0,
            "local_10": 20,
            "global_5": 0,
            "local_5": 28
        },
        "ddmd": {
            "total_entries": 12,
            "global_10": 1,
            "local_10": 5,
            "global_5": 4,
            "local_5": 7
        }
    }
    
    # Print individual workflow error rates
    for workflow, rates in error_rates.items():
        print(f"{workflow.upper()}:")
        print(f"  Error rate over 10% global margin of error: {rates['global_10']}/{rates['total_entries']}")
        print(f"  Error rate over 10% local margin of error: {rates['local_10']}/{rates['total_entries']}")
        print(f"  Error rate over 5% global margin of error: {rates['global_5']}/{rates['total_entries']}")
        print(f"  Error rate over 5% local margin of error: {rates['local_5']}/{rates['total_entries']}")
        print()
    
    # Calculate and print total error rates across all workflows
    total_entries = sum(rates['total_entries'] for rates in error_rates.values())
    total_global_10 = sum(rates['global_10'] for rates in error_rates.values())
    total_local_10 = sum(rates['local_10'] for rates in error_rates.values())
    total_global_5 = sum(rates['global_5'] for rates in error_rates.values())
    total_local_5 = sum(rates['local_5'] for rates in error_rates.values())
    
    print("TOTAL ACROSS ALL WORKFLOWS:")
    print(f"  Total entries: {total_entries}")
    print(f"  Error rate over 10% global margin of error: {total_global_10}/{total_entries} ({total_global_10/total_entries*100:.1f}%)")
    print(f"  Error rate over 10% local margin of error: {total_local_10}/{total_entries} ({total_local_10/total_entries*100:.1f}%)")
    print(f"  Error rate over 5% global margin of error: {total_global_5}/{total_entries} ({total_global_5/total_entries*100:.1f}%)")
    print(f"  Error rate over 5% local margin of error: {total_local_5}/{total_entries} ({total_local_5/total_entries*100:.1f}%)")
    print()
    
    # Calculate and print correct rates
    print("CORRECT RATES (within margin of error):")
    print(f"  Correct rate within 10% global margin: {(total_entries-total_global_10)/total_entries*100:.1f}%")
    print(f"  Correct rate within 10% local margin: {(total_entries-total_local_10)/total_entries*100:.1f}%")
    print(f"  Correct rate within 5% global margin: {(total_entries-total_global_5)/total_entries*100:.1f}%")
    print(f"  Correct rate within 5% local margin: {(total_entries-total_local_5)/total_entries*100:.1f}%")
    print()

def calculate_baseline_performance_improvement():
    """Calculate and print performance improvement compared to baseline runtime"""
    
    # Data
    dfman_runtime = {
        "workflow_name": ["1kgenome", "pyflextrkr", "ddmd"],
        "runtime_sec": [508, 877, 249]
    }
    
    spm_runtime = {
        "workflow_name": ["1kgenome", "pyflextrkr", "ddmd"],
        "runtime_sec": [211.03, 439.14, 214.00]
    }
    
    baseline_runtime = {
        "workflow_name": ["1kgenome", "pyflextrkr", "ddmd"],
        "runtime_sec": [973.11, 474.86, 212.00]
    }
    
    print("\n" + "="*60)
    print("PERFORMANCE IMPROVEMENT vs BASELINE RUNTIME")
    print("="*60)
    
    # Calculate total improvements
    total_actual = sum(dfman_runtime["runtime_sec"])
    total_spm = sum(spm_runtime["runtime_sec"])
    total_baseline = sum(baseline_runtime["runtime_sec"])
    
    actual_improvement = ((total_baseline - total_actual) / total_baseline) * 100
    spm_improvement = ((total_baseline - total_spm) / total_baseline) * 100
    
    print(f"Total Baseline Runtime: {total_baseline:.2f} seconds")
    print(f"Total DFMan Runtime: {total_actual:.2f} seconds")
    print(f"Total SPM-Predicted Runtime: {total_spm:.2f} seconds")
    print()
    print(f"DFMan vs Baseline: {actual_improvement:.1f}% improvement ({total_baseline/total_actual:.1f}x faster)")
    print(f"SPM vs Baseline: {spm_improvement:.1f}% improvement ({total_baseline/total_spm:.1f}x faster)")
    print()
    
    # Calculate individual workflow improvements
    for i, workflow in enumerate(dfman_runtime["workflow_name"]):
        actual = dfman_runtime["runtime_sec"][i]
        spm = spm_runtime["runtime_sec"][i]
        baseline = baseline_runtime["runtime_sec"][i]
        
        actual_improvement = ((baseline - actual) / baseline) * 100
        spm_improvement = ((baseline - spm) / baseline) * 100
        
        print(f"{workflow.upper()}:")
        print(f"  Baseline: {baseline:.2f} seconds")
        print(f"  DFMan: {actual:.2f} seconds ({actual_improvement:.1f}% improvement, {baseline/actual:.1f}x faster)")
        print(f"  SPM: {spm:.2f} seconds ({spm_improvement:.1f}% improvement, {baseline/spm:.1f}x faster)")
        print()

def main():
    """Main function to run the workflow runtime comparison"""
    print("Generating workflow runtime comparison plot...")
    plot_workflow_runtime_comparison()
    print("Plot saved as 'workflow_runtime_comparison.pdf' and 'workflow_runtime_comparison.png'")
    
    # Calculate and print baseline performance improvements
    calculate_baseline_performance_improvement()

if __name__ == "__main__":
    main()
