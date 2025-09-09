#!/usr/bin/env python3
"""
SPM Figures for DeepDriveMD Workflow Analysis
Converted from Jupyter notebook to Python script

This script analyzes SPM (Storage Performance Model) results for DeepDriveMD workflow
and generates visualizations for rank errors and time deviations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os

# Color palette for consistent visualization
colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # yellow-green
    '#17becf'   # cyan
]

def create_runtime_data():
    """Create runtime data dictionary for workflow stages"""
    # runtime_data = {
    #     "store_conf": [
    #         "SSD 2n", "SSD 4n",
    #         "BeeGFS 2n", "BeeGFS 4n",
    #     ],
    #     "openmm+aggregate": [114, 143, 110, 113],
    #     "openmm+train": [205, 231, 212, 214],
    #     "openmm+inference": [126, 169, 122, 134]
    # }
    io_times_dict = {
        "openmm_write": 0.05025775266666667,
        "aggregate_read": 2.748506033,
        "aggregate_write": 0.10176218799999999,
        "training_read": 11.551488481,
        "training_write": 1.95473184,
        "inference_read": 1.850840342,
        "inference_write": 0.000028633
    }

    total_io_per_task = {
        "openmm": 0.05025775266666667,
        "aggregate": 2.850268221,
        "training": 13.506220321,
        "inference": 1.8508689749999998
    }

    stage_io_times = {
        "openmm+aggregate": 2.9005259736666664,
        "openmm+train": 13.556478073666668,
        "openmm+inference": 1.9011267276666666
    }
    # runtime_data = {
    #     "store_conf": [
    #         "SSD 2n", "SSD 4n",
    #         "BeeGFS 2n", "BeeGFS 4n"
    #     ],
    #     "openmm+aggregate": [114, 143, 110, 113],
    #     "openmm+train": [205, 231, 212, 214],
    #     "openmm+inference": [126, 169, 122, 134]
    # }
    runtime_data = {
        "store_conf": [
            "SSD 2n", "SSD 4n",
            "BeeGFS 2n", "BeeGFS 4n"
        ],
        "openmm+aggregate": [114.4933333, 143.0266667, 110.236, 112.9476666],
        "openmm+train": [204.6166666, 231.1243333, 212.3223333, 213.5576666],
        "openmm+inference": [126.0763333, 169.466, 121.5653333, 133.5383333]
    }
    return runtime_data

def create_spm_data():
    """Create SPM data dictionary for workflow stages"""
    # p+c and c+stage_out-c

    spm_data = {
        "store_conf": ['SSD 2n', 'SSD 4n', 'BeeGFS 2n', 'BeeGFS 4n'],
        "openmm+aggregate": [0.7711193228257445, 0.5260819556291868, 0.901756292470608, 0.8583140893892308],
        "openmm+train": [1.1722044371205718, 0.927167069924014, 2.0073399864998382, 1.963897783418461],
        "openmm+inference": [0.733508084421108, 0.4884707172245504, 0.7993784585949972, 0.7559362555136201],
    }
    return spm_data

def create_runtime_rank_dict(runtime_data):
    """Create runtime ranking dictionary"""
    df = pd.DataFrame(runtime_data)
    runtime_rank_dict = {}
    
    for col in df.columns[1:]:  # exclude 'store_conf'
        sorted_df = df[['store_conf', col]].sort_values(by=col)
        rankings = {row['store_conf']: rank + 1 for rank, row in enumerate(sorted_df.to_dict('records'))}
        runtime_rank_dict[col] = [
            {
                "store_conf": row['store_conf'],
                "runtime": row[col],
                "rank": rankings[row['store_conf']]
            }
            for _, row in df.iterrows()
        ]
    
    return runtime_rank_dict

def create_spm_rank_dict(spm_data):
    """Create SPM ranking dictionary"""
    df = pd.DataFrame(spm_data)
    
    # Helper to extract number of nodes
    def extract_nodes(conf):
        return int(conf.split()[1][:-1])
    
    # Construct dictionary
    spm_rank_dict = {}
    for column in df.columns[1:]:
        # Rank using value ascending, then node count descending
        sorted_df = df.sort_values(
            by=[column, "store_conf"],
            key=lambda col: (
                col if col.name != "store_conf" else col.map(lambda x: -extract_nodes(x))
            ),
            ascending=[True, True]
        ).reset_index()

        # Build a lookup table for ranks
        rank_lookup = {
            df.loc[i, "store_conf"]: rank + 1 for rank, i in enumerate(sorted_df["index"])
        }

        # Store spm_rank_dict per column
        spm_rank_dict[column] = [
            {
                "store_conf": row["store_conf"],
                "runtime": row[column],
                "rank": rank_lookup[row["store_conf"]]
            }
            for _, row in df.iterrows()
        ]
    
    return spm_rank_dict

def compute_rank_error(runtime_rank_dict, spm_rank_dict):
    """Compute rank error between runtime and SPM predictions"""
    stages = list(runtime_rank_dict.keys())
    num_ranks = len(runtime_rank_dict[stages[0]])  # Typically 4 for DDMD
    rank_error_table = []

    for rank_idx in range(num_ranks):  # R1 to R4
        rank_row = []
        for stage in stages:
            # Find store_conf predicted by SPM at current rank
            spm_entry = next((entry for entry in spm_rank_dict[stage] if entry["rank"] == rank_idx + 1), None)
            if not spm_entry:
                rank_row.append(None)
                continue
            predicted_conf = spm_entry["store_conf"]

            # Find actual rank of the predicted store_conf
            actual_entry = next((entry for entry in runtime_rank_dict[stage] if entry["store_conf"] == predicted_conf), None)
            if not actual_entry:
                rank_row.append(None)
                continue
            actual_rank = actual_entry["rank"]

            # Calculate absolute difference
            rank_error = abs((rank_idx + 1) - actual_rank)
            rank_row.append(rank_error)
        rank_error_table.append(rank_row)

    # Format as DataFrame
    rank_labels = [f"R{i+1}" for i in range(num_ranks)]
    df_rank_error = pd.DataFrame(rank_error_table, columns=stages, index=rank_labels)
    return df_rank_error

def compute_local_rank_deviation(runtime_rank_dict, spm_rank_dict):
    """Compute local rank deviation between runtime and SPM predictions"""
    stages = list(runtime_rank_dict.keys())
    num_ranks = len(runtime_rank_dict[stages[0]])  # Typically 4 ranks for DDMD
    penalty_table = []

    for rank_idx in range(num_ranks):  # R1 to R4
        rank_row = []
        for stage in stages:
            # Get the actual store_conf at this rank
            actual_entry = next((entry for entry in runtime_rank_dict[stage] if entry["rank"] == rank_idx + 1), None)
            if not actual_entry:
                rank_row.append(None)
                continue
            actual_conf = actual_entry["store_conf"]
            actual_runtime = actual_entry["runtime"]

            # Get the SPM-predicted rank for that same store_conf
            spm_entry = next((entry for entry in spm_rank_dict[stage] if entry["rank"] == rank_idx + 1), None)
            
            if not spm_entry:
                rank_row.append(None)
                continue
            spm_conf = spm_entry["store_conf"]
            spm_actual_entry = next((entry for entry in runtime_rank_dict[stage] if entry["store_conf"] == spm_conf), None)
            spm_runtime = spm_actual_entry["runtime"]

            # Compute penalty: can be negative or positive
            penalty_percent = 100 * (spm_runtime - actual_runtime) / actual_runtime
            rank_row.append(penalty_percent)
        penalty_table.append(rank_row)

    # Format as DataFrame
    rank_labels = [f"R{i+1}" for i in range(num_ranks)]
    df_penalty = pd.DataFrame(penalty_table, columns=stages, index=rank_labels)
    return df_penalty

def compute_global_rank_deviation(runtime_rank_dict, spm_rank_dict):
    """Compute global rank deviation between runtime and SPM predictions"""
    stages = list(runtime_rank_dict.keys())
    num_ranks = len(runtime_rank_dict[stages[0]])  # Typically 4 ranks for DDMD
    penalty_table = []

    for rank_idx in range(num_ranks):  # R1 to R4
        rank_row = []
        for stage in stages:
            # Get the actual store_conf at this rank
            actual_entry = next((entry for entry in runtime_rank_dict[stage] if entry["rank"] == rank_idx + 1), None)
            if not actual_entry:
                rank_row.append(None)
                continue
            actual_conf = actual_entry["store_conf"]
            actual_runtime = actual_entry["runtime"]

            # Get the SPM-predicted rank actual runtime of workflow with the same rank storages
            rank_runtime = 0
            for st in stages:
                entry = next((entry for entry in runtime_rank_dict[st] if entry["rank"] == rank_idx + 1), None)
                rank_runtime += entry["runtime"]
            
            # Get the SPM-predicted rank actual runtime for that same store_conf
            spm_entry = next((entry for entry in spm_rank_dict[stage] if entry["rank"] == rank_idx + 1), None)
            
            if not spm_entry:
                rank_row.append(None)
                continue
            spm_conf = spm_entry["store_conf"]
            spm_actual_entry = next((entry for entry in runtime_rank_dict[stage] if entry["store_conf"] == spm_conf), None)
            spm_runtime = spm_actual_entry["runtime"]

            # Compute penalty: can be negative or positive
            penalty_percent = 100 * (spm_runtime - actual_runtime) / rank_runtime
            rank_row.append(penalty_percent)
        penalty_table.append(rank_row)

    # Format as DataFrame
    rank_labels = [f"R{i+1}" for i in range(num_ranks)]
    df_penalty = pd.DataFrame(penalty_table, columns=stages, index=rank_labels)
    return df_penalty

def plot_rank_error_old(rank_error_table):
    """Plot SPM rank error visualization (old version - no PDF save)"""
    # Labels (workflows) and Ranks
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    ranks = ['R 1', 'R 2', 'R 3', 'R 4']

    # Convert DataFrame to list of lists (row-wise values only)
    rank_error = rank_error_table.values.tolist()

    # Optional: Display the result
    for i, row in enumerate(rank_error, start=1):
        print(f"R{i}: {row}")

    # Bar plot config
    x = np.arange(len(labels))
    bar_width = 0.09

    plt.figure(figsize=(12, 4))
    for i in range(len(ranks)):
        plt.bar(x + i * bar_width, rank_error[i], width=bar_width, color=colors[i], label=ranks[i], zorder=3)

    # Add tick for 0 bars
    for i in range(len(ranks)):
        for j, val in enumerate(rank_error[i]):
            xpos = x[j] + i * bar_width
            plt.plot(xpos, 0, marker='|', color='black', markersize=10, zorder=4)

    # Axis and legend
    plt.xticks(x + (len(ranks) - 1) * bar_width / 2, labels, fontsize=16)
    plt.yticks(np.arange(0, 10, 1), fontsize=16)
    plt.ylim(0, len(ranks)+1)
    plt.ylabel('SPM Rank Error', fontsize=16)
    plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)

    # Legend on top
    plt.legend(title='Rank', title_fontsize=16, fontsize=16, ncol=9, loc='lower center',
               bbox_to_anchor=(0.5, 0.7), columnspacing=0.5)

    plt.tight_layout()
    # No PDF save - just show the plot
    plt.show()

def calculate_rbo(runtime_rank_dict, spm_rank_dict, p=0.9, top_k=2):
    """Calculate Rank-Biased Overlap (RBO) score between runtime and SPM rankings for top-k ranks"""
    rbo_scores = {}
    
    for stage in runtime_rank_dict.keys():
        # Get runtime ranking (list of store_conf in rank order)
        runtime_ranking = []
        for entry in sorted(runtime_rank_dict[stage], key=lambda x: x['rank']):
            runtime_ranking.append(entry['store_conf'])
        
        # Get SPM ranking (list of store_conf in rank order)
        spm_ranking = []
        for entry in sorted(spm_rank_dict[stage], key=lambda x: x['rank']):
            spm_ranking.append(entry['store_conf'])
        
        # Limit to top-k rankings
        runtime_top_k = runtime_ranking[:top_k]
        spm_top_k = spm_ranking[:top_k]
        
        # Calculate RBO for top-k only
        rbo_score = 0
        for d in range(1, min(len(runtime_top_k), len(spm_top_k)) + 1):
            # Get top-d elements from both rankings
            runtime_top_d = set(runtime_top_k[:d])
            spm_top_d = set(spm_top_k[:d])
            
            # Calculate overlap
            overlap = len(runtime_top_d.intersection(spm_top_d))
            rbo_score += (p ** (d - 1)) * (overlap / d)
        
        rbo_scores[stage] = rbo_score * (1 - p)
    
    return rbo_scores

def calculate_kendall_tau(runtime_rank_dict, spm_rank_dict, top_k=2):
    """Calculate Kendall's Tau rank-order correlation for top-k ranks"""
    from scipy.stats import kendalltau
    
    tau_scores = {}
    
    for stage in runtime_rank_dict.keys():
        # Get runtime ranking (list of store_conf in rank order)
        runtime_ranking = []
        for entry in sorted(runtime_rank_dict[stage], key=lambda x: x['rank']):
            runtime_ranking.append(entry['store_conf'])
        
        # Get SPM ranking (list of store_conf in rank order)
        spm_ranking = []
        for entry in sorted(spm_rank_dict[stage], key=lambda x: x['rank']):
            spm_ranking.append(entry['store_conf'])
        
        # Limit to top-k rankings
        runtime_top_k = runtime_ranking[:top_k]
        spm_top_k = spm_ranking[:top_k]
        
        # Create rank mappings for common elements
        common_elements = set(runtime_top_k).intersection(set(spm_top_k))
        
        if len(common_elements) < 2:
            tau_scores[stage] = 0.0
            continue
        
        # Get ranks for common elements
        runtime_ranks = []
        spm_ranks = []
        
        for element in common_elements:
            runtime_ranks.append(runtime_top_k.index(element) + 1)  # 1-based ranking
            spm_ranks.append(spm_top_k.index(element) + 1)  # 1-based ranking
        
        # Calculate Kendall's Tau
        tau, _ = kendalltau(runtime_ranks, spm_ranks)
        tau_scores[stage] = tau if not np.isnan(tau) else 0.0
    
    return tau_scores

def calculate_overlap_coefficient(runtime_rank_dict, spm_rank_dict, top_k=4):
    """Calculate Overlap Coefficient for top-k ranks"""
    overlap_scores = {}
    
    for stage in runtime_rank_dict.keys():
        # Get runtime ranking (list of store_conf in rank order)
        runtime_ranking = []
        for entry in sorted(runtime_rank_dict[stage], key=lambda x: x['rank']):
            runtime_ranking.append(entry['store_conf'])
        
        # Get SPM ranking (list of store_conf in rank order)
        spm_ranking = []
        for entry in sorted(spm_rank_dict[stage], key=lambda x: x['rank']):
            spm_ranking.append(entry['store_conf'])
        
        # Limit to top-k rankings
        runtime_top_k = set(runtime_ranking[:top_k])
        spm_top_k = set(spm_ranking[:top_k])
        
        # Calculate Overlap Coefficient
        # Overlap Coefficient = |A ∩ B| / min(|A|, |B|)
        intersection = len(runtime_top_k.intersection(spm_top_k))
        min_size = min(len(runtime_top_k), len(spm_top_k))
        
        if min_size == 0:
            overlap_scores[stage] = 0.0
        else:
            overlap_scores[stage] = intersection / min_size
    
    return overlap_scores

def plot_rank_error(rank_error_table):
    """Plot SPM rank error visualization"""
    # Labels (workflows) and Ranks
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    ranks = ['R 1', 'R 2', 'R 3', 'R 4']

    # Convert DataFrame to list of lists (row-wise values only)
    rank_error = rank_error_table.values.tolist()

    # Bar plot config
    x = np.arange(len(labels))
    bar_width = 0.09

    plt.figure(figsize=(12, 4))
    
    # Plot rank errors
    for i in range(len(ranks)):
        plt.bar(x + i * bar_width, rank_error[i], width=bar_width, color=colors[i], label=ranks[i], zorder=3)

    # Add tick for 0 bars
    for i in range(len(ranks)):
        for j, val in enumerate(rank_error[i]):
            xpos = x[j] + i * bar_width
            plt.plot(xpos, 0, marker='|', color='black', markersize=10, zorder=4)

    # Axis and legend
    plt.xticks(x + (len(ranks) - 1) * bar_width / 2, labels, fontsize=16)
    plt.yticks(np.arange(0, 10, 1))
    plt.tick_params(axis='y', labelsize=16)
    plt.ylim(0, len(ranks)+1)
    plt.ylabel('SPM Rank Error', fontsize=16)
    plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)
    plt.legend(title='Rank', title_fontsize=16, fontsize=16, ncol=4, loc='lower center',
               bbox_to_anchor=(0.5, 0.7), columnspacing=0.5)

    plt.tight_layout()
    plt.savefig("ddmd/ddmd_spm_rank_error.pdf", format='pdf', bbox_inches='tight')
    plt.show()

# def plot_rbo_scores(runtime_rank_dict, spm_rank_dict):
#     """Plot RBO and Kendall's Tau correlation scores"""
#     # Labels (workflows)
#     labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']

#     # Calculate RBO scores for top-4 ranks
#     rbo_scores = calculate_rbo(runtime_rank_dict, spm_rank_dict, top_k=4)
    
#     # Calculate Kendall's Tau scores for top-4 ranks
#     tau_scores = calculate_kendall_tau(runtime_rank_dict, spm_rank_dict, top_k=4)
    
#     # Print scores
#     print("\nRank-Biased Overlap (RBO) Scores (Top-4):")
#     for stage, score in rbo_scores.items():
#         print(f"  {stage}: {score:.4f}")
    
#     print("\nKendall's Tau Scores (Top-4):")
#     for stage, score in tau_scores.items():
#         print(f"  {stage}: {score:.4f}")

#     # Plot RBO and Kendall's Tau scores
#     rbo_values = [rbo_scores[label.replace(' \n+ ', '+').replace(' ', '')] for label in labels]
#     tau_values = [tau_scores[label.replace(' \n+ ', '+').replace(' ', '')] for label in labels]
    
#     x_pos = np.arange(len(labels))
#     width = 0.35
    
#     plt.figure(figsize=(8, 6))
    
#     # Plot both metrics (RBO and Kendall's Tau)
#     bars1 = plt.bar(x_pos - width/2, rbo_values, width, label='RBO (Top-4)', color='steelblue', alpha=0.7)
#     bars2 = plt.bar(x_pos + width/2, tau_values, width, label="Kendall's τ (Top-4)", color='darkorange', alpha=0.7)
    
#     # Add value labels on bars
#     for i, (bar, value) in enumerate(zip(bars1, rbo_values)):
#         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                 f'{value:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
#     for i, (bar, value) in enumerate(zip(bars2, tau_values)):
#         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                 f'{value:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

#     # Axis setup
#     plt.xticks(x_pos, labels, fontsize=16)
#     plt.ylabel('Correlation Score', fontsize=16)
#     plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
#     plt.ylim(-1.1, 1.1)
#     plt.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)
#     plt.legend(fontsize=16, loc='upper right')
#     plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

#     plt.tight_layout()
#     plt.savefig("ddmd/ddmd_spm_rbo.pdf", format='pdf', bbox_inches='tight')
#     plt.show()

def calculate_tier_classification_accuracy(runtime_rank_dict, spm_rank_dict):
    """Calculate tier classification accuracy by grouping storage options into performance tiers"""
    tier_accuracies = {}
    
    for stage in runtime_rank_dict.keys():
        # Get runtime values and sort them
        runtime_values = []
        for entry in runtime_rank_dict[stage]:
            runtime_values.append((entry['store_conf'], entry['runtime']))
        runtime_values.sort(key=lambda x: x[1])  # Sort by runtime (ascending)
        
        # Get SPM values and sort them
        spm_values = []
        for entry in spm_rank_dict[stage]:
            spm_values.append((entry['store_conf'], entry['runtime']))
        spm_values.sort(key=lambda x: x[1])  # Sort by SPM value (ascending)
        
        # Group into tiers (high/low performance) - 2 tiers for DDMD
        n_storages = len(runtime_values)
        tier_size = n_storages // 2
        
        # Runtime tiers (lower runtime = better performance)
        runtime_tiers = {'high': [], 'low': []}
        for i, (conf, _) in enumerate(runtime_values):
            if i < tier_size:
                runtime_tiers['high'].append(conf)
            else:
                runtime_tiers['low'].append(conf)
        
        # SPM tiers (lower SPM value = better performance)
        spm_tiers = {'high': [], 'low': []}
        for i, (conf, _) in enumerate(spm_values):
            if i < tier_size:
                spm_tiers['high'].append(conf)
            else:
                spm_tiers['low'].append(conf)
        
        # Calculate accuracy for each tier
        tier_accuracy = 0
        for tier in ['high', 'low']:
            correct = len(set(runtime_tiers[tier]).intersection(set(spm_tiers[tier])))
            total = len(runtime_tiers[tier])
            tier_accuracy += correct / total if total > 0 else 0
        
        tier_accuracies[stage] = tier_accuracy / 2  # Average across 2 tiers
    
    return tier_accuracies

def calculate_tier_classification_accuracy_with_details(runtime_rank_dict, spm_rank_dict):
    """Calculate tier classification accuracy with detailed tier breakdown"""
    tier_accuracies = {}
    tier_details = {}
    
    for stage in runtime_rank_dict.keys():
        # Get runtime values and sort them
        runtime_values = []
        for entry in runtime_rank_dict[stage]:
            runtime_values.append((entry['store_conf'], entry['runtime']))
        runtime_values.sort(key=lambda x: x[1])  # Sort by runtime (ascending)
        
        # Get SPM values and sort them
        spm_values = []
        for entry in spm_rank_dict[stage]:
            spm_values.append((entry['store_conf'], entry['runtime']))
        spm_values.sort(key=lambda x: x[1])  # Sort by SPM value (ascending)
        
        # Group into tiers (high/low performance) - 2 tiers for DDMD
        n_storages = len(runtime_values)
        tier_size = n_storages // 2
        
        # Runtime tiers (lower runtime = better performance)
        runtime_tiers = {'high': [], 'low': []}
        for i, (conf, _) in enumerate(runtime_values):
            if i < tier_size:
                runtime_tiers['high'].append(conf)
            else:
                runtime_tiers['low'].append(conf)
        
        # SPM tiers (lower SPM value = better performance)
        spm_tiers = {'high': [], 'low': []}
        for i, (conf, _) in enumerate(spm_values):
            if i < tier_size:
                spm_tiers['high'].append(conf)
            else:
                spm_tiers['low'].append(conf)
        
        # Calculate accuracy for each tier
        tier_accuracy = 0
        for tier in ['high', 'low']:
            correct = len(set(runtime_tiers[tier]).intersection(set(spm_tiers[tier])))
            total = len(runtime_tiers[tier])
            tier_accuracy += correct / total if total > 0 else 0
        
        tier_accuracies[stage] = tier_accuracy / 2  # Average across 2 tiers
        tier_details[stage] = {
            'runtime_tiers': runtime_tiers,
            'spm_tiers': spm_tiers
        }
    
    return tier_accuracies, tier_details

def calculate_precision_recall_at_k(runtime_rank_dict, spm_rank_dict, k=5):
    """Calculate Precision@K and Recall@K metrics"""
    precision_scores = {}
    recall_scores = {}
    
    for stage in runtime_rank_dict.keys():
        # Get top-k from runtime ranking
        runtime_top_k = []
        for entry in sorted(runtime_rank_dict[stage], key=lambda x: x['rank']):
            if entry['rank'] <= k:
                runtime_top_k.append(entry['store_conf'])
        
        # Get top-k from SPM ranking
        spm_top_k = []
        for entry in sorted(spm_rank_dict[stage], key=lambda x: x['rank']):
            if entry['rank'] <= k:
                spm_top_k.append(entry['store_conf'])
        
        # Calculate Precision@K: fraction of SPM top-k that are in runtime top-k
        intersection = set(runtime_top_k).intersection(set(spm_top_k))
        precision = len(intersection) / len(spm_top_k) if len(spm_top_k) > 0 else 0
        
        # Calculate Recall@K: fraction of runtime top-k that are in SPM top-k
        recall = len(intersection) / len(runtime_top_k) if len(runtime_top_k) > 0 else 0
        
        precision_scores[stage] = precision
        recall_scores[stage] = recall
    
    return precision_scores, recall_scores

def _storage_type_from_conf(conf):
    conf_upper = conf.upper()
    if conf_upper.startswith('SSD') or conf_upper.startswith('TMPFS'):
        return 'Local'
    if conf_upper.startswith('BEEGFS'):
        return 'Shared'
    return 'Unknown'

def calculate_storage_type_accuracy(runtime_rank_dict, spm_rank_dict):
    """Calculate accuracy of storage type (Local vs Shared) prediction per stage across ranks."""
    type_acc = {}
    stages = list(runtime_rank_dict.keys())
    num_ranks = len(next(iter(runtime_rank_dict.values())))
    for stage in stages:
        matches = 0
        total = 0
        runtime_by_rank = {entry['rank']: entry['store_conf'] for entry in runtime_rank_dict[stage]}
        spm_by_rank = {entry['rank']: entry['store_conf'] for entry in spm_rank_dict[stage]}
        for r in range(1, num_ranks + 1):
            if r in runtime_by_rank and r in spm_by_rank:
                rt_type = _storage_type_from_conf(runtime_by_rank[r])
                spm_type = _storage_type_from_conf(spm_by_rank[r])
                matches += 1 if rt_type == spm_type else 0
                total += 1
        type_acc[stage] = (matches / total) if total > 0 else 0.0
    return type_acc

def plot_storage_type_classification_accuracy(runtime_rank_dict, spm_rank_dict):
    """Plot Local vs Shared storage type classification accuracy per stage."""
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    acc = calculate_storage_type_accuracy(runtime_rank_dict, spm_rank_dict)
    values = [acc[label.replace(' \n+ ', '+').replace(' ', '')] for label in labels]
    x_pos = np.arange(len(labels))
    plt.figure(figsize=(5, 5))
    bars = plt.bar(x_pos, values, color='steelblue', alpha=0.7)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    plt.xticks(x_pos, labels, fontsize=16)
    plt.ylabel('Storage Type Accuracy (Local vs Shared)', fontsize=16)
    plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig("ddmd/ddmd_spm_storage_type_class_acc.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_tier_classification_accuracy(runtime_rank_dict, spm_rank_dict):
    """Plot tier classification accuracy with tier breakdown details"""
    # Labels (workflows)
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    
    # Calculate tier classification accuracy and get tier details
    tier_accuracies, tier_details = calculate_tier_classification_accuracy_with_details(runtime_rank_dict, spm_rank_dict)
    
    # Print scores and tier details
    print("\nTier Classification Accuracy:")
    for stage, score in tier_accuracies.items():
        print(f"  {stage}: {score:.4f}")
    
    print("\nTier Classifications:")
    for stage, details in tier_details.items():
        print(f"\n  {stage}:")
        for tier in ['high', 'low']:
            runtime_tier = details['runtime_tiers'][tier]
            spm_tier = details['spm_tiers'][tier]
            correct = len(set(runtime_tier).intersection(set(spm_tier)))
            total = len(runtime_tier)
            print(f"    {tier.upper()} Performance:")
            print(f"      Runtime: {runtime_tier}")
            print(f"      SPM:     {spm_tier}")
            print(f"      Correct: {correct}/{total}")
    
    # Plot tier classification accuracy
    tier_values = [tier_accuracies[label.replace(' \n+ ', '+').replace(' ', '')] for label in labels]
    
    x_pos = np.arange(len(labels))
    
    plt.figure(figsize=(5, 5))
    
    # Plot tier classification accuracy
    bars = plt.bar(x_pos, tier_values, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, tier_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Axis setup
    plt.xticks(x_pos, labels, fontsize=16)
    plt.ylabel('Tier Classification Accuracy', fontsize=16)
    plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)
    # plt.title('SPM Tier Classification Accuracy\n(High/Low Performance Tiers)', fontsize=16, fontweight='bold')
    
    # Add tier classification details as text below the plot
    # tier_text = "Tier Classifications:\n"
    # for i, label in enumerate(labels):
    #     stage_key = label.replace(' \n+ ', '+').replace(' ', '')
    #     if stage_key in tier_details:
    #         details = tier_details[stage_key]
    #         tier_text += f"\n{label}:\n"
    #         for tier in ['high', 'low']:
    #             runtime_tier = details['runtime_tiers'][tier]
    #             spm_tier = details['spm_tiers'][tier]
    #             correct = len(set(runtime_tier).intersection(set(spm_tier)))
    #             total = len(runtime_tier)
    #             tier_text += f"  {tier.upper()}: Runtime={runtime_tier}, SPM={spm_tier} ({correct}/{total})\n"
    
    # Add text box with tier details
    # plt.figtext(0.02, 0.02, tier_text, fontsize=8, verticalalignment='bottom', 
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.3)  # Make room for the text box
    plt.savefig("ddmd/ddmd_spm_tier_class_acc.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_precision_recall_at_k(runtime_rank_dict, spm_rank_dict, k=5):
    """Plot Precision@K and Recall@K metrics"""
    # Labels (workflows)
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    
    # Calculate Precision@K and Recall@K
    precision_scores, recall_scores = calculate_precision_recall_at_k(runtime_rank_dict, spm_rank_dict, k)
    
    # Print scores
    print(f"\nPrecision@{k} Scores:")
    for stage, score in precision_scores.items():
        print(f"  {stage}: {score:.4f}")
    
    print(f"\nRecall@{k} Scores:")
    for stage, score in recall_scores.items():
        print(f"  {stage}: {score:.4f}")
    
    # Plot Precision@K and Recall@K
    precision_values = [precision_scores[label.replace(' \n+ ', '+').replace(' ', '')] for label in labels]
    recall_values = [recall_scores[label.replace(' \n+ ', '+').replace(' ', '')] for label in labels]
    
    x_pos = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(12, 4))
    
    # Plot both metrics
    bars1 = plt.bar(x_pos - width/2, precision_values, width, label=f'Precision@{k}', color='steelblue', alpha=0.7)
    bars2 = plt.bar(x_pos + width/2, recall_values, width, label=f'Recall@{k}', color='darkorange', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, precision_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    for i, (bar, value) in enumerate(zip(bars2, recall_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Axis setup
    plt.xticks(x_pos, labels, fontsize=16)
    plt.ylabel('Precision/Recall Score', fontsize=16)
    plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)
    plt.legend(fontsize=16, loc='upper right')
    plt.title(f'SPM Precision@{k} and Recall@{k} Metrics', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("ddmd/ddmd_spm_prec_recall_5.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_time_deviation_old(local_deviation_table, global_deviation_table):
    """Plot SPM time deviation visualization (old combined version - no PDF save)"""
    # Labels (workflows) and Ranks
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    ranks = ['R 1', 'R 2', 'R 3', 'R 4']

    # Bar and dot plot data
    local_deviation = local_deviation_table.values.tolist()
    global_deviation = global_deviation_table.values.tolist()

    x = np.arange(len(labels))
    bar_width = 0.1
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Plot bars
    for i in range(len(ranks)):
        for j, val in enumerate(local_deviation[i]):
            xpos = x[j] + i * bar_width
            label = ranks[i] if j == 0 else None  # Only label once per rank
            bar = ax1.bar(xpos, val, width=bar_width, color=colors[i], label=label, zorder=3)

            # Add text if bar > 50
            if val > 50:
                ax1.text(
                    xpos, 40, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7, rotation=90, color='white',
                    bbox=dict(facecolor='black', edgecolor='white', boxstyle='square,pad=0.8', alpha=0.6)
                )
            if val < -50:
                ax1.text(
                    xpos, -40, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7, rotation=90, color='white',
                    bbox=dict(facecolor='black', edgecolor='white', boxstyle='square,pad=0.8', alpha=0.6)
                )

    # Tick marker for 0 bars
    for i in range(len(ranks)):
        for j, val in enumerate(local_deviation[i]):
            xpos = x[j] + i * bar_width
            ax1.plot(xpos, 0, marker='|', color='black', markersize=5, zorder=4)

    # Plot dots with annotations
    for i in range(len(ranks)):
        for j, val in enumerate(global_deviation[i]):
            xpos = x[j] + i * bar_width
            if val != 0:
                ax1.scatter(
                    xpos, val,
                    color=colors[i],
                    marker='o',
                    s=30,
                    edgecolors='black',
                    linewidths=0.8,
                    zorder=6
                )

                # Add text below dot if > 50
                if val > 50:
                    ax1.text(
                        xpos, 20, f'{val:.1f}',
                        ha='center', va='top', fontsize=7, rotation=90,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.8', alpha=0.8)
                    )
                if val < -50:
                    ax1.text(
                        xpos, -20, f'{val:.1f}',
                        ha='center', va='top', fontsize=7, rotation=90,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.8', alpha=0.8)
                    )

    # Axes setup
    centered_x = x + ((len(ranks) - 1) * bar_width) / 2
    ax1.set_xticks(centered_x)
    ax1.set_xticklabels(labels, fontsize=16)
    ax1.set_yticks(np.arange(-100, 101, 10))
    ax1.set_ylim(-50, 50)
    ax1.set_ylabel('SPM Deviation Time (%)', fontsize=16)
    ax1.set_xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    ax1.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)

    # Legend
    ax1.legend(title='Rank', title_fontsize=16, fontsize=16, ncol=6, loc='upper center', 
               bbox_to_anchor=(0.45, 0.27), columnspacing=1)

    plt.tight_layout()
    # No PDF save - just show the plot
    plt.show()

    # Calculate error rates
    local_errors_10 = sum(1 for row in local_deviation for val in row if abs(val) > 10)
    global_errors_10 = sum(1 for row in global_deviation for val in row if abs(val) > 10)
    local_errors_5 = sum(1 for row in local_deviation for val in row if abs(val) > 5)
    global_errors_5 = sum(1 for row in global_deviation for val in row if abs(val) > 5)
    total_entries = len(local_deviation) * len(labels)
    
    print(f"Error rate over 10% global margin of error: {global_errors_10}/{total_entries}")
    print(f"Error rate over 10% local margin of error: {local_errors_10}/{total_entries}")
    print(f"Error rate over 5% global margin of error: {global_errors_5}/{total_entries}")
    print(f"Error rate over 5% local margin of error: {local_errors_5}/{total_entries}")

def plot_global_time_deviation(global_deviation_table):
    """Plot only global time deviation as bars"""
    # Labels (workflows) and Ranks
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    ranks = ['R 1', 'R 2', 'R 3', 'R 4']

    # Global deviation data
    global_deviation = global_deviation_table.values.tolist()

    x = np.arange(len(labels))
    bar_width = 0.1
    fig, ax1 = plt.subplots(figsize=(5, 5))

    # Plot bars for global deviation
    for i in range(len(ranks)):
        for j, val in enumerate(global_deviation[i]):
            xpos = x[j] + i * bar_width
            label = ranks[i] if j == 0 else None  # Only label once per rank
            bar = ax1.bar(xpos, val, width=bar_width, color=colors[i], label=label, zorder=3)

    # Tick marker for 0 bars
    for i in range(len(ranks)):
        for j, val in enumerate(global_deviation[i]):
            xpos = x[j] + i * bar_width
            ax1.plot(xpos, 0, marker='|', color='black', markersize=5, zorder=4)

    # Axes setup
    centered_x = x + ((len(ranks) - 1) * bar_width) / 2
    ax1.set_xticks(centered_x)
    ax1.set_xticklabels(labels, fontsize=16)
    ax1.set_yticks(np.arange(-12, 12, 2))
    ax1.set_ylim(-12, 12)
    ax1.set_ylabel('SPM Deviation Time (%)', fontsize=16)
    ax1.set_xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    ax1.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0)

    # Legend
    ax1.legend(title='Rank', title_fontsize=15, fontsize=15, ncol=2, loc='upper center', 
               bbox_to_anchor=(0.5, 0.3), columnspacing=1)

    plt.tight_layout()
    plt.savefig("ddmd/ddmd_spm_time_deviation.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def print_percent_deviation(global_deviation_table):
    """Print percent deviation for each producer+consumer stage"""
    print("\n" + "="*60)
    print("PERCENT DEVIATION BY PRODUCER+CONSUMER STAGE")
    print("="*60)
    
    # Labels (workflows) and Ranks
    labels = ['openmm+aggregate', 'openmm+train', 'openmm+inference']
    ranks = ['R1', 'R2', 'R3', 'R4']
    
    # Convert DataFrame to list of lists
    global_deviation = global_deviation_table.values.tolist()
    
    # Print header
    print(f"{'Stage':<20}", end="")
    for rank in ranks:
        print(f"{rank:>8}", end="")
    print()
    print("-" * (20 + 8 * len(ranks)))
    
    # Print each stage's deviations
    for j, stage in enumerate(labels):
        print(f"{stage:<20}", end="")
        for i in range(len(ranks)):
            deviation = global_deviation[i][j]
            if deviation is not None:
                print(f"{deviation:>7.1f}%", end="")
            else:
                print(f"{'N/A':>7}", end="")
        print()
    
    print()
    
    # Calculate and print summary statistics
    print("SUMMARY STATISTICS:")
    print("-" * 30)
    
    # Calculate statistics for each stage
    for j, stage in enumerate(labels):
        deviations = [global_deviation[i][j] for i in range(len(ranks)) if global_deviation[i][j] is not None]
        if deviations:
            mean_dev = np.mean(deviations)
            std_dev = np.std(deviations)
            max_dev = np.max(deviations)
            min_dev = np.min(deviations)
            abs_deviations = [abs(d) for d in deviations]
            mean_abs_dev = np.mean(abs_deviations)
            
            print(f"{stage}:")
            print(f"  Mean deviation: {mean_dev:.2f}%")
            print(f"  Mean absolute deviation: {mean_abs_dev:.2f}%")
            print(f"  Standard deviation: {std_dev:.2f}%")
            print(f"  Range: [{min_dev:.2f}%, {max_dev:.2f}%]")
            print()

def plot_spm_values(spm_data):
    """Plot SPM values for SSD and BeeGFS only"""
    # Filter for SSD and BeeGFS only (indices 0,1 for SSD and 2,3 for BeeGFS)
    ssd_indices = [0, 1]  # SSD 2n, SSD 4n
    beegfs_indices = [2, 3]  # BeeGFS 2n, BeeGFS 4n
    
    # Labels for producer-consumer pairs
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    
    # Storage configurations to plot
    storage_configs = ['SSD 2n', 'SSD 4n', 'BeeGFS 2n', 'BeeGFS 4n']
    storage_indices = ssd_indices + beegfs_indices
    
    x = np.arange(len(labels))
    bar_width = 0.18
    
    plt.figure(figsize=(12, 4))
    
    # Plot bars for each storage configuration
    for i, (config, idx) in enumerate(zip(storage_configs, storage_indices)):
        values = []
        for stage in labels:
            # Get the SPM value for this stage and storage configuration
            stage_key = stage.replace(' \n+ ', '+').replace(' ', '')
            if stage_key in spm_data:
                values.append(spm_data[stage_key][idx])
            else:
                values.append(0)
        
        plt.bar(x + i * bar_width, values, width=bar_width, 
                label=config, color=colors[i], alpha=0.8)
    
    # Customize the plot
    plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    plt.ylabel('Rank Score', fontsize=16)
    plt.title('Rank Score by Storage Configuration (SSD and BeeGFS)', fontsize=16)
    plt.xticks(x + (len(storage_configs) - 1) * bar_width / 2, labels, fontsize=16)
    plt.legend(fontsize=16, ncol=2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig("ddmd/ddmd_spm_rank.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_spm_values_log(spm_data):
    """Plot SPM values for SSD and BeeGFS only with log scale"""
    # Filter for SSD and BeeGFS only (indices 0,1 for SSD and 2,3 for BeeGFS)
    ssd_indices = [0, 1]  # SSD 2n, SSD 4n
    beegfs_indices = [2, 3]  # BeeGFS 2n, BeeGFS 4n
    
    # Labels for producer-consumer pairs
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    
    # Storage configurations to plot
    storage_configs = ['SSD 2n', 'SSD 4n', 'BeeGFS 2n', 'BeeGFS 4n']
    storage_indices = ssd_indices + beegfs_indices
    
    x = np.arange(len(labels))
    bar_width = 0.18
    
    plt.figure(figsize=(12, 4))
    
    # Plot bars for each storage configuration
    for i, (config, idx) in enumerate(zip(storage_configs, storage_indices)):
        values = []
        for stage in labels:
            # Get the SPM value for this stage and storage configuration
            stage_key = stage.replace(' \n+ ', '+').replace(' ', '')
            if stage_key in spm_data:
                values.append(spm_data[stage_key][idx])
            else:
                values.append(0)
        
        plt.bar(x + i * bar_width, values, width=bar_width, 
                label=config, color=colors[i], alpha=0.8)
    
    # Customize the plot with log scale
    plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    plt.ylabel('Rank Score (log)', fontsize=16)
    plt.title('Rank Score by Storage Configuration (SSD and BeeGFS) - Log Scale', fontsize=16)
    plt.xticks(x + (len(storage_configs) - 1) * bar_width / 2, labels, fontsize=16)
    plt.legend(fontsize=16, ncol=2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.yscale('log')  # Set y-axis to log scale
    
    plt.tight_layout()
    plt.savefig("ddmd/ddmd_spm_rank_log.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_workflow_time(runtime_data):
    """Plot workflow time for SSD and BeeGFS only"""
    # Filter for SSD and BeeGFS only (indices 0,1 for SSD and 2,3 for BeeGFS)
    ssd_indices = [0, 1]  # SSD 2n, SSD 4n
    beegfs_indices = [2, 3]  # BeeGFS 2n, BeeGFS 4n
    
    # Labels for producer-consumer pairs
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    
    # Storage configurations to plot
    storage_configs = ['SSD 2n', 'SSD 4n', 'BeeGFS 2n', 'BeeGFS 4n']
    storage_indices = ssd_indices + beegfs_indices
    
    x = np.arange(len(labels))
    bar_width = 0.18
    
    plt.figure(figsize=(12, 4))
    
    # Plot bars for each storage configuration
    for i, (config, idx) in enumerate(zip(storage_configs, storage_indices)):
        values = []
        for stage in labels:
            # Get the runtime value for this stage and storage configuration
            stage_key = stage.replace(' \n+ ', '+').replace(' ', '')
            if stage_key in runtime_data:
                values.append(runtime_data[stage_key][idx])
            else:
                values.append(0)
        
        plt.bar(x + i * bar_width, values, width=bar_width, 
                label=config, color=colors[i], alpha=0.8)
    
    # Customize the plot
    plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    plt.ylabel('Runtime (seconds)', fontsize=16)
    plt.title('Workflow Time by Storage Configuration (SSD and BeeGFS)', fontsize=16)
    plt.xticks(x + (len(storage_configs) - 1) * bar_width / 2, labels, fontsize=16)
    plt.legend(fontsize=16, ncol=2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig("ddmd/ddmd_wf_time.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_workflow_time_log(runtime_data):
    """Plot workflow time for SSD and BeeGFS only with log scale"""
    # Filter for SSD and BeeGFS only (indices 0,1 for SSD and 2,3 for BeeGFS)
    ssd_indices = [0, 1]  # SSD 2n, SSD 4n
    beegfs_indices = [2, 3]  # BeeGFS 2n, BeeGFS 4n
    
    # Labels for producer-consumer pairs
    labels = ['openmm \n+ aggregate', 'openmm \n+ train', 'openmm \n+ inference']
    
    # Storage configurations to plot
    storage_configs = ['SSD 2n', 'SSD 4n', 'BeeGFS 2n', 'BeeGFS 4n']
    storage_indices = ssd_indices + beegfs_indices
    
    x = np.arange(len(labels))
    bar_width = 0.18
    
    plt.figure(figsize=(12, 4))
    
    # Plot bars for each storage configuration
    for i, (config, idx) in enumerate(zip(storage_configs, storage_indices)):
        values = []
        for stage in labels:
            # Get the runtime value for this stage and storage configuration
            stage_key = stage.replace(' \n+ ', '+').replace(' ', '')
            if stage_key in runtime_data:
                values.append(runtime_data[stage_key][idx])
            else:
                values.append(0)
        
        plt.bar(x + i * bar_width, values, width=bar_width, 
                label=config, color=colors[i], alpha=0.8)
    
    # Customize the plot with log scale
    plt.xlabel('DeepDriveMD Producer-Consumer Pairs', fontsize=16)
    plt.ylabel('Runtime (seconds, log)', fontsize=16)
    plt.title('Workflow Time by Storage Configuration (SSD and BeeGFS) - Log Scale', fontsize=16)
    plt.xticks(x + (len(storage_configs) - 1) * bar_width / 2, labels, fontsize=16)
    plt.legend(fontsize=16, ncol=2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.yscale('log')  # Set y-axis to log scale
    
    plt.tight_layout()
    plt.savefig("ddmd/ddmd_wf_time_log.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def calculate_total_workflow_time(runtime_data, spm_rank_dict):
    """Calculate total workflow time using best ranked SPM storage selection"""
    print("\n" + "="*60)
    print("TOTAL WORKFLOW TIME CALCULATION")
    print("="*60)
    
    # Get the best ranked storage configuration for each stage
    best_storages = {}
    total_time = 0
    
    for stage, entries in spm_rank_dict.items():
        # Find the entry with rank 1 (best rank)
        best_entry = next((entry for entry in entries if entry["rank"] == 1), None)
        if best_entry:
            best_storage = best_entry["store_conf"]
            best_storages[stage] = best_storage
            
            # Get the corresponding runtime value
            runtime_value = runtime_data[stage][runtime_data["store_conf"].index(best_storage)]
            total_time += runtime_value
            
            print(f"{stage}: {best_storage} -> {runtime_value:.2f} seconds")
    
    print(f"\nTotal workflow time: {total_time:.2f} seconds")
    print(f"Best storage configuration: {best_storages}")
    
    return total_time, best_storages

def calculate_total_workflow_time_primary_selection(runtime_data, spm_data):
    """Calculate total workflow time using primary storage selection based on largest SPM stage"""
    print("\n" + "="*60)
    print("TOTAL WORKFLOW TIME CALCULATION (PRIMARY SELECTION METHOD)")
    print("="*60)
    
    # Find the stage with the largest SPM value
    max_spm_value = 0
    max_spm_stage = None
    
    for stage in spm_data.keys():
        if stage == "store_conf":
            continue
        for spm_value in spm_data[stage]:
            if spm_value > max_spm_value:
                max_spm_value = spm_value
                max_spm_stage = stage
    
    print(f"Stage with largest SPM: {max_spm_stage}")
    print(f"Largest SPM value: {max_spm_value:.6f}")
    
    # Find the storage configuration with the smallest SPM value for this stage
    min_spm_value = float('inf')
    min_spm_storage = None
    
    for i, spm_value in enumerate(spm_data[max_spm_stage]):
        if spm_value < min_spm_value:
            min_spm_value = spm_value
            min_spm_storage = spm_data["store_conf"][i]
    
    print(f"Smallest SPM value for {max_spm_stage}: {min_spm_value:.6f}")
    print(f"Primary storage selection: {min_spm_storage}")
    
    # Apply the primary storage selection to all stages
    total_time = 0
    primary_storage_index = spm_data["store_conf"].index(min_spm_storage)
    
    print(f"\nApplying {min_spm_storage} to all stages:")
    for stage in runtime_data.keys():
        if stage == "store_conf":
            continue
        runtime_value = runtime_data[stage][primary_storage_index]
        total_time += runtime_value
        print(f"{stage}: {min_spm_storage} -> {runtime_value:.2f} seconds")
    
    print(f"\nTotal workflow time (primary selection): {total_time:.2f} seconds")
    print(f"Primary storage configuration: {min_spm_storage}")
    
    return total_time, min_spm_storage

def calculate_beeGFS_least_nodes_runtime(runtime_data):
    """Calculate total workflow runtime using default BeeGFS with least node configuration"""
    print("\n" + "="*60)
    print("BEEGFS LEAST NODES RUNTIME CALCULATION")
    print("="*60)
    
    # For DeepDriveMD: BeeGFS 2n is the least node configuration (index 2)
    beeGFS_least_nodes = "BeeGFS 2n"
    beeGFS_index = runtime_data["store_conf"].index(beeGFS_least_nodes)
    
    total_time = 0
    print(f"Using {beeGFS_least_nodes} for all stages:")
    
    for stage in runtime_data.keys():
        if stage == "store_conf":
            continue
        runtime_value = runtime_data[stage][beeGFS_index]
        total_time += runtime_value
        print(f"{stage}: {beeGFS_least_nodes} -> {runtime_value:.2f} seconds")
    
    print(f"\nTotal workflow runtime with {beeGFS_least_nodes}: {total_time:.2f} seconds")
    
    return total_time

def main():
    """Main function to run the SPM analysis"""
    # Create output directory
    output_dir = "ddmd"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print("Creating runtime data...")
    runtime_data = create_runtime_data()
    
    print("Creating SPM data...")
    spm_data = create_spm_data()
    
    print("Computing runtime rankings...")
    runtime_rank_dict = create_runtime_rank_dict(runtime_data)
    
    print("Computing SPM rankings...")
    spm_rank_dict = create_spm_rank_dict(spm_data)
    
    # Display runtime rankings
    print("\nRuntime Rankings:")
    print(json.dumps(runtime_rank_dict, indent=2))
    
    # Display SPM rankings
    print("\nSPM Rankings:")
    print(json.dumps(spm_rank_dict, indent=2))
    
    print("\nComputing rank errors...")
    rank_error_table = compute_rank_error(runtime_rank_dict, spm_rank_dict)
    print(rank_error_table)
    
    print("\nComputing local rank deviations...")
    local_deviation_table = compute_local_rank_deviation(runtime_rank_dict, spm_rank_dict)
    print(local_deviation_table)
    
    print("\nComputing global rank deviations...")
    global_deviation_table = compute_global_rank_deviation(runtime_rank_dict, spm_rank_dict)
    print(global_deviation_table)
    
    print("\nGenerating visualizations...")
    plot_rank_error_old(rank_error_table)
    plot_rank_error(rank_error_table)
    # plot_rbo_scores(runtime_rank_dict, spm_rank_dict)  # Commented out RBO and Kendall's Tau
    plot_tier_classification_accuracy(runtime_rank_dict, spm_rank_dict)
    plot_storage_type_classification_accuracy(runtime_rank_dict, spm_rank_dict)
    plot_precision_recall_at_k(runtime_rank_dict, spm_rank_dict, k=5)
    plot_time_deviation_old(local_deviation_table, global_deviation_table)
    plot_global_time_deviation(global_deviation_table)
    print_percent_deviation(global_deviation_table)
    
    print("\nGenerating additional plots...")
    plot_spm_values(spm_data)
    plot_spm_values_log(spm_data)
    plot_workflow_time(runtime_data)
    plot_workflow_time_log(runtime_data)
    
    print("\nCalculating total workflow time...")
    total_time, best_storages = calculate_total_workflow_time(runtime_data, spm_rank_dict)
    
    print("\nCalculating total workflow time (primary selection method)...")
    total_time_primary, primary_storage = calculate_total_workflow_time_primary_selection(runtime_data, spm_data)
    
    print("\nCalculating BeeGFS least nodes runtime...")
    beeGFS_runtime = calculate_beeGFS_least_nodes_runtime(runtime_data)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
