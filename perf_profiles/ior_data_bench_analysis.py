#!/usr/bin/env python3
"""
IOR Data Benchmark Analysis

This script mirrors the move_data_bench_analysis workflow but for IOR data.
It constructs a 6D DataFrame with dimensions:
  - operation, aggregateFilesizeMB, numTasks, numNodes, storageType, transferSize

And generates the requested plots:
  - Fix transferSize to 4KB and numNodes to 2
  - Plot Transfer Rate (MB/s) vs. tasksPerNode
  - Subplots for SSD write, SSD read, BeeGFS write, BeeGFS read
  - Legend shows different Total I/O (aggregateFilesizeMB)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from ior_utils import (
    collect_ior_data,
    save_master_ior_df,
    IOR_PARAMS,
)


def ensure_plots_dir(plots_dir: str) -> str:
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")
    return plots_dir


def build_master_df(data_dir: str) -> pd.DataFrame:
    """
    Build the 6D IOR master DataFrame from JSON files in data_dir.
    Returns a DataFrame with columns defined by IOR_PARAMS.
    """
    df = collect_ior_data(data_dir)
    if df.empty:
        print("Warning: No IOR data found. DataFrame is empty.")
        return df

    # Keep only required columns and order them
    for col in IOR_PARAMS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[IOR_PARAMS].copy()

    # Basic cleanup: drop rows with non-positive throughput
    before = len(df)
    df = df[df["trMiB"] > 0].copy()
    after = len(df)
    if after != before:
        print(f"Dropped {before - after} rows with non-positive trMiB")

    return df


def plot_tr_vs_tasks_per_node_by_filesize(
    df: pd.DataFrame,
    transfer_size_bytes: int,
    num_nodes: int,
    plots_dir: str,
    filename_prefix: str = None,
    max_series: int | None = None,
    show_error_bars: bool = False,
):
    """
    Create a 2x2 subplot figure with:
      - (0,0) SSD write
      - (0,1) SSD read
      - (1,0) BeeGFS write
      - (1,1) BeeGFS read

    X-axis: tasksPerNode, Y-axis: trMiB
    Legend: aggregateFilesizeMB values
    """
    # Filter by fixed transfer size and nodes
    subset = df[(df["transferSize"] == transfer_size_bytes) & (df["numNodes"] == num_nodes)]
    print(f"Filtered subset has {len(subset)} records")
    if subset.empty:
        print("No data after filtering by transferSize and numNodes. Skipping plot.")
        return

    # Create filename prefix based on actual parameters
    if filename_prefix is None:
        transfer_size_kb = transfer_size_bytes // 1024
        filename_prefix = f"ior_tr_vs_tpn_{transfer_size_kb}kb_{num_nodes}n"

    panels = [
        ("ssd", "write", "SSD write"),
        ("ssd", "read", "SSD read"),
        ("beegfs", "write", "BeeGFS write"),
        ("beegfs", "read", "BeeGFS read"),
    ]
    for storage, op, title in panels:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))

        df_panel = subset[(subset["storageType"] == storage) & (subset["operation"] == op)]
        print(f"{title}: {len(df_panel)} records")
        if df_panel.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            out_path = os.path.join(plots_dir, f"{filename_prefix}_{storage}_{op}.pdf")
            plt.savefig(out_path, format="pdf", bbox_inches="tight")
            print(f"Saved plot: {out_path}")
            plt.close(fig)
            continue

        # Sort legend labels consistently
        sizes_all = sorted(df_panel["aggregateFilesizeMB"].unique())
        sizes = sizes_all[:max_series] if (max_series is not None) else sizes_all
        print(f"\n{title} - Unique Total I/O sizes (MB): {sizes_all}")
        if max_series is not None and len(sizes_all) > max_series:
            print(f"{title} - Plotting first {max_series} series: {sizes}")

        # Color map for sizes
        cmap = plt.get_cmap("tab10" if len(sizes) <= 10 else "tab20")
        color_map: Dict[int, tuple] = {sz: cmap(i % cmap.N) for i, sz in enumerate(sizes)}

        for sz in sizes:
            block = df_panel[df_panel["aggregateFilesizeMB"] == sz]
            if block.empty:
                continue
            
            # Group by tasksPerNode and calculate mean and std
            grouped = block.groupby('tasksPerNode')['trMiB'].agg(['mean', 'std', 'count']).reset_index()
            grouped = grouped.sort_values('tasksPerNode')
            
            # Calculate standard error (std / sqrt(n))
            grouped['std_error'] = grouped['std'] / np.sqrt(grouped['count'])
            
            # Convert MiB/s to GB/s (1 MiB = 1.048576 MB, 1 GB = 1000 MB)
            # So 1 MiB/s = 1.048576/1000 GB/s = 0.001048576 GB/s
            mean_gb_s = grouped['mean'] * 0.001048576
            std_error_gb_s = grouped['std_error'] * 0.001048576
            
            # Plot with or without error bars
            if show_error_bars:
                ax.errorbar(
                    grouped['tasksPerNode'],
                    mean_gb_s,
                    yerr=std_error_gb_s,
                    label=f"{int(sz/1024)} GB",
                    color=color_map[sz],
                    marker='o',
                    capsize=5,
                    capthick=2,
                    linewidth=2,
                    markersize=6
                )
            else:
                ax.plot(
                    grouped['tasksPerNode'],
                    mean_gb_s,
                    label=f"{int(sz/1024)} GB",
                    color=color_map[sz],
                    marker='o',
                    linewidth=2,
                    markersize=6
                )
        
        # x-axis to log
        # ax.set_xscale('log')
        ax.set_xlabel("tasksPerNode", fontsize=15)
        

        ax.set_ylabel("Transfer Rate (GB/s)", fontsize=15)
        # ax.set_title(f"{title}  (4KB, 2 nodes)")
        ax.grid(True, alpha=0.3)
        
        # Set tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        
        # Create custom legend with only markers (no error bars)
        handles, labels = ax.get_legend_handles_labels()
        # Create new handles with only markers
        new_handles = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, handle in enumerate(handles):
            # Get color from the errorbar container
            if hasattr(handle, 'get_color'):
                color = handle.get_color()
            else:
                # For ErrorbarContainer, get color from the line
                color = handle[0].get_color()
            
            # Create a new line2D object with only the marker
            new_handle = plt.Line2D([0], [0], marker='o', linestyle='None', 
                                   color=color, markersize=8)
            new_handles.append(new_handle)
        
        ax.legend(new_handles, labels, title="Total I/O (GB)", fontsize=15, title_fontsize=15, framealpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"{filename_prefix}_{storage}_{op}.pdf")
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        print(f"Saved plot: {out_path}")
        plt.close(fig)


def plot_tr_vs_tasks_per_node_by_transfer_size(
    df: pd.DataFrame,
    aggregate_filesize_mb: int,
    num_nodes: int,
    plots_dir: str,
    filename_prefix: str = None,
    max_series: int | None = None,
    show_error_bars: bool = False,
):
    """
    Create 4 separate plots for:
      - SSD write
      - SSD read  
      - BeeGFS write
      - BeeGFS read

    X-axis: tasksPerNode, Y-axis: trMiB
    Legend: transferSize values (in B, KB, MB)
    """
    # Filter by fixed aggregate file size and nodes
    subset = df[(df["aggregateFilesizeMB"] == aggregate_filesize_mb) & (df["numNodes"] == num_nodes)]
    print(f"Filtered subset has {len(subset)} records")
    if subset.empty:
        print("No data after filtering by aggregateFilesizeMB and numNodes. Skipping plot.")
        return

    # Create filename prefix based on actual parameters
    if filename_prefix is None:
        filename_prefix = f"ior_tr_vs_tpn_{aggregate_filesize_mb}mb_{num_nodes}n"

    panels = [
        ("ssd", "write", "SSD write"),
        ("ssd", "read", "SSD read"),
        ("beegfs", "write", "BeeGFS write"),
        ("beegfs", "read", "BeeGFS read"),
    ]
    
    for storage, op, title in panels:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))

        df_panel = subset[(subset["storageType"] == storage) & (subset["operation"] == op)]
        print(f"{title}: {len(df_panel)} records")
        if df_panel.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            out_path = os.path.join(plots_dir, f"{filename_prefix}_{storage}_{op}.pdf")
            plt.savefig(out_path, format="pdf", bbox_inches="tight")
            print(f"Saved plot: {out_path}")
            plt.close(fig)
            continue

        # Sort legend labels consistently
        transfer_sizes_all = sorted(df_panel["transferSize"].unique())
        transfer_sizes = transfer_sizes_all[:max_series] if (max_series is not None) else transfer_sizes_all
        print(f"\n{title} - Unique Transfer Sizes (bytes): {transfer_sizes_all}")
        if max_series is not None and len(transfer_sizes_all) > max_series:
            print(f"{title} - Plotting first {max_series} series: {transfer_sizes}")

        # Color map for transfer sizes
        cmap = plt.get_cmap("tab10" if len(transfer_sizes) <= 10 else "tab20")
        color_map: Dict[int, tuple] = {ts: cmap(i % cmap.N) for i, ts in enumerate(transfer_sizes)}

        for ts in transfer_sizes:
            block = df_panel[df_panel["transferSize"] == ts]
            if block.empty:
                continue
            
            # Group by tasksPerNode and calculate mean and std
            grouped = block.groupby('tasksPerNode')['trMiB'].agg(['mean', 'std', 'count']).reset_index()
            grouped = grouped.sort_values('tasksPerNode')
            
            # Calculate standard error (std / sqrt(n))
            grouped['std_error'] = grouped['std'] / np.sqrt(grouped['count'])
            
            # Format transfer size for legend
            if ts < 1024:
                ts_label = f"{ts} B"
            elif ts < 1024 * 1024:
                ts_label = f"{ts // 1024} KB"
            else:
                ts_label = f"{ts // (1024 * 1024)} MB"
            
            # Convert MiB/s to GB/s (1 MiB = 1.048576 MB, 1 GB = 1000 MB)
            # So 1 MiB/s = 1.048576/1000 GB/s = 0.001048576 GB/s
            mean_gb_s = grouped['mean'] * 0.001048576
            std_error_gb_s = grouped['std_error'] * 0.001048576
            
            # Plot with or without error bars
            if show_error_bars:
                ax.errorbar(
                    grouped['tasksPerNode'],
                    mean_gb_s,
                    yerr=std_error_gb_s,
                    label=f"TS {ts_label}",
                    color=color_map[ts],
                    marker='o',
                    capsize=5,
                    capthick=2,
                    linewidth=2,
                    markersize=6
                )
            else:
                ax.plot(
                    grouped['tasksPerNode'],
                    mean_gb_s,
                    label=f"TS {ts_label}",
                    color=color_map[ts],
                    marker='o',
                    linewidth=2,
                    markersize=6
                )
        
        # Set y-axis limit to 1.5x the maximum y value
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.8)
        
        ax.set_xlabel("tasksPerNode", fontsize=15)
        ax.set_ylabel("Transfer Rate (GB/s)", fontsize=15)
        ax.grid(True, alpha=0.3)
        
        # Set tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        
        # Create custom legend with only markers (no error bars)
        handles, labels = ax.get_legend_handles_labels()
        # Create new handles with only markers
        new_handles = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, handle in enumerate(handles):
            # Get color from the errorbar container
            if hasattr(handle, 'get_color'):
                color = handle.get_color()
            else:
                # For ErrorbarContainer, get color from the line
                color = handle[0].get_color()
            
            # Create a new line2D object with only the marker
            new_handle = plt.Line2D([0], [0], marker='o', linestyle='None', 
                                   color=color, markersize=8)
            new_handles.append(new_handle)
        
        ax.legend(new_handles, labels, title="Transfer Size", fontsize=14, title_fontsize=15, framealpha=0.3, 
                 ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.05), columnspacing=0.5)

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"{filename_prefix}_{storage}_{op}.pdf")
        plt.savefig(out_path, format="pdf", bbox_inches="tight")
        print(f"Saved plot: {out_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="IOR Data Benchmark Analysis")
    parser.add_argument("--data-dir", type=str, default="./ior_data", help="Directory containing IOR JSON data")
    parser.add_argument("--plots-dir", type=str, default="./ior_plots", help="Directory to save plots")
    parser.add_argument("--save-master", action="store_true", help="Save master IOR DataFrame to CSV")
    parser.add_argument("--max-series", type=int, default=None, help="Max number of legend series to plot (e.g., 3)")
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes to plot")
    parser.add_argument("--show-error-bars", action="store_true", help="Show error bars on plots")
    args = parser.parse_args()

    plots_dir = ensure_plots_dir(args.plots_dir)

    if args.num_nodes is None:
        num_nodes = 2
    else:
        num_nodes = args.num_nodes

    # Build DataFrame
    df = build_master_df(args.data_dir)
    if df.empty:
        return

    # Optionally save master CSV
    if args.save_master:
        save_master_ior_df(df, os.path.join(os.path.dirname(plots_dir), "master_ior_df.csv"))

    # Required plot: 4KB, 2 nodes
    transfer_size_bytes = 4 * 1024  # 4KB
    
    # Debug: Check what node counts are available in the data
    print(f"\nAvailable node counts in data: {sorted(df['numNodes'].unique())}")
    print(f"Available transfer sizes in data: {sorted(df['transferSize'].unique())}")
    
    # Check if we have data for the requested parameters
    subset = df[(df["transferSize"] == transfer_size_bytes) & (df["numNodes"] == num_nodes)]
    print(f"Data points for transferSize={transfer_size_bytes} and numNodes={num_nodes}: {len(subset)}")
    
    # Debug: Check data types and exact values
    print(f"transfer_size_bytes type: {type(transfer_size_bytes)}, value: {transfer_size_bytes}")
    print(f"num_nodes type: {type(num_nodes)}, value: {num_nodes}")
    print(f"Data transferSize types: {df['transferSize'].dtype}")
    print(f"Data numNodes types: {df['numNodes'].dtype}")
    
    # Check for 4KB specifically
    four_kb_data = df[df["transferSize"] == 4096]
    print(f"Records with transferSize=4096: {len(four_kb_data)}")
    if len(four_kb_data) > 0:
        print(f"Node counts for 4KB data: {sorted(four_kb_data['numNodes'].unique())}")
    
    if subset.empty:
        print(f"No data found for transferSize={transfer_size_bytes} and numNodes={num_nodes}")
        print("Available combinations:")
        for ts in sorted(df['transferSize'].unique()):
            for nn in sorted(df['numNodes'].unique()):
                count = len(df[(df["transferSize"] == ts) & (df["numNodes"] == nn)])
                if count > 0:
                    print(f"  transferSize={ts}, numNodes={nn}: {count} records")
    
    plot_tr_vs_tasks_per_node_by_filesize(
        df,
        transfer_size_bytes,
        num_nodes,
        plots_dir,
        max_series=3, # plot first 3 series
        show_error_bars=args.show_error_bars,
    )
    
    # Additional plots: Fix total I/O to 1GB, vary transfer size
    print("\n" + "="*50)
    print("Generating additional plots: Transfer Size Analysis")
    print("="*50)
    
    aggregate_filesize_mb = 1024  # 1GB
    num_nodes = 4
    print(f"Fixed aggregate file size: {aggregate_filesize_mb} MB (1GB)")
    print(f"Fixed number of nodes: {num_nodes}")
    
    # Check if we have data for this combination
    subset_ts = df[(df["aggregateFilesizeMB"] == aggregate_filesize_mb) & (df["numNodes"] == num_nodes)]
    print(f"Data points for aggregateFilesizeMB={aggregate_filesize_mb} and numNodes={num_nodes}: {len(subset_ts)}")
    
    if not subset_ts.empty:
        plot_tr_vs_tasks_per_node_by_transfer_size(
            df,
            aggregate_filesize_mb,
            num_nodes,
            plots_dir,
            max_series=6, # plot first 3 series
            show_error_bars=args.show_error_bars,
        )
    else:
        print("No data found for transfer size analysis. Skipping additional plots.")


if __name__ == "__main__":
    main()


