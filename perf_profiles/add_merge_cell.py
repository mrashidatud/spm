#!/usr/bin/env python3
"""
Script to add merge functionality to move_data_bench_analysis.ipynb
"""

import json
import os

def add_merge_cell_to_notebook():
    """Add a merge cell to the move_data_bench_analysis.ipynb notebook."""
    
    notebook_file = 'move_data_bench_analysis.ipynb'
    
    if not os.path.exists(notebook_file):
        print(f"Error: {notebook_file} not found.")
        return
    
    # Read the existing notebook
    with open(notebook_file, 'r') as f:
        notebook = json.load(f)
    
    # Create the merge cell content
    merge_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Save the cp_data DataFrame to CSV\n",
            "data_df.to_csv(\"master_move_data.csv\", index=False)\n",
            "print(f\"CP data saved to master_move_data.csv with shape: {data_df.shape}\")\n",
            "\n",
            "# Load the master IOR dataframe\n",
            "ior_df = pd.read_csv('master_ior_df.csv')\n",
            "print(f\"Loaded master IOR DataFrame with shape: {ior_df.shape}\")\n",
            "\n",
            "# Define keys used for joining\n",
            "merge_keys = ['operation', 'randomOffset', 'transferSize', 'aggregateFilesizeMB',\n",
            "              'numTasks', 'numNodes', 'tasksPerNode', 'parallelism']\n",
            "\n",
            "# Make sure merge keys have the same dtype in both DataFrames (convert to str here)\n",
            "for key in merge_keys:\n",
            "    ior_df[key] = ior_df[key].astype(str)\n",
            "    data_df[key] = data_df[key].astype(str)\n",
            "\n",
            "# Perform the outer merge (keep all rows and columns from both DataFrames)\n",
            "merged_df = pd.merge(ior_df, data_df, on=merge_keys, how='outer')\n",
            "\n",
            "# Save the merged DataFrame\n",
            "merged_df.to_csv('updated_master_ior_df.csv', index=False)\n",
            "\n",
            "# Show final shape and new columns\n",
            "print(\"Merged DataFrame shape:\", merged_df.shape)\n",
            "print(\"New columns added:\", set(merged_df.columns) - set(ior_df.columns))\n",
            "print(\"Updated master DataFrame saved to: updated_master_ior_df.csv\")\n",
            "\n",
            "# Display summary of the merged data\n",
            "print(f\"\\nSummary of merged data:\")\n",
            "print(f\"Original IOR records: {len(ior_df)}\")\n",
            "print(f\"CP data records: {len(data_df)}\")\n",
            "print(f\"Merged records: {len(merged_df)}\")\n",
            "print(f\"Storage types in merged data: {sorted(merged_df['storageType'].dropna().unique())}\")\n"
        ]
    }
    
    # Add the merge cell to the notebook
    notebook['cells'].append(merge_cell)
    
    # Save the updated notebook
    with open(notebook_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Successfully added merge cell to {notebook_file}")
    print("You can now run the notebook to merge cp_data with master IOR DataFrame.")

if __name__ == "__main__":
    add_merge_cell_to_notebook()