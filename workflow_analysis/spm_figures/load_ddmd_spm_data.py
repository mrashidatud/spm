#!/usr/bin/env python3
"""
Load SPM data from CSV file for DDMD workflow

This script loads data from the SPM results CSV and filters it according to
the specific producer-consumer pairs for DDMD workflow:
- openmm+aggregate: openmm+aggregate
- openmm+train: openmm+train
- openmm+inference: openmm+inference
"""

import pandas as pd
import numpy as np

def load_spm_data_from_csv(csv_file):
    """Load SPM data from CSV file for DDMD workflow"""
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV with {len(df)} rows")
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Initialize the spm_data dictionary
    spm_data = {
        "store_conf": [
            "SSD 2n", "SSD 4n",
            "BeeGFS 2n", "BeeGFS 4n"
        ],
        "openmm+aggregate": [],
        "openmm+train": [],
        "openmm+inference": []
    }
    
    # Helper function to find SPM value based on specific conditions
    def find_spm_value(df, producer, consumer, prod_storage, cons_storage, 
                       prod_tasks, cons_tasks):
        """Find SPM value for specific producer-consumer configuration"""
        mask = (
            (df['producer'] == producer) &
            (df['consumer'] == consumer) &
            (df['producerStorageType'] == prod_storage) &
            (df['consumerStorageType'] == cons_storage) &
            (df['producerTasksPerNode'] == prod_tasks) &
            (df['consumerTasksPerNode'] == cons_tasks)
        )
        filtered = df[mask]
        if len(filtered) > 0:
            return filtered.iloc[0]['SPM']
        else:
            print(f"Warning: No data found for {producer} -> {consumer} with "
                  f"prod_storage={prod_storage}, cons_storage={cons_storage}, "
                  f"prod_tasks={prod_tasks}, cons_tasks={cons_tasks}")
            return None
    
    print("\n" + "="*80)
    print("EXTRACTING SPM VALUES FOR DDMD WORKFLOW")
    print("="*80)
    
    # 1. openmm+aggregate: openmm+aggregate
    print("\n1. Processing openmm+aggregate...")
    
    # SSD configurations
    ssd_2n = find_spm_value(df, "openmm", "aggregate", 
                           "ssd", "ssd", 6, 1)
    ssd_4n = find_spm_value(df, "openmm", "aggregate", 
                           "ssd", "ssd", 3, 1)
    
    # BeeGFS configurations
    beegfs_2n = find_spm_value(df, "openmm", "aggregate", 
                              "beegfs", "beegfs", 6, 1)
    beegfs_4n = find_spm_value(df, "openmm", "aggregate", 
                              "beegfs", "beegfs", 3, 1)
    
    # Combine all values for openmm+aggregate
    spm_data["openmm+aggregate"] = [ssd_2n, ssd_4n, beegfs_2n, beegfs_4n]
    
    print("openmm+aggregate values:", spm_data["openmm+aggregate"])
    
    # 2. openmm+train: openmm+train
    print("\n2. Processing openmm+train...")
    
    # SSD configurations
    ssd_2n = find_spm_value(df, "openmm", "training", 
                           "ssd", "ssd", 6, 1)
    ssd_4n = find_spm_value(df, "openmm", "training", 
                           "ssd", "ssd", 3, 1)
    
    # BeeGFS configurations
    beegfs_2n = find_spm_value(df, "openmm", "training", 
                              "beegfs", "beegfs", 6, 1)
    beegfs_4n = find_spm_value(df, "openmm", "training", 
                              "beegfs", "beegfs", 3, 1)
    
    # Combine all values for openmm+train
    spm_data["openmm+train"] = [ssd_2n, ssd_4n, beegfs_2n, beegfs_4n]
    
    print("openmm+train values:", spm_data["openmm+train"])
    
    # 3. openmm+inference: openmm+inference
    print("\n3. Processing openmm+inference...")
    
    # SSD configurations
    ssd_2n = find_spm_value(df, "openmm", "inference", 
                           "ssd", "ssd", 6, 1)
    ssd_4n = find_spm_value(df, "openmm", "inference", 
                           "ssd", "ssd", 3, 1)
    
    # BeeGFS configurations
    beegfs_2n = find_spm_value(df, "openmm", "inference", 
                              "beegfs", "beegfs", 6, 1)
    beegfs_4n = find_spm_value(df, "openmm", "inference", 
                              "beegfs", "beegfs", 3, 1)
    
    # Combine all values for openmm+inference
    spm_data["openmm+inference"] = [ssd_2n, ssd_4n, beegfs_2n, beegfs_4n]
    
    print("openmm+inference values:", spm_data["openmm+inference"])
    
    return spm_data

def main():
    """Main function to load and display SPM data"""
    
    # File path
    csv_file = "../workflow_spm_results/ddmd_4n_l_filtered_spm_results.csv"
    
    try:
        # Load the SPM data
        spm_data = load_spm_data_from_csv(csv_file)
        
        # Print the final structure
        print("\n" + "="*80)
        print("FINAL SPM DATA STRUCTURE")
        print("="*80)
        
        for key, values in spm_data.items():
            if key == "store_conf":
                print(f"\n{key}: {values}")
            else: 
                print(f"\n{key}:")
                for i, val in enumerate(values):
                    if val is not None:
                        print(f"  {spm_data['store_conf'][i]}: {val}")
                    else:
                        print(f"  {spm_data['store_conf'][i]}: None")
        
        print("\n" + "="*80)
        print("COPYABLE FORMAT:")
        print("="*80)
        print("spm_data = {")
        print(f'    "store_conf": {spm_data["store_conf"]},')
        
        for key, values in spm_data.items():
            if key != "store_conf":
                # Convert None values to 0.0 for the copyable format
                formatted_values = []
                for val in values:
                    if val is not None:
                        formatted_values.append(float(val))
                    else:
                        formatted_values.append(0.0)
                print(f'    "{key}": {formatted_values},')
        
        print("}")
        
        return spm_data
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    main()
