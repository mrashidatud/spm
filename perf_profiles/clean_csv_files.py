#!/usr/bin/env python3
"""
Script to clean IOR CSV files by removing rows where trMiB is 0
"""

import pandas as pd
import os

def clean_csv_file(filename):
    """
    Clean a CSV file by removing rows where trMiB is 0
    
    Args:
        filename: Path to the CSV file
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    
    # Read the CSV file
    df = pd.read_csv(filename)
    original_count = len(df)
    
    # Remove rows where trMiB is 0
    df_clean = df[df['trMiB'] > 0]
    removed_count = original_count - len(df_clean)
    
    # Save the cleaned data back to the same file
    df_clean.to_csv(filename, index=False)
    
    print(f"Cleaned {filename}:")
    print(f"  Original records: {original_count}")
    print(f"  Cleaned records: {len(df_clean)}")
    print(f"  Removed records: {removed_count}")
    print()

def main():
    """Main function to clean all IOR CSV files"""
    print("=== Cleaning IOR CSV Files ===\n")
    
    # List of CSV files to clean
    csv_files = [
        'ior_data_beegfs.csv',
        'ior_data_ssd.csv', 
        'ior_data_nfs.csv',
        'ior_data_tmpfs.csv'
    ]
    
    # Clean each file
    for filename in csv_files:
        clean_csv_file(filename)
    
    print("=== Cleaning Complete ===")
    print("All individual storage CSV files have been cleaned.")
    print("Rows with trMiB = 0 have been removed.")

if __name__ == "__main__":
    main() 