# constants.py -------------------------------------------------
# Storage universe & global sweep settings
STORAGES      = ["beegfs", "tmpfs", "ssd"]
GLOBAL_NODES  = [1, 2, 4]  # same N for every stage

# Project I/O
DATA_DIR      = "ddmd"   # <== adjust to your repo layout
OUT_DIR       = "ddmd"                 # where plots/csvs will be written

# Inputs
# JSON with stages/parallelism/precedence (unchanged)
META_FILE     = f"{DATA_DIR}/ddmd_script_order.json"  # adjust name/path if needed

# NEW: CSV is now the single source of truth for SPM rows
SPM_CSV_FILE  = f"{DATA_DIR}/ddmd_4n_l_filtered_spm_results.csv"

# Policy: avoid stage-out transitions we don't have costs for
FORBIDDEN_CROSS = {("ssd", "tmpfs"), ("tmpfs", "ssd")}
