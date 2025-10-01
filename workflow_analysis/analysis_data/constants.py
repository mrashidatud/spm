# constants.py -------------------------------------------------
# Storage universe & global sweep settings
STORAGES      = ["beegfs", "tmpfs", "ssd"]
GLOBAL_NODES  = [2, 4, 8]  # same N for every stage

# Project I/O
DATA_DIR      = "pyflxtrkr/pyflex_s9_48f"   # <== adjust to your repo layout
OUT_DIR       = "pyflxtrkr/pyflex_s9_48f"                 # where plots/csvs will be written

# Inputs
# JSON with stages/parallelism/precedence (unchanged)
META_FILE     = f"{DATA_DIR}/pyflextrkr_s9_script_order.json"  # adjust name/path if needed

# NEW: CSV is now the single source of truth for SPM rows
SPM_CSV_FILE  = f"{DATA_DIR}/pyflex_s9_48f_filtered_spm_results.csv"

# Policy: avoid stage-out transitions we don't have costs for
FORBIDDEN_CROSS = {("ssd", "tmpfs"), ("tmpfs", "ssd")}
