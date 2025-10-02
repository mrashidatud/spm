"""
Random Forest-based transfer rate estimation for workflow analysis.
Mirrors the public API of workflow_interpolation but replaces the core
interpolation with a learned model.
"""

import warnings
from typing import Tuple
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

from .workflow_config import MULTI_NODES, STORAGE_LIST


def _prepare_training_data(
    df_ior: pd.DataFrame,
    operation: str,
    par_col: str,
    transfer_rate_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    df = df_ior[df_ior["operation"] == operation].copy()
    cols_needed = [
        "aggregateFilesizeMB",
        "numNodes",
        par_col,
        "transferSize",
        transfer_rate_col,
    ]
    df = df.dropna(subset=cols_needed)
    if df.empty:
        return np.empty((0, 4)), np.empty((0,))

    X = df[["aggregateFilesizeMB", "numNodes", par_col, "transferSize"]].to_numpy(dtype=float)
    y = df[transfer_rate_col].to_numpy(dtype=float)
    return X, y


# In-memory model cache: keys are (storage, operation, par_col, transfer_rate_col)
_MODEL_CACHE: dict[tuple, RandomForestRegressor] = {}


def _default_model_dir() -> str:
    here = os.path.dirname(__file__)
    # Save under perf_profiles/models relative to the repo root
    return os.path.normpath(os.path.join(here, "../../perf_profiles/models"))


def _model_path(storage_filter: str, operation: str, par_col: str, transfer_rate_col: str, model_dir: str | None = None) -> str:
    if model_dir is None:
        model_dir = _default_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    safe_storage = str(storage_filter).replace("/", "_")
    filename = f"rf_{safe_storage}_{operation}_{par_col}_{transfer_rate_col}.joblib"
    return os.path.join(model_dir, filename)


def _load_or_train_model(
    df_ior_storage: pd.DataFrame,
    operation: str,
    par_col: str,
    transfer_rate_col: str,
    storage_filter: str,
    model_dir: str | None = None,
) -> RandomForestRegressor:
    cache_key = (str(storage_filter), str(operation), str(par_col), str(transfer_rate_col))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    path = _model_path(storage_filter, operation, par_col, transfer_rate_col, model_dir)
    if os.path.exists(path):
        rf = load(path)
        _MODEL_CACHE[cache_key] = rf
        return rf

    X, y = _prepare_training_data(df_ior_storage, operation, par_col, transfer_rate_col)
    if X.shape[0] == 0:
        raise ValueError(f"No rows found for the specified operation: {operation}.")

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    dump(rf, path)
    _MODEL_CACHE[cache_key] = rf
    return rf


def preload_models(
    df_ior_sorted: pd.DataFrame,
    storage_list: list[str],
    operations: list[str],
    par_col: str,
    transfer_rate_col: str,
) -> None:
    # Determine available storages in IOR data
    available_storages = set(df_ior_sorted.get("storageType", pd.Series([], dtype=str)).unique())
    for storage in storage_list:
        storage_filter = "beegfs" if storage == "pfs" else storage
        if storage_filter not in available_storages:
            continue
        df_storage = df_ior_sorted[df_ior_sorted["storageType"] == storage_filter]
        for op in operations:
            try:
                _load_or_train_model(
                    df_ior_storage=df_storage,
                    operation=op,
                    par_col=par_col,
                    transfer_rate_col=transfer_rate_col,
                    storage_filter=storage_filter,
                )
            except Exception:
                # Skip if not trainable for this combo
                continue


def calculate_randomforestregressor(
    df_ior_sorted: pd.DataFrame,
    target_operation,
    target_aggregateFilesizeMB,
    target_numNodes,
    target_parallelism,
    target_transfer_size,
    par_col,
    transferRate_column,
    multi_nodes: bool = True,
    debug: bool = False,
):
    """
    Estimate transfer rate using a persisted RandomForestRegressor model.

    If a model for (storage, operation, par_col, transferRate_column) exists, it is loaded.
    Otherwise, it is trained once from the provided df_ior_sorted (filtered to that storage)
    and saved under perf_profiles/models for future runs.

    Returns:
    - tuple: (estimated_transfer_rate, transfer_size_slope)
    """

    # Determine storage identifier from provided df (assumes single storage filtered upstream)
    if "storageType" in df_ior_sorted.columns:
        storages = sorted(df_ior_sorted["storageType"].dropna().unique())
        storage_filter = str(storages[0]) if len(storages) > 0 else "unknown"
    else:
        storage_filter = "unknown"

    # Load or train model
    rf = _load_or_train_model(
        df_ior_storage=df_ior_sorted,
        operation=target_operation,
        par_col=par_col,
        transfer_rate_col=transferRate_column,
        storage_filter=storage_filter,
    )

    # Predict for target feature vector
    x_target = np.array([
        float(target_aggregateFilesizeMB),
        float(target_numNodes),
        float(target_parallelism),
        float(target_transfer_size),
    ], dtype=float).reshape(1, -1)

    y_pred = float(rf.predict(x_target)[0])

    # Estimate slope wrt transferSize via finite difference
    # Choose epsilon based on data spread to be meaningful but small
    ts_values = df_ior_sorted["transferSize"].dropna().to_numpy(dtype=float)
    if ts_values.size > 0:
        ts_std = float(np.std(ts_values))
        epsilon = max(1.0, ts_std * 0.01)
    else:
        epsilon = 1.0

    x_eps = x_target.copy()
    x_eps[0, 3] = float(target_transfer_size) + epsilon
    y_pred_eps = float(rf.predict(x_eps)[0])
    ts_slope = (y_pred_eps - y_pred) / epsilon if epsilon != 0 else 0.0

    # Ensure non-negative prediction as transfer rates are non-negative
    if y_pred < 0:
        y_pred = max(0.0, y_pred)

    return y_pred, ts_slope


def estimate_transfer_rates_for_workflow(
    wf_pfs_df,
    df_ior_sorted,
    storage_list,
    allowed_parallelism=None,
    multi_nodes: bool = True,
    debug: bool = False,
):
    """
    Same behavior as in workflow_interpolation, but uses the RF-based estimator.
    """
    warnings.filterwarnings("ignore")

    if allowed_parallelism is None:
        max_parallelism = wf_pfs_df["parallelism"].max() if "parallelism" in wf_pfs_df.columns else 1
        allowed_parallelism = [1, max_parallelism]
        if debug:
            print(f"Using default allowed_parallelism: {allowed_parallelism}")

    cp_scp_parallelism = set(wf_pfs_df.loc[wf_pfs_df["operation"].isin(["cp", "scp", "none"]), "parallelism"].unique())
    allowed_parallelism = sorted(set(allowed_parallelism).union(cp_scp_parallelism))

    if debug:
        stage_tasks = wf_pfs_df[wf_pfs_df["taskName"].str.contains("stage_in|stage_out", na=False)]
        print(f"Found {len(stage_tasks)} stage_in/stage_out tasks:")
        for _, task in stage_tasks.iterrows():
            print(f"  Task: {task['taskName']}, Operation: {task['operation']}, Storage: {task['storageType']}")

    # Preload models once per run for seen operations and storages
    ops_in_wf = sorted(set(wf_pfs_df["operation"].unique()) - {"none"})
    par_col = "tasksPerNode" if multi_nodes else "parallelism"
    try:
        preload_models(
            df_ior_sorted=df_ior_sorted,
            storage_list=storage_list,
            operations=ops_in_wf,
            par_col=par_col,
            transfer_rate_col="trMiB",
        )
    except Exception:
        pass

    for index, row in wf_pfs_df.iterrows():
        operation = row["operation"]
        transfer_size = row["transferSize"]
        aggregateFilesizeMB = row["aggregateFilesizeMB"]
        numNodes = row["numNodes"]
        task_name = row.get("taskName", "Unknown")

        if operation == "none":
            for storage in storage_list:
                for parallelism in allowed_parallelism:
                    col_name_tr_storage = f"estimated_trMiB_{storage}_{parallelism}p"
                    col_name_ts_slope = f"estimated_ts_slope_{storage}_{parallelism}p"
                    if col_name_tr_storage not in wf_pfs_df.columns:
                        wf_pfs_df[col_name_tr_storage] = None
                    if col_name_ts_slope not in wf_pfs_df.columns:
                        wf_pfs_df[col_name_ts_slope] = None
                    wf_pfs_df.at[index, col_name_tr_storage] = 0.0
                    wf_pfs_df.at[index, col_name_ts_slope] = 0.0
            continue

        if multi_nodes:
            task_parallelism = row["tasksPerNode"]
            parallelism_range = [task_parallelism]
            par_col = "tasksPerNode"
        else:
            task_parallelism = row["parallelism"]
            parallelism_range = [p for p in allowed_parallelism if p <= task_parallelism]
            par_col = "parallelism"

        if operation in ["cp", "scp"]:
            storage_types = [row["storageType"]]
        else:
            storage_types = storage_list

        for storage in storage_types:
            for parallelism in parallelism_range:
                col_name_tr_storage = f"estimated_trMiB_{storage}_{parallelism}p"
                col_name_ts_slope = f"estimated_ts_slope_{storage}_{parallelism}p"

                try:
                    storage_filter = f"{storage}"
                    if storage == "pfs":
                        storage_filter = "beegfs"

                    df_ior_storage = df_ior_sorted[df_ior_sorted["storageType"] == storage_filter]
                    if df_ior_storage.empty:
                        if debug:
                            print(f"No data found for storage type: {storage_filter}")
                        continue

                    estimated_trMiB_storage, ts_slope = calculate_randomforestregressor(
                        df_ior_storage,
                        operation,
                        aggregateFilesizeMB,
                        numNodes,
                        parallelism,
                        transfer_size,
                        par_col,
                        "trMiB",
                        multi_nodes,
                        debug and (task_name == "individuals"),
                    )
                except Exception as e:
                    if debug:
                        print(
                            f"RF estimation error for storage {storage}, parallelism {parallelism}: {e}"
                        )
                    estimated_trMiB_storage = None
                    ts_slope = None

                if col_name_tr_storage not in wf_pfs_df.columns:
                    wf_pfs_df[col_name_tr_storage] = None
                if col_name_ts_slope not in wf_pfs_df.columns:
                    wf_pfs_df[col_name_ts_slope] = None

                wf_pfs_df.at[index, col_name_tr_storage] = estimated_trMiB_storage
                if ts_slope is not None:
                    wf_pfs_df.at[index, col_name_ts_slope] = float(ts_slope)

                if debug and estimated_trMiB_storage is not None and estimated_trMiB_storage < 0:
                    print(
                        f"[NEGATIVE TRANSFER RATE] Task: {task_name}, Operation: {operation}, Storage: {storage}, Parallelism: {parallelism}, "
                        f"aggregateFilesizeMB: {aggregateFilesizeMB}, numNodes: {numNodes}, transfer_size: {transfer_size}, "
                        f"col_name: {col_name_tr_storage}, estimated_trMiB_storage: {estimated_trMiB_storage}, ts_slope: {ts_slope}"
                    )

                if debug:
                    print(
                        f"Task[{task_name}] Storage[{storage}] Parallelism[{parallelism}] aggregateFilesizeMB[{aggregateFilesizeMB}] -> {col_name_tr_storage} = {estimated_trMiB_storage}"
                    )

    return wf_pfs_df


def calculate_aggregate_filesize_per_node(wf_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    # Reuse the same implementation by importing from the interpolation module to avoid divergence
    from .workflow_interpolation import calculate_aggregate_filesize_per_node as _impl
    return _impl(wf_df, debug)


