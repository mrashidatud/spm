# model.py
# =============================================================================
# WORKFLOW MAKESPAN — RULES & LIMITATIONS (Concise)
# -----------------------------------------------------------------------------
# Barriered timeline per stage_order k:
#   in_k   = max over all stage_in actions needed BEFORE running stages at order k
#   exec_k = max over execution time of stages at order k
#            (for each stage C: READ on σ(C) + WRITE on σ(C))
#   out_k  = max over all stage_out actions AFTER finishing stages at order k
#   TOTAL  = Σ_k (in_k + exec_k + out_k)
#
# Associations of costs:
#   • stage_in/out: ONLY data movement (cp/scp).
#   • execution:    ONLY read/write of stages.
#
# Execution (per stage C)
#   • READ(C): max over predecessors P of consumer-side read on σ(C).
#       Use SPM rows with (producer=P, consumer=C, storage_src=σ(C), storage_dst=σ(C));
#       prefer cons_tpn == tpn(C).
#     For “initial-only” C:
#       - if order is minimal and σ(C) is local, use local-read from stage_in row;
#       - otherwise use shared-read from stage_in row (and for later orders C must be on beegfs).
#   • WRITE(C):
#       - If C has successors D with ORDER[D] > ORDER[C]: take the MAX producer-side write on σ(C)
#         using direct execution rows:
#             (producer = C, consumer = D, storage_src = σ(C), storage_dst = σ(C))
#         TPN preference: prefer prod_tpn == tpn(C); if absent, fall back to any available row.
#
#       - If C is FINAL (no successors):
#           • If σ(C) is LOCAL (e.g., ssd/tmpfs):
#               use the producer-side WRITE from the final stage_out row:
#                   (producer = C, consumer = stage_out-C,
#                    storage_src = σ(C), storage_dst = σ(C)-beegfs)
#               TPN preference: prefer prod_tpn == tpn(C); if absent, fall back to cons_tpn == tpn(C),
#               otherwise use any available row.
#
#           • If σ(C) is SHARED (beegfs):
#               use the producer-side WRITE from stage_out rows with beegfs as source:
#                   (producer = C, consumer = stage_out-C,
#                    storage_src = beegfs, storage_dst starts with "beegfs-")
#               TPN preference: prefer prod_tpn == tpn(C); if absent, fall back to cons_tpn == tpn(C),
#               otherwise use any available row.
#
#   • NOTE: σ(P) and σ(C) may differ; any cross-storage movement required because σ(P) ≠ σ(C)
#     is charged as cp/scp in the stage_out phases (not in execution).
#
# Stage_in (in_k)
#   • FIRST order ONLY (per available CSV):
#       beegfs→local cp via stage_in rows: charge PRODUCER-side cp only (prod_spm),
#       with prod_tpn == 1. Execution READ then uses local-read for those stages.
#   • LATER orders: no stage_in cp available; initial-only stages must be on beegfs.
#
# Stage_out (out_k)
#   • NON-FINAL orders:
#       - beegfs→local cp (from producer’s stage_out rows): CONSUMER-side cp only (cons_spm),
#         with cons_tpn == 1.
#       - local→local scp fan-out: CONSUMER-side scp only (cons_spm), with cons_tpn == 1.
#       De-duplicate by (producer, target storage) and take per-order max.
#   • FINAL order:
#       - local→beegfs cp contributes ONLY cp (CONSUMER-side, cons_spm with cons_tpn==1) to out_k.
#       - The WRITE (producer-side) time reported in those stage_out rows belongs to execution (WRITE(C)).
#
# TPN filtering
#   • stage_out movement (cp/scp): cons_tpn == 1
#   • stage_in cp (first order):  prod_tpn == 1
#   • execution READ:             prefer cons_tpn == tpn(C)
#   • execution WRITE:            prefer prod_tpn == tpn(C) (final-write via stage_out may fallback to cons_tpn==tpn(C))
#
# Validity (reflects CSV availability)
#   • No local↔local CROSS (e.g., ssd↔tmpfs) anywhere (unmeasured).
#   • No local→beegfs before the final order (unavailable).
#   • Initial-only stage in later orders cannot use local (no stage_in cp available).
#
# CSV Reporting (this file)
#   • For each stage S (ordered by stage_order, then alphabetically), we report four columns:
#       in_S, read_S, write_S, out_S
#     Meaning of each column and when it is populated:
#       - in_S    : stage-in (cp/scp) cost CONSIDERED for S at its order (first order only, beegfs→local cp, producer-side).
#       - read_S  : execution READ cost CONSIDERED for S (max over all feasible predecessors P on σ(S), or initial-read).
#       - write_S : execution WRITE cost CONSIDERED for S (max over feasible successors D on σ(S); if S is final on local,
#                   use the producer-side WRITE from the final stage_out row).
#       - out_S   : stage-out (cp/scp) cost CONSIDERED for S at its order (non-final: beegfs→local cp and/or local→local fan-out,
#                   final: local→beegfs cp; all consumer-side).
#
#   • IMPORTANT: These per-stage fields show the value that was actually CONSIDERED in the max-comparison for that order,
#     even if they did NOT win the barrier and are NOT on the critical path. A hyphen “-” is used ONLY when the component
#     was NOT APPLICABLE / NOT EVALUATED under the current storage assignment (e.g., no stage_in needed, no stage_out path, etc.).
#     If multiple candidates existed for a component of S (e.g., multiple predecessors for read_S), the field shows the MAX
#     among the candidates considered for S at that order.
#
#   • Per-order aggregates still appear:
#       in_oK   : max of all in_S considered at order K
#       exec_oK : max of per-stage (read_S + write_S) at order K
#       out_oK  : max of all out_S considered at order K
#     TOTAL = Σ_K (in_oK + exec_oK + out_oK).
#
#   • The critical_path field records which stage/phase achieved each order’s barrier max (tokens like stage_in_S, read_S, write_S, stage_out_S).
#
#   • The CSV also includes nodes, storage choices (σ(S) per stage), and is row-sorted by total (then nodes).
# =============================================================================

import os
import itertools
import math
import numpy as np
import pandas as pd

from constants import STORAGES, GLOBAL_NODES, OUT_DIR, FORBIDDEN_CROSS
from parser import load_rank, load_meta

DF = load_rank()
TASKS, ORDER, PREDS, CONSUMERS = load_meta()
ORDERS = sorted(set(ORDER.values()))
os.makedirs(OUT_DIR, exist_ok=True)

LOCAL_STORES = tuple(s for s in STORAGES if s != "beegfs")
O_MIN = min(ORDERS)
O_MAX = max(ORDERS)

# -----------------------------
# Helpers & lookups
# -----------------------------
def tpn_map(N: int):
    return {s: max(1, math.ceil(TASKS[s] / N)) for s in TASKS}

def _nz(x) -> float:
    try:
        xv = float(x)
        return 0.0 if np.isnan(xv) else xv
    except Exception:
        return 0.0

def _min_col(d: pd.DataFrame, col: str) -> float:
    if d is None or d.empty or col not in d.columns:
        return np.nan
    vals = d[col]
    return float(np.nanmin(vals.values)) if len(vals) else np.nan

def _stage_sort_key(name: str):
    return (ORDER[name], name.lower())

def only_initial(c: str) -> bool:
    preds = [p for p in PREDS[c] if p is not None]
    return len(preds) == 1 and preds[0] == "initial_data"

def _order_groups():
    grp = {}
    for s, o in ORDER.items():
        grp.setdefault(o, []).append(s)
    for o in grp:
        grp[o].sort(key=str.lower)
    return dict(sorted(grp.items()))

# -----------------------------
# TPN-aware filters
# -----------------------------
def _tpn_filter_stage_out_cons(base: pd.DataFrame) -> pd.DataFrame:
    # movement cp/scp via stage_out uses cons_tpn==1
    if base is None or base.empty or ("cons_tpn" not in base.columns):
        return base
    return base[(base["cons_tpn"] == 1)]

def _tpn_filter_stage_in_prod(base: pd.DataFrame) -> pd.DataFrame:
    # first-order stage_in cp uses prod_tpn==1
    if base is None or base.empty or ("prod_tpn" not in base.columns):
        return base
    return base[(base["prod_tpn"] == 1)]

def _tpn_filter_exec_read(base: pd.DataFrame, c_tpn_val: int) -> pd.DataFrame:
    # prefer cons_tpn == tpn(C) for read
    if base is None or base.empty or ("cons_tpn" not in base.columns):
        return base
    sel = base[base["cons_tpn"] == c_tpn_val]
    return sel if not sel.empty else base

def _tpn_filter_exec_write(base: pd.DataFrame, c_tpn_val: int) -> pd.DataFrame:
    # prefer prod_tpn == tpn(C) for write
    if base is None or base.empty or ("prod_tpn" not in base.columns):
        return base
    sel = base[base["prod_tpn"] == c_tpn_val]
    return sel if not sel.empty else base

def _tpn_filter_final_write_stageout(base: pd.DataFrame, c_tpn_val: int) -> pd.DataFrame:
    # prefer prod_tpn==tpn(C); fallback cons_tpn==tpn(C); else any
    if base is None or base.empty:
        return base
    have_p = "prod_tpn" in base.columns
    have_c = "cons_tpn" in base.columns
    if have_p:
        psel = base[base["prod_tpn"] == c_tpn_val]
        if not psel.empty:
            return psel
    if have_c:
        csel = base[base["cons_tpn"] == c_tpn_val]
        if not csel.empty:
            return csel
    return base

# -----------------------------
# Cost accessors
# -----------------------------
# movement (stage_out cp/scp) — consumer side only
def stage_out_consumer_cost(stage: str, src: str, combined_dst: str) -> float:
    d = DF[
        (DF.producer == stage)
        & (DF.consumer == f"stage_out-{stage}")
        & (DF.storage_src == src)
        & (DF.storage_dst == combined_dst)
    ]
    d = _tpn_filter_stage_out_cons(d)
    return _min_col(d, "cons_spm")

# first-order stage_in cp — producer side only
def stage_in_cp_producer_cost(stage: str, dst_local: str) -> float:
    d = DF[
        (DF.producer == f"stage_in-{stage}")
        & (DF.consumer == stage)
        & (DF.storage_src == f"beegfs-{dst_local}")
        & (DF.storage_dst == dst_local)
    ]
    d = _tpn_filter_stage_in_prod(d)
    return _min_col(d, "prod_spm")

# shared read for initial-only (consumer side)
def stage_in_shared_read_runtime(stage: str, cons_tpn_val: int) -> float:
    d = DF[
        (DF.producer == f"stage_in-{stage}")
        & (DF.consumer == stage)
        & (DF.storage_src == "beegfs")
        & (DF.storage_dst == "beegfs")
    ]
    d = _tpn_filter_exec_read(d, cons_tpn_val)
    return _min_col(d, "cons_spm")

# local read for initial-only in first order after cp (consumer side)
def stage_in_initial_local_read_runtime(stage: str, dst_local: str, cons_tpn_val: int) -> float:
    d = DF[
        (DF.producer == f"stage_in-{stage}")
        & (DF.consumer == stage)
        & (DF.storage_src == f"beegfs-{dst_local}")
        & (DF.storage_dst == dst_local)
    ]
    d = _tpn_filter_exec_read(d, cons_tpn_val)
    return _min_col(d, "cons_spm")

# execution READ for (P->C) measured on σ(C): consumer side only
def exec_read_cost(p: str, c: str, store_y: str, c_tpn_val: int) -> float:
    d = DF[
        (DF.producer == p)
        & (DF.consumer == c)
        & (DF.storage_src == store_y)
        & (DF.storage_dst == store_y)
    ]
    d = _tpn_filter_exec_read(d, c_tpn_val)
    return _min_col(d, "cons_spm")

# execution WRITE for (C->D) measured on σ(C): producer side only
def exec_write_cost(c: str, d: str, store_y: str, c_tpn_val: int) -> float:
    d = DF[
        (DF.producer == c)
        & (DF.consumer == d)
        & (DF.storage_src == store_y)
        & (DF.storage_dst == store_y)
    ]
    d = _tpn_filter_exec_write(d, c_tpn_val)
    return _min_col(d, "prod_spm")

# final WRITE for stage C via stage_out row (local->beegfs): producer side only
def final_exec_write_from_stageout(c: str, store_y: str, c_tpn_val: int) -> float:
    d = DF[
        (DF.producer == c)
        & (DF.consumer == f"stage_out-{c}")
        & (DF.storage_src == store_y)
        & (DF.storage_dst == f"{store_y}-beegfs")
    ]
    d = _tpn_filter_final_write_stageout(d, c_tpn_val)
    return _min_col(d, "prod_spm")

# final WRITE for stage C when it runs on beegfs:
# use stage_out rows with storage_src='beegfs' (producer writes to shared),
# and any storage_dst that starts with 'beegfs-'. Prefer prod_tpn==tpn(C),
# fallback cons_tpn==tpn(C), else any.
def final_exec_write_on_beegfs_from_stageout(c: str, c_tpn_val: int) -> float:
    d = DF[
        (DF.producer == c)
        & (DF.consumer == f"stage_out-{c}")
        & (DF.storage_src == "beegfs")
    ]
    if not d.empty and "storage_dst" in d.columns:
        d = d[d["storage_dst"].astype(str).str.startswith("beegfs-")]
    d = _tpn_filter_final_write_stageout(d, c_tpn_val)
    return _min_col(d, "prod_spm")


# -----------------------------
# Validity filter
# -----------------------------
def config_is_valid(cfg: dict) -> bool:
    # forbid local<->local CROSS (unmeasured), e.g., ssd<->tmpfs
    for c, preds in PREDS.items():
        if not preds:
            continue
        for p in preds:
            if p == "initial_data":
                continue
            X, Y = cfg[p], cfg[c]
            if X != Y and (X, Y) in FORBIDDEN_CROSS:
                return False
            # No local->beegfs before the final order (unavailable in CSV)
            if ORDER[p] < ORDER[c] and ORDER[c] < O_MAX:
                if X in LOCAL_STORES and Y == "beegfs":
                    return False
    # Later orders with only initial_data cannot use local (no stage_in cp available)
    for c in TASKS:
        if only_initial(c) and ORDER[c] > O_MIN and cfg[c] != "beegfs":
            return False
    return True

# -----------------------------
# Stage_in actions (in_k) for order k
# -----------------------------
def _stage_in_actions_for_order(order_k: int, cfg: dict):
    """
    FIRST order only:
      - stage_in cp (beegfs->local) for stages on local.
      - cost: PRODUCER-only (prod_spm with prod_tpn==1)
    Returns list of actions: {stage, cost}
    """
    actions = []
    if order_k != O_MIN:
        return actions

    for c in [s for s in TASKS if ORDER[s] == order_k]:
        Y = cfg[c]
        if Y in LOCAL_STORES:
            # allow if initial-only OR any predecessor on beegfs
            if only_initial(c) or any(cfg.get(p) == "beegfs" for p in PREDS[c] if p != "initial_data"):
                cost = _nz(stage_in_cp_producer_cost(c, Y))
                if cost > 0:
                    actions.append({"stage": c, "cost": cost})
    return actions

# -----------------------------
# Stage_out actions (out_k) for order k
# -----------------------------
def _stage_out_actions_for_order(order_k: int, cfg: dict):
    """
    NON-FINAL orders:
      - For each producer p in this order, and for each downstream consumer storage Y:
          * if Y in local and p on beegfs: beegfs->Y cp (CONSUMER-only)
          * if Y in local: Y->Y fan-out scp (CONSUMER-only)
      - De-dup by (p, Y); attach action to stage p.

    FINAL order:
      - For each producer p in this order with local storage Yp:
          * local->beegfs cp: CONSUMER-only cost (goes to out_k).
          * (WRITE for p is accounted in execution via final_exec_write_from_stageout.)

    Returns list of actions: {stage, cost}
    """
    actions = []
    stages_k = [s for s in TASKS if ORDER[s] == order_k]

    if order_k < O_MAX:
        so_targets = {}   # ('so', p, Y) -> cost
        fan_targets = {}  # ('fan', p, Y) -> cost

        for p in stages_k:
            succs = [c for c in TASKS if ORDER[c] > order_k and p in PREDS[c]]
            if not succs:
                continue
            targets = sorted({cfg[c] for c in succs})
            for Y in targets:
                if Y in LOCAL_STORES:
                    if cfg[p] == "beegfs":
                        # beegfs->Y cp
                        val = _nz(stage_out_consumer_cost(p, "beegfs", f"beegfs-{Y}"))
                        key = ('so', p, Y)
                        so_targets[key] = max(so_targets.get(key, 0.0), val)
                    # local fan-out Y->Y scp
                    valf = _nz(stage_out_consumer_cost(p, Y, f"{Y}-{Y}"))
                    keyf = ('fan', p, Y)
                    fan_targets[keyf] = max(fan_targets.get(keyf, 0.0), valf)

        # materialize one action per (p,Y) using the larger of cp/fan options if both present
        combined = {}
        for (_, p, Y), v in so_targets.items():
            combined[(p, Y)] = max(combined.get((p, Y), 0.0), v)
        for (_, p, Y), v in fan_targets.items():
            combined[(p, Y)] = max(combined.get((p, Y), 0.0), v)

        for (p, Y), v in combined.items():
            if v > 0:
                actions.append({"stage": p, "cost": v})

    else:
        # FINAL: local->beegfs cp (consumer-only) to out_k
        for p in stages_k:
            Yp = cfg[p]
            if Yp in LOCAL_STORES:
                cp_cost = _nz(stage_out_consumer_cost(p, Yp, f"{Yp}-beegfs"))
                if cp_cost > 0:
                    actions.append({"stage": p, "cost": cp_cost})

    return actions

# -----------------------------
# Execution per stage (READ + WRITE)
# -----------------------------
def _exec_time_for_stage(c: str, cfg: dict, N: int):
    """
    Compute execution for stage c:
      read_max  = max over predecessors (consumer-side read on σ(c)) or initial-read path
      write_max = max over successors (producer-side write on σ(c)) or final-write via stage_out
      exec_time = read_max + write_max

    Returns:
      exec_time, read_max, write_max, read_considered, write_considered
        - read_considered: at least one feasible read candidate was evaluated
        - write_considered: at least one feasible write candidate was evaluated
    """
    Y = cfg[c]
    tpn = tpn_map(N)
    ctpn = tpn.get(c, 1)

    # READ side
    read_candidates = []
    for p in PREDS[c]:
        if p == "initial_data":
            continue
        val = _nz(exec_read_cost(p, c, Y, ctpn))
        # consider only if there was a matching row (val > 0 or val == 0 but existed);
        # conservative: treat presence of predecessor as considered
        read_candidates.append(val)

    if only_initial(c):
        if ORDER[c] == O_MIN:
            if Y in LOCAL_STORES:
                init_val = _nz(stage_in_initial_local_read_runtime(c, Y, ctpn))
            else:
                init_val = _nz(stage_in_shared_read_runtime(c, ctpn))
        else:
            init_val = _nz(stage_in_shared_read_runtime(c, ctpn))
        read_candidates.append(init_val)

    read_considered = len(read_candidates) > 0
    read_max = max(read_candidates) if read_candidates else 0.0

    # WRITE side
    write_candidates = []
    succs = CONSUMERS.get(c, [])
    real_succs = [d for d in succs if ORDER[d] > ORDER[c]]
    if real_succs:
        for d in real_succs:
            val = _nz(exec_write_cost(c, d, Y, ctpn))
            write_candidates.append(val)
    else:
        # final stage write via stage_out local->beegfs if on local
        if Y in LOCAL_STORES:
            # local -> beegfs: producer-side write from stage_out(local->beegfs)
            val = _nz(final_exec_write_from_stageout(c, Y, ctpn))
            write_candidates.append(val)
        elif Y == "beegfs":
            # beegfs producer write: take producer-side write from stage_out(beegfs->*)
            val = _nz(final_exec_write_on_beegfs_from_stageout(c, ctpn))
            write_candidates.append(val)
    write_considered = len(write_candidates) > 0
    write_max = max(write_candidates) if write_candidates else 0.0

    return read_max + write_max, read_max, write_max, read_considered, write_considered

# -----------------------------
# One configuration evaluation
# -----------------------------
def one_config(N: int, cfg: dict):
    order_map = _order_groups()

    # per-stage output cells
    stage_cells = {s: {"in": "-", "read": "-", "write": "-", "out": "-"} for s in TASKS}

    in_times = {}
    exec_times = {}
    out_times = {}
    critical_tokens = []

    # ---- per-order barriers ----
    for ok, stages_k in order_map.items():
        # IN_k
        in_actions = _stage_in_actions_for_order(ok, cfg)
        in_k = max([a["cost"] for a in in_actions], default=0.0)
        in_times[ok] = in_k
        # NEW: stamp every considered stage_in cost for this order
        per_stage_in = {}
        for a in in_actions:
            per_stage_in[a["stage"]] = max(per_stage_in.get(a["stage"], 0.0), a["cost"])
        for s, v in per_stage_in.items():
            stage_cells[s]["in"] = f"{v:.6f}"
        if in_k > 0:
            # pick any action hitting the max (stable tie-break via sort)
            winners = [a for a in in_actions if abs(a["cost"] - in_k) < 1e-12]
            winners.sort(key=lambda x: _stage_sort_key(x["stage"]))
            win = winners[0]
            stage_cells[win["stage"]]["in"] = f"{win['cost']:.6f}"
            critical_tokens.append(f"stage_in_{win['stage']}")

        # --------- EXEC_k ----------
        per_stage_exec = []
        for c in stages_k:
            et, rmax, wmax, r_cons, w_cons = _exec_time_for_stage(c, cfg, N)
            per_stage_exec.append((c, et, rmax, wmax, r_cons, w_cons))

        # NEW: stamp every considered read/write for this order
        for (c, et, rmax, wmax, r_cons, w_cons) in per_stage_exec:
            if r_cons:
                stage_cells[c]["read"] = f"{rmax:.6f}"
            # if not considered, leave as '-'
            if w_cons:
                stage_cells[c]["write"] = f"{wmax:.6f}"
            # if not considered, leave as '-'

        exec_k = max([et for (_, et, _, _, _, _) in per_stage_exec], default=0.0)
        exec_times[ok] = exec_k

        # Mark the stage(s) that actually set exec_k (keeps your critical path tokens)
        if exec_k > 0:
            winners = [t for t in per_stage_exec if abs(t[1] - exec_k) < 1e-12]
            winners.sort(key=lambda x: _stage_sort_key(x[0]))
            c, _, rmax, wmax, r_cons, w_cons = winners[0]
            if r_cons and rmax > 0:
                critical_tokens.append(f"read_{c}")
            if w_cons and wmax > 0:
                critical_tokens.append(f"write_{c}")

        # OUT_k
        out_actions = _stage_out_actions_for_order(ok, cfg)
        out_k = max([a["cost"] for a in out_actions], default=0.0)
        out_times[ok] = out_k
        # NEW: stamp every considered stage_out cost for this order
        per_stage_out = {}
        for a in out_actions:
            per_stage_out[a["stage"]] = max(per_stage_out.get(a["stage"], 0.0), a["cost"])
        for s, v in per_stage_out.items():
            stage_cells[s]["out"] = f"{v:.6f}"

        if out_k > 0:
            winners = [a for a in out_actions if abs(a["cost"] - out_k) < 1e-12]
            winners.sort(key=lambda x: _stage_sort_key(x["stage"]))
            win = winners[0]
            stage_cells[win["stage"]]["out"] = f"{win['cost']:.6f}"
            critical_tokens.append(f"stage_out_{win['stage']}")

    total = float(sum(in_times.values()) + sum(exec_times.values()) + sum(out_times.values()))
    critical_path = "->".join(critical_tokens) if critical_tokens else ""
    return total, in_times, exec_times, out_times, stage_cells, critical_path

# -----------------------------
# Sweep and CSV output
# -----------------------------
def sweep(thresh=None) -> pd.DataFrame:
    rows = []
    ordered_stages = sorted(TASKS.keys(), key=_stage_sort_key)

    for N in GLOBAL_NODES:
        for vals in itertools.product(STORAGES, repeat=len(TASKS)):
            cfg = dict(zip(TASKS, vals))
            if not config_is_valid(cfg):
                continue

            total, in_t, exe_t, out_t, stage_cells, crit = one_config(N, cfg)
            if thresh and total >= thresh:
                continue

            row = {f"{k}_stor": v for k, v in cfg.items()}
            row.update(nodes=N, total=total)
            for o in ORDERS:
                row[f"in_o{o}"] = in_t.get(o, 0.0)
                row[f"exec_o{o}"] = exe_t.get(o, 0.0)
                row[f"out_o{o}"] = out_t.get(o, 0.0)
            row["critical_path"] = crit

            # per-stage reporting: in_S, read_S, write_S, out_S
            for s in ordered_stages:
                row[f"in_{s}"]   = stage_cells[s]["in"]
                row[f"read_{s}"] = stage_cells[s]["read"]
                row[f"write_{s}"]= stage_cells[s]["write"]
                row[f"out_{s}"]  = stage_cells[s]["out"]

            rows.append(row)

    df = pd.DataFrame(rows)

    base_cols = ["nodes", "total"] \
                + [f"in_o{o}" for o in ORDERS] \
                + [f"exec_o{o}" for o in ORDERS] \
                + [f"out_o{o}" for o in ORDERS] \
                + ["critical_path"]

    stor_cols = [f"{s}_stor" for s in ordered_stages]

    per_stage_cols = []
    for s in ordered_stages:
        per_stage_cols += [f"in_{s}", f"read_{s}", f"write_{s}", f"out_{s}"]

    ordered_all = [c for c in base_cols + stor_cols + per_stage_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_all]
    df = df[ordered_all + remaining]

    # sort rows by total then nodes
    return df.sort_values(["total", "nodes"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

def write_csvs() -> pd.DataFrame:
    full = sweep()
    out_path = f"{OUT_DIR}/workflow_makespan_stageorder.csv"
    full.to_csv(out_path, index=False)
    return full
