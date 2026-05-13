import numpy as np
import pandas as pd
from scipy import stats

# ================================================================
# ADVANCED FEATURE ENGINEERING
# ================================================================


def infer_protocol_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer a unified 'Protocol' column for each record.
    Logic:
      - If l4_tcp == 1 → 'TCP'
      - If l4_udp == 1 → 'UDP'
      - If icmp_type > 0 → 'ICMP'
      - Else → 'OTHER'
    """
    df = df.copy()

    # Ensure presence of indicator columns
    if "l4_tcp" not in df.columns:
        df["l4_tcp"] = 0
    if "l4_udp" not in df.columns:
        df["l4_udp"] = 0

    # Boolean flag for ICMP existence
    icmp_present = "icmp_type" in df.columns

    df["Protocol"] = np.select(
        [
            df["l4_tcp"] == 1,
            df["l4_udp"] == 1,
            icmp_present and (df.get("icmp_type", 0) > 0)
        ],
        ["TCP", "UDP", "ICMP"],
        default="OTHER"
    )
    return df


def compute_jitter_iat_features(raw_df,agg_df,time_col="Timestamp_parsed",bin_col="time_bin_10ms"):
    """
    Compute intra-bin IAT (inter-arrival time) and jitter features and merge them
    into an aggregated dataframe. Works for any time resolution (1ms, 5ms, 10ms, etc.)
    as long as 'bin_col' corresponds to the bin column used in agg_df.
    """

    # --- Candidate columns for inter-arrival times ---
    iat_candidates = [
        "inter_arrival_time",
        "time_since_previously_displayed_frame",
        "IAT",
        "iat"
    ]
    iat_col = next((c for c in iat_candidates if c in raw_df.columns), None)
    df_work = raw_df.copy()

    # --- Ensure timestamp exists ---
    if time_col not in df_work.columns:
        if "Timestamp" in df_work.columns:
            df_work[time_col] = pd.to_datetime(df_work["Timestamp"], errors="coerce")
        else:
            raise ValueError("No timestamp column found for IAT computation.")

    # --- Compute inter-arrival times if missing ---
    if iat_col is None:
        if "Flow ID" in df_work.columns:
            df_work = df_work.sort_values(["Flow ID", time_col])
            df_work["iat_tmp"] = (
                df_work.groupby("Flow ID")[time_col]
                .diff()
                .dt.total_seconds()
                .fillna(0)
            )
        else:
            df_work = df_work.sort_values(time_col)
            df_work["iat_tmp"] = df_work[time_col].diff().dt.total_seconds().fillna(0)
        iat_col = "iat_tmp"
    else:
        df_work[iat_col] = (
            pd.to_numeric(df_work[iat_col], errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )

    # --- Floor timestamps into the specified bin size ---
    df_work[bin_col] = pd.to_datetime(df_work[time_col]).dt.floor(
        bin_col.replace("time_bin_", "")
    )

    # --- Group per bin and compute IAT statistics ---
    grp = df_work.groupby(bin_col)[iat_col]
    jitter_agg = grp.agg([
        ("iat_count", "count"),
        ("iat_mean", "mean"),
        ("iat_std", "std"),
        ("iat_median", "median"),
        ("iat_max", "max"),
        ("iat_min", "min"),
    ]).reset_index()

    # Derived micro features
    jitter_agg["iat_cv"] = np.where(
        jitter_agg["iat_mean"] > 0,
        jitter_agg["iat_std"] / jitter_agg["iat_mean"],
        0
    )

    # Mean absolute difference within bins (micro-burstiness)
    def mean_abs_diff(s):
        s = np.asarray(s)
        return float(np.mean(np.abs(np.diff(s)))) if len(s) > 1 else 0.0

    madd = (
        df_work.groupby(bin_col)[iat_col]
        .apply(lambda s: mean_abs_diff(s.values))
        .reset_index(name="iat_madd")
    )
    jitter_agg = pd.merge(jitter_agg, madd, on=bin_col, how="left").fillna(0)

    # --- Log transforms for stability ---
    for c in ["iat_mean", "iat_std", "iat_cv", "iat_madd"]:
        jitter_agg[f"log_{c}"] = np.log1p(np.clip(jitter_agg[c], 0, None))

    jitter_agg = jitter_agg.fillna(0)

    # --- Merge into agg_df using the same bin column dynamically ---
    if bin_col not in agg_df.columns:
        raise KeyError(
            f"{bin_col} not found in aggregated dataframe columns: {agg_df.columns.tolist()}"
        )

    merged = pd.merge(
        agg_df,
        jitter_agg,
        on=bin_col,  # dynamic column (e.g., time_bin_1ms, time_bin_5ms, time_bin_10ms)
        how="left"
    ).fillna(0)

    # --- Verify key column remains intact ---
    if bin_col not in merged.columns:
        merged[bin_col] = agg_df[bin_col]

    return merged


# ------------------------------
# STEP 5: Source / Flow Concentration Weighting
# ------------------------------

def compute_source_flow_concentration(raw_df, agg_df, bin_col="time_bin_5ms",
                                      src_col="Src IP", flow_col="Flow ID"):
    """
    Compute per-bin source/flow concentration metrics and return agg_df
    augmented with:
      - unique_src_count, unique_flow_count
      - src_entropy (Shannon)
      - src_gini (concentration)
      - concentration_score in [0,1] (1 = highly concentrated)
    """
    df_raw = raw_df.copy()
    # ensure bin_col exists in raw (if not, floor timestamps similarly)
    if bin_col not in df_raw.columns:
        df_raw[bin_col] = pd.to_datetime(df_raw["Timestamp_parsed"]).dt.floor(bin_col.replace("time_bin_", ""))

    # Use safe column names if dataset uses different labels
    src_col = src_col if src_col in df_raw.columns else next((c for c in df_raw.columns if "src" in c.lower()), None)
    flow_col = flow_col if flow_col in df_raw.columns else next((c for c in df_raw.columns if "flow id" in c.lower() or "flow_id" in c.lower()), None)

    if src_col is None:
        raise ValueError("No source column found (expected Src IP or similar).")

    # Aggregate per bin: counts per source
    bin_src_counts = (
        df_raw.groupby([bin_col, src_col])
        .size()
        .rename("count")
        .reset_index()
    )

    # Per-bin aggregation metrics
    def shannon_entropy(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum()) if len(p) > 0 else 0.0

    per_bin = []
    for b, group in bin_src_counts.groupby(bin_col):
        counts = group["count"].values
        unique_srcs = len(counts)
        total = counts.sum()
        ent = shannon_entropy(counts)
        # Simpson index as alternate concentration measure
        simpson = np.sum((counts / total) ** 2) if total > 0 else 0.0
        # Gini-like: normalized (0..1) using Simpson
        gini_like = simpson  # higher => more concentrated (1 => single source)
        per_bin.append({
            bin_col: b,
            "unique_src_count": unique_srcs,
            "total_events": int(total),
            "src_entropy": ent,
            "src_simpson": simpson,
            "src_gini_like": gini_like
        })

    concentration_df = pd.DataFrame(per_bin).fillna(0)

    # Merge with agg_df ensuring all bins are present
    merged = pd.merge(agg_df, concentration_df, on=bin_col, how="left").fillna(0)

    # Convert entropy -> concentration score in [0,1]
    # We normalize entropy by log2(unique_src_count+1). For single-source bins, denominator ~0 -> handled.
    def entropy_to_concentration(row):
        u = max(1, int(row.get("unique_src_count", 1)))
        max_ent = np.log2(u) if u > 1 else 1.0
        ent = row.get("src_entropy", 0.0)
        score = 1.0 - (ent / max_ent) if max_ent > 0 else 1.0
        return float(np.clip(score, 0.0, 1.0))

    merged["concentration_score"] = merged.apply(entropy_to_concentration, axis=1)
    # Ensure numeric
    merged["unique_src_count"] = merged["unique_src_count"].astype(int)
    merged["unique_flow_count"] = merged.get("unique_flow_count", merged["unique_src_count"]).fillna(0).astype(int)

    return merged

def generate_concentration_weighted_ci(
    model,
    X_scaled: np.ndarray,
    agg_df_with_conc: pd.DataFrame,
    bin_col: str = "time_bin_10ms",
    base_factor: float = 0.25,
    alpha: float = 0.8,
    min_factor: float = 0.15,
    return_details: bool = False
):
    """
    Compute preds + concentration-weighted CIs.
    - model: ImprovedPoissonConcentrationML (already fitted & local_protocol_factor set)
    - X_scaled: features scaled with benign scaler
    - agg_df_with_conc: aggregated df containing bin_col and 'concentration_score' (0..1)
    - base_factor: factor chosen by micro-calibration (e.g., 0.25)
    - alpha: exponent controlling nonlinearity (higher -> stronger emphasis on high conc.)
    - min_factor: lower bound on factor (avoid zero widths)
    Returns: preds, ci_new (Nx2), per_bin_factors (N,)
    """
    # --- Base preds + CI (model already uses model.local_protocol_factor if set) ---
    preds, ci = model.predict_with_confidence(X_scaled, return_bound_details=False)
    lower, upper = ci[:,0].astype(float), ci[:,1].astype(float)
    widths = np.clip(upper - lower, 1e-12, None)

    # Align agg_df rows to X_scaled order by index — require same ordering as X_scaled
    if bin_col in agg_df_with_conc.columns:
        conc = agg_df_with_conc[bin_col].map(
            dict(zip(agg_df_with_conc[bin_col].values, agg_df_with_conc.get("concentration_score", 0.0).values))
        )
        # If mapping fails because of duplicates / alignment issues, fallback to column fetch:
        try:
            conc_vals = agg_df_with_conc["concentration_score"].values
        except Exception:
            conc_vals = np.zeros(len(widths))
    else:
        # If bin_col not present, try direct concentration_score vector
        conc_vals = agg_df_with_conc.get("concentration_score", np.zeros(len(widths)))
        if len(conc_vals) != len(widths):
            conc_vals = np.resize(conc_vals, len(widths))

    # Safe clip 0..1
    conc_vals = np.clip(np.nan_to_num(conc_vals, nan=0.0), 0.0, 1.0)

    # Compute per-bin multiplicative factor in (min_factor, base_factor]
    # More concentrated -> factor closer to min_factor (narrower)
    # formula: factor = min_factor + (base_factor - min_factor) * (1 - conc_vals**alpha)
    per_bin_factors = min_factor + (base_factor - min_factor) * (1.0 - (conc_vals ** float(alpha)))

    # Ensure factor is in sensible bounds
    per_bin_factors = np.clip(per_bin_factors, min(min_factor, base_factor), max(min_factor, base_factor))

    # Apply factor to widths (smaller factor -> narrower width)
    new_widths = widths * per_bin_factors

    # Recreate symmetric CI around preds (keep preds unchanged)
    half = new_widths / 2.0
    new_lower = np.clip(preds - half, 0.0, None)
    new_upper = preds + half

    ci_new = np.vstack([new_lower, new_upper]).T

    if return_details:
        return preds, ci_new, per_bin_factors, {
            "orig_mean_width": float(np.mean(widths)),
            "new_mean_width": float(np.mean(new_widths)),
            "alpha": alpha, "min_factor": min_factor, "base_factor": base_factor
        }
    return preds, ci_new, per_bin_factors


def infer_protocol_ciciot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer protocol using CICIoT2023 structure.
    Uses 'Protocol Type' column.
    """
    df = df.copy()

    if "Protocol Type" in df.columns:
        proto_map = {
            6: "TCP",
            17: "UDP",
            1: "ICMP"
        }
        df["Protocol"] = df["Protocol Type"].map(proto_map).fillna("OTHER")
    else:
        df["Protocol"] = "OTHER"

    return df


def aggregate_time_bins_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight aggregation for packet-level datasets.
    Matches CIC_IoT2024_IDAD_primary notebook behavior.
    """

    agg = df.groupby("time_bin").agg(
        event_count=("Protocol", "size"),
        TCP=("Protocol", lambda x: (x == "TCP").sum()),
        UDP=("Protocol", lambda x: (x == "UDP").sum()),
        ICMP=("Protocol", lambda x: (x == "ICMP").sum()),
        OTHER=("Protocol", lambda x: (x == "OTHER").sum()),
    ).reset_index().fillna(0).sort_values("time_bin").reset_index(drop=True)

    return agg