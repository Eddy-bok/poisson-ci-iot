import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from src.poisson_ci_detector import ImprovedPoissonConcentrationML
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)

def poisson_diagnostic(df_agg, label="Dataset", verbose=False):
    """
    Enhanced Poisson diagnostic:
    Evaluates all numeric, integer-like columns as potential count-based features.
    Returns mean, variance, VMR, and dispersion classification.
    Prints only when verbose=True.
    """

    numeric_cols = df_agg.select_dtypes(include=[np.number]).columns.tolist()
    count_cols = [c for c in numeric_cols if "count" in c.lower()]

    int_like_cols = [
        c for c in numeric_cols
        if c not in count_cols
        and (df_agg[c].min() >= 0)
        and np.allclose(df_agg[c] % 1, 0, atol=1e-3)
    ]

    selected_cols = count_cols + int_like_cols

    if not selected_cols:
        if verbose:
            print("[WARN] No numeric count-like columns found for diagnostic.")
        return pd.DataFrame()

    results = []

    for col in selected_cols:
        vals = np.nan_to_num(df_agg[col].values, nan=0.0)
        mean_val = np.mean(vals)
        var_val = np.var(vals)
        vmr = var_val / (mean_val + 1e-8)

        if vmr > 1.1:
            status = "Over-dispersed"
        elif vmr < 0.7:
            status = "Under-dispersed"
        else:
            status = "≈ Poisson"

        results.append({
            "Feature": col,
            "Mean": mean_val,
            "Variance": var_val,
            "VMR": vmr,
            "Status": status
        })

    df_results = pd.DataFrame(results)

    if verbose:
        print(f"\n=== Poisson Behaviour Diagnostic — {label} ===")
        print(df_results.to_string(index=False, formatters={
            "Mean": "{:.3f}".format,
            "Variance": "{:.3f}".format,
            "VMR": "{:.3f}".format
        }))

        summary = df_results["Status"].value_counts().to_dict()
        print("\n[SUMMARY]")
        for k, v in summary.items():
            print(f"  {k:15s}: {v} features")

        print(f"\n[AVERAGE VMR] {df_results['VMR'].mean():.3f}")

    return df_results



def eval_baseline(anom_attack, anom_benign):
    TP = int(np.sum(anom_attack == 1))
    FN = int(np.sum(anom_attack == 0))
    FP = int(np.sum(anom_benign == 1))
    TN = int(np.sum(anom_benign == 0))

    recall = TP / (TP + FN) if (TP + FN) else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    fpr = FP / (FP + TN) if (FP + TN) else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "Recall": recall, "Precision": precision,
        "F1": f1, "FPR": fpr
    }


# ============================================================
# DETECTION DELAY FUNCTION (REUSABLE)
# ============================================================
def compute_delay(anom_vec, time_series, label):
    if not np.any(anom_vec):
        print(f"{label}: No detection")
        return None

    first_idx = np.where(anom_vec)[0][0]
    delay = (time_series[first_idx] - time_series[0]) / np.timedelta64(1, "s")

    print(f"{label}: delay={delay:.6f}s, index={first_idx}")
    return float(delay)


def combined_stream_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    fpr       = np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0)

    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = np.nan

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "ROC-AUC": auc
    }



def f1_at_target_fpr(true_labels, scores, target_fpr=0.01):
    true_labels = np.asarray(true_labels).astype(int)
    scores = np.asarray(scores).astype(float)

    fpr_arr, tpr_arr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr_arr, tpr_arr)

    idx = min(np.searchsorted(fpr_arr, target_fpr), len(fpr_arr) - 1)

    actual_fpr = fpr_arr[idx]
    recall = tpr_arr[idx]

    n_pos = np.sum(true_labels == 1)
    n_neg = np.sum(true_labels == 0)

    tp = recall * n_pos
    fp = actual_fpr * n_neg

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "AUC": roc_auc,
        f"F1@FPR={target_fpr}": f1,
        f"Recall@FPR={target_fpr}": recall,
        "ActualFPR": actual_fpr,
    }



def compute_metrics(flags, labels):
    flags = np.asarray(flags).astype(bool)
    labels = np.asarray(labels).astype(int)

    tp = np.sum(flags & (labels == 1))
    fp = np.sum(flags & (labels == 0))
    fn = np.sum(~flags & (labels == 1))
    tn = np.sum(~flags & (labels == 0))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {"Prec": prec, "Rec": rec, "F1": f1, "FPR": fpr, "TP": tp, "FP": fp, "FN": fn, "TN": tn}




def evaluate_at_fprs(true_labels, scores, target_fprs):
    fpr_arr, tpr_arr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr_arr, tpr_arr)
    results = {}
    for tfpr in target_fprs:
        idx = min(np.searchsorted(fpr_arr, tfpr), len(fpr_arr) - 1)
        actual_fpr = fpr_arr[idx]
        recall = tpr_arr[idx]

        n_pos = np.sum(true_labels == 1)
        n_neg = np.sum(true_labels == 0)

        tp = recall * n_pos
        fp = actual_fpr * n_neg

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

        results[tfpr] = {
            'Recall': recall,
            'Prec': prec,
            'F1': f1,
            'ActualFPR': actual_fpr
        }

    return roc_auc, results