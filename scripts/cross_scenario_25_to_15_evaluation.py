# =============================================================================
# CROSS-SCENARIO 25 -> 15 EVALUATION PIPELINE
#
# This script covers the complete 25 -> 15 experiment:
#   1) Raw OMNeT++ data -> compact node-level datasets (59 behavioral features)
#   2) Leakage removal (59 -> 56)
#   3) Source-only grouped 5-fold statistical filtering + union
#   4) Matched external evaluation with RF / DT / SVM / KNN / MLP
#   5) Run-level bootstrap confidence intervals (2,000 replicates)
#   6) KS distribution-shift analysis on the proposed six features
#      with 2,000 complete-run bootstrap replicates
#   7) Machine-readable metrics, KS results, retained features, and protocol audit
#
# Source scenario : 25 nodes, attacker host[15]
# Target scenario : 15 nodes, attacker host[8]
# =============================================================================

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.exceptions import ConvergenceWarning

try:
    from IPython.display import display
except ImportError:
    display = None


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
N_INNER_FOLDS = 5
CORR_THRESHOLD = 0.99
N_BOOTSTRAP = 2000
KS_BOOTSTRAP = 2000
CONFIDENCE_LEVEL = 0.95

RAW_DATA_DIR = Path(r"F:/2026/RAWDATASET")

# By default, validate and reuse existing compact CSVs.
# Missing files are built from RAW; REBUILD_COMPACT_DATASETS=True explicitly rebuilds them.
COMPACT_DIR = Path(r"G:/FINAL_BEHAVIORAL_DATASETS/CROSS_25_15_FULL_PIPELINE")
OUTPUT_DIR = Path(r"G:/FINAL_BEHAVIORAL_DATASETS/CROSS_25_15_FULL_PIPELINE_B1_GROUPED")
RESULTS_DIR = OUTPUT_DIR / "evaluation"
REBUILD_COMPACT_DATASETS = False  # True only if deliberately recreating compact CSVs
COMPACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_25_RAW = RAW_DATA_DIR / "25NODE1BLACK500RUN600S.csv"
TARGET_15_RAW = RAW_DATA_DIR / "15NODE1BLACK500RUN600S.csv"

SOURCE_25_COMPACT = COMPACT_DIR / "compact_25_FULL59_WITH_RUN.csv"
TARGET_15_COMPACT = COMPACT_DIR / "compact_15_FULL59_WITH_RUN.csv"

SOURCE_PATH = SOURCE_25_COMPACT
TARGET_PATH = TARGET_15_COMPACT

EXPECTED_RAW_FEATURES = 59

LEAKAGE_FEATURES = [
    "rreqDropCount:count",
    "rrepSpoofedCount:count",
    "dataDropCount:count",
]

PROPOSED_FEATURES = [
    "numRREQsSent:count",
    "numRREPsSent:count",
    "packetSent:count",
    "numPacketsForwarded:count",
    "arpReplySent:count",
    "linkBroken:count",
]

METADATA_COLUMNS = {
    "run",
    "host_id_full",
    "node_id",
    "run_condition",
    "is_attacker",
}

REPRESENTATION_ORDER = [
    "Full non-leakage",
    "Statistical filtering",
    "Proposed 6 features",
]

CLASSIFIER_ORDER = ["RF", "DT", "SVM", "KNN", "MLP"]

SCENARIOS = [
    {
        "name": "25-NODE SOURCE",
        "raw_path": SOURCE_25_RAW,
        "output_path": SOURCE_25_COMPACT,
        "num_nodes": 25,
        "expected_runs": 1000,
        "attacker_ids": ["host[15]"],
    },
    {
        "name": "15-NODE TARGET",
        "raw_path": TARGET_15_RAW,
        "output_path": TARGET_15_COMPACT,
        "num_nodes": 15,
        "expected_runs": 1000,
        "attacker_ids": ["host[8]"],
    },
]


# =============================================================================
# 2. RAW OMNeT++ DATA -> COMPACT NODE-LEVEL DATASETS
# =============================================================================

def build_compact_dataset(cfg):
    print("\n" + "=" * 120)
    print(cfg["name"])
    print("=" * 120)
    print("RAW   :", cfg["raw_path"])
    print("OUTPUT:", cfg["output_path"])

    if not cfg["raw_path"].exists():
        raise FileNotFoundError(f"Raw file not found:\n{cfg['raw_path']}")

    # Read only the four columns used by the node-profile pipeline.
    df_raw = pd.read_csv(
        cfg["raw_path"],
        usecols=["run", "module", "name", "value"],
        low_memory=False,
    )

    print("\n[1] RAW LOAD")
    print("Raw shape:", df_raw.shape)

    df_raw["host_id_full"] = (
        df_raw["module"].astype(str).str.extract(
            r"(AODVNetwork\.host\[\d+\])",
            expand=False,
        )
    )

    df_raw = df_raw[df_raw["host_id_full"].notna()].copy()
    print("Rows after host extraction:", len(df_raw))

    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    df_raw = df_raw[df_raw["value"].notna()].copy()

    print("\n[2] BUILD NODE-LEVEL MATRIX")
    df = (
        df_raw.pivot_table(
            index=["run", "host_id_full"],
            columns="name",
            values="value",
        )
        .reset_index()
        .fillna(0)
    )

    del df_raw
    gc.collect()

    print("Node-level matrix shape before metadata:", df.shape)

    host_short = df["host_id_full"].astype(str).str.extract(
        r"(host\[\d+\])",
        expand=False,
    )

    is_attack_run = df["run"].astype(str).str.contains(
        "Attacker",
        case=False,
        na=False,
    )

    attacker_ids = set(cfg["attacker_ids"])

    df["is_attacker"] = (
        is_attack_run & host_short.isin(attacker_ids)
    ).astype(int)

    df["run_condition"] = np.where(
        is_attack_run,
        "Attack",
        "Normal",
    )

    df["node_id"] = (
        host_short
        .str.extract(r"host\[(\d+)\]", expand=False)
        .astype(int)
    )

    print("\n[3] STRUCTURE AUDIT")

    run_table = (
        df[["run", "run_condition"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rows_per_run = df.groupby("run").size()
    unique_runs = int(run_table.shape[0])

    print("Unique runs            :", unique_runs)
    print(
        "Rows/run min/median/max:",
        int(rows_per_run.min()),
        float(rows_per_run.median()),
        int(rows_per_run.max()),
    )
    print("Run conditions:")
    print(
        run_table["run_condition"]
        .value_counts()
        .sort_index()
    )
    print("Node labels:")
    print(
        df["is_attacker"]
        .value_counts()
        .sort_index()
    )

    if unique_runs != cfg["expected_runs"]:
        raise RuntimeError(
            f"{cfg['name']}: expected {cfg['expected_runs']} runs, "
            f"found {unique_runs}."
        )

    if (
        rows_per_run.min() != cfg["num_nodes"]
        or rows_per_run.max() != cfg["num_nodes"]
    ):
        raise RuntimeError(
            f"{cfg['name']}: each run must contain exactly "
            f"{cfg['num_nodes']} nodes."
        )

    attack_runs = int(
        (run_table["run_condition"] == "Attack").sum()
    )

    expected_attacker_rows = (
        attack_runs
        * len(cfg["attacker_ids"])
    )

    actual_attacker_rows = int(
        df["is_attacker"].sum()
    )

    print("Expected attacker rows:", expected_attacker_rows)
    print("Actual attacker rows  :", actual_attacker_rows)

    if expected_attacker_rows != actual_attacker_rows:
        raise RuntimeError(
            f"{cfg['name']}: attacker-label audit failed."
        )

    metadata_columns = {
        "run",
        "host_id_full",
        "node_id",
        "run_condition",
        "is_attacker",
    }

    behavioral_features = [
        c
        for c in df.columns
        if c not in metadata_columns
    ]

    print("\n[4] FEATURE AUDIT")
    print(
        "Behavioral feature count:",
        len(behavioral_features),
    )

    if len(behavioral_features) != EXPECTED_RAW_FEATURES:
        print("Detected features:")
        for feature in behavioral_features:
            print(" -", feature)

        raise RuntimeError(
            f"{cfg['name']}: expected "
            f"{EXPECTED_RAW_FEATURES} behavioral features, "
            f"found {len(behavioral_features)}."
        )

    compact_columns = [
        "run",
        "host_id_full",
        "node_id",
        "run_condition",
        "is_attacker",
    ] + behavioral_features

    compact = df[
        compact_columns
    ].copy()

    compact.to_csv(
        cfg["output_path"],
        index=False,
    )

    print("\n[5] SAVED")
    print("Compact shape:", compact.shape)
    print("Saved to:", cfg["output_path"])

    audit = {
        "Scenario": cfg["name"],
        "Runs": unique_runs,
        "Node_Rows": int(len(compact)),
        "Behavioral_Features": len(behavioral_features),
        "Normal_Runs": int(
            (run_table["run_condition"] == "Normal").sum()
        ),
        "Attack_Runs": attack_runs,
        "Benign_Rows": int(
            (compact["is_attacker"] == 0).sum()
        ),
        "Blackhole_Rows": actual_attacker_rows,
        "Output": str(cfg["output_path"]),
    }

    del compact, df
    gc.collect()

    return audit


def validate_existing_compact(cfg):
    """Audit existing compact data before reuse; do not silently trust a path."""
    compact_path = cfg["output_path"]
    frame = pd.read_csv(compact_path, low_memory=False)
    needed = {"run", "host_id_full", "node_id", "run_condition", "is_attacker"}
    if not needed.issubset(frame.columns):
        raise RuntimeError(f"{cfg['name']}: missing required metadata in {compact_path}")
    if frame["run"].isna().any() or frame["host_id_full"].isna().any():
        raise RuntimeError(f"{cfg['name']}: missing run/host values")
    run_sizes = frame.groupby("run").size()
    if (len(run_sizes) != cfg["expected_runs"]
            or run_sizes.min() != cfg["num_nodes"]
            or run_sizes.max() != cfg["num_nodes"]):
        raise RuntimeError(f"{cfg['name']}: compact run/size audit failed")
    run_table = frame[["run", "run_condition"]].drop_duplicates()
    if len(run_table) != cfg["expected_runs"]:
        raise RuntimeError(f"{cfg['name']}: inconsistent run_condition per run")
    attack_runs = int((run_table["run_condition"] == "Attack").sum())
    if set(run_table["run_condition"].astype(str)) != {"Attack", "Normal"}:
        raise RuntimeError(f"{cfg['name']}: unsupported run conditions")
    host_short = frame["host_id_full"].astype(str).str.extract(
        r"(host\[\d+\])", expand=False
    )
    expected_node_labels = (
        frame["run_condition"].eq("Attack")
        & host_short.isin(set(cfg["attacker_ids"]))
    ).astype(int)
    if not np.array_equal(expected_node_labels.to_numpy(),
                          frame["is_attacker"].to_numpy(dtype=int)):
        raise RuntimeError(f"{cfg['name']}: cached attack-label audit failed")
    features = [c for c in frame.columns if c not in METADATA_COLUMNS]
    if len(features) != EXPECTED_RAW_FEATURES:
        raise RuntimeError(f"{cfg['name']}: expected 59 behavioral features, got {len(features)}")
    if frame[features].isna().any().any():
        raise RuntimeError(f"{cfg['name']}: NaNs present in existing compact feature matrix")
    report = {
        "Scenario": cfg["name"],
        "Runs": len(run_sizes),
        "Node_Rows": len(frame),
        "Behavioral_Features": len(features),
        "Normal_Runs": len(run_sizes) - attack_runs,
        "Attack_Runs": attack_runs,
        "Benign_Rows": int((frame["is_attacker"] == 0).sum()),
        "Blackhole_Rows": int((frame["is_attacker"] == 1).sum()),
        "Output": str(compact_path),
        "Preparation": "REUSED_VERIFIED",
    }
    del frame
    gc.collect()
    print(f"{cfg['name']}: existing compact reused and audited — PASS")
    return report


preparation_audits = []
for cfg in SCENARIOS:
    if cfg["output_path"].exists() and not REBUILD_COMPACT_DATASETS:
        preparation_audits.append(validate_existing_compact(cfg))
    else:
        record = build_compact_dataset(cfg)
        record["Preparation"] = "REBUILT_FROM_RAW"
        preparation_audits.append(record)
    gc.collect()


preparation_audit_df = pd.DataFrame(
    preparation_audits
)

preparation_audit_path = (
    OUTPUT_DIR
    / "compact_25_15_preparation_audit.csv"
)

preparation_audit_df.to_csv(
    preparation_audit_path,
    index=False,
)

print("\n" + "=" * 130)
print(
    "25 -> 15 COMPACT DATASET PREPARATION: COMPLETE"
)
print("=" * 130)

print(
    preparation_audit_df.to_string(
        index=False
    )
)

print(
    "\nSaved preparation audit:",
    preparation_audit_path,
)



# =============================================================================
# 3. MATCHED EXTERNAL EVALUATION
# =============================================================================

print("=" * 140)
print("SECTION 4.2 — 25 -> 15 MATCHED EXTERNAL EVALUATION")
print("FULL 56 / SOURCE-ONLY GROUPED B1 UNION / PROPOSED 6")
print("RF / DT / SVM / KNN / MLP")
print("=" * 140)

if not SOURCE_PATH.exists() or not TARGET_PATH.exists():
    raise FileNotFoundError("Compact datasets were not created successfully.")

source_df = pd.read_csv(SOURCE_PATH, low_memory=False)
target_df = pd.read_csv(TARGET_PATH, low_memory=False)

print("\n[1] COMPACT DATASETS")
print("25-node source:", source_df.shape)
print("15-node target:", target_df.shape)

if source_df["run"].nunique() != 1000:
    raise RuntimeError("25-node source must contain 1000 runs.")
if target_df["run"].nunique() != 1000:
    raise RuntimeError("15-node target must contain 1000 runs.")

if source_df.groupby("run").size().min() != 25 or source_df.groupby("run").size().max() != 25:
    raise RuntimeError("25-node run structure failed.")
if target_df.groupby("run").size().min() != 15 or target_df.groupby("run").size().max() != 15:
    raise RuntimeError("15-node run structure failed.")

print("25-node run structure: PASS")
print("15-node run structure: PASS")


# =============================================================================
# FEATURE SCHEMA + FULL 56
# =============================================================================

source_raw_features = [c for c in source_df.columns if c not in METADATA_COLUMNS]
target_raw_features = [c for c in target_df.columns if c not in METADATA_COLUMNS]

print("\n[2] RAW FEATURE SCHEMA")
print("Source raw behavioral features:", len(source_raw_features))
print("Target raw behavioral features:", len(target_raw_features))

if len(source_raw_features) != 59 or len(target_raw_features) != 59:
    raise RuntimeError("Expected 59 behavioral features in both compact datasets.")

missing_source_features_in_target = [
    f for f in source_raw_features if f not in target_raw_features
]
extra_target_features = [
    f for f in target_raw_features if f not in source_raw_features
]

print("Source features missing in target:", missing_source_features_in_target)
print("Extra target-only features:", extra_target_features)

if missing_source_features_in_target:
    raise RuntimeError(
        "25->15 schema mismatch. Do NOT use a target-driven feature intersection."
    )

missing_leakage = [f for f in LEAKAGE_FEATURES if f not in source_raw_features]
if missing_leakage:
    raise RuntimeError(f"Missing predefined leakage variables: {missing_leakage}")

full_features = [f for f in source_raw_features if f not in LEAKAGE_FEATURES]
if len(full_features) != 56:
    raise RuntimeError(f"Expected 56 non-leakage features, found {len(full_features)}.")

print("Full non-leakage count:", len(full_features))


# =============================================================================
# SOURCE-ONLY STATISTICAL FILTER — GROUPED 5-FOLD UNION
# =============================================================================
# External transfer uses all X25 source runs, independently of the
# intra-dataset TRAIN70/HOLDOUT30 split. In each grouped fold, fit
# zero-variance and Pearson (|r| >= 0.99) filters on inner-training runs
# only, resolving correlated features by source inner-training label
# correlation. Use the union of the five retained feature sets.
# X15 target data and labels do not influence source feature selection.
# Final classifiers are trained on all X25 runs for each representation.
# =============================================================================

def fit_b1_post_filters(X_inner_train, y_inner_train, corr_threshold=0.99):
    """Source-only zero-variance and Pearson pruning with label-based tie-break."""
    selector = VarianceThreshold(threshold=0.0)
    X_zv_np = selector.fit_transform(X_inner_train)
    after_zv = X_inner_train.columns[selector.get_support()].tolist()
    if not after_zv:
        raise RuntimeError("B1 removed all source features at zero-variance stage")

    X_zv = pd.DataFrame(X_zv_np, columns=after_zv, index=X_inner_train.index)
    corr_matrix = X_zv.corr().abs()
    target_corr = X_zv.apply(
        lambda col: col.corr(y_inner_train)
    ).abs().fillna(0.0)
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    auto_drop = set()
    for col in upper_tri.columns:
        correlated_features = upper_tri.index[
            upper_tri[col] >= corr_threshold
        ].tolist()
        for feature in correlated_features:
            if target_corr[col] >= target_corr[feature]:
                auto_drop.add(feature)
            else:
                auto_drop.add(col)

    after_corr = [feature for feature in after_zv if feature not in auto_drop]
    if not after_corr:
        raise RuntimeError("B1 removed all source features at Pearson stage")
    return after_zv, after_corr


def fit_source_only_b1_inner_union(source, feature_columns, n_folds=5, seed=42):
    """B1 selection: five grouped, stratified inner-training fits then union."""
    # Preserve the source run order from the compact dataset.
    run_table = source[["run", "run_condition"]].drop_duplicates().reset_index(drop=True)
    if run_table["run"].nunique() != len(run_table):
        raise RuntimeError("A simulation run has conflicting condition labels")
    if (run_table["run_condition"].value_counts().min() < n_folds
            or set(run_table["run_condition"]) != {"Normal", "Attack"}):
        raise RuntimeError("Insufficient Normal/Attack source runs for grouped folds")

    source_run_series = source["run"].astype(str)
    run_table["run"] = run_table["run"].astype(str)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    inner_feature_sets = []
    inner_zero_sets = []
    audit_records = []
    selected_frequency = {feature: 0 for feature in feature_columns}

    for fold_id, (fit_ids, val_ids) in enumerate(
        cv.split(np.zeros(len(run_table)), run_table["run_condition"]), start=1
    ):
        fit_run_ids = set(run_table.iloc[fit_ids]["run"])
        val_run_ids = set(run_table.iloc[val_ids]["run"])
        if fit_run_ids.intersection(val_run_ids):
            raise RuntimeError("B1 source inner-fold run overlap detected")
        train_idx = np.flatnonzero(source_run_series.isin(fit_run_ids).to_numpy())
        valid_idx = np.flatnonzero(source_run_series.isin(val_run_ids).to_numpy())
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise RuntimeError("B1 source inner fold is empty")
        X_inner = source.iloc[train_idx][feature_columns].copy()
        y_inner = source.iloc[train_idx]["is_attacker"].astype(int).copy()
        after_zv, after_corr = fit_b1_post_filters(
            X_inner, y_inner, corr_threshold=CORR_THRESHOLD
        )
        inner_feature_sets.append(set(after_corr))
        inner_zero_sets.append(set(after_zv))
        for feature in after_corr:
            selected_frequency[feature] += 1
        audit_records.append({
            "Inner_Fold": fold_id,
            "Source_Training_Runs": len(fit_run_ids),
            "Source_Validation_Runs": len(val_run_ids),
            "Training_Node_Rows": len(train_idx),
            "Validation_Node_Rows": len(valid_idx),
            "Run_Overlap": 0,
            "After_ZeroVariance": len(after_zv),
            "After_Pearson": len(after_corr),
            "Selected_Features": "|".join(after_corr),
            "Label_Scope": "SOURCE_INNER_TRAIN_ONLY",
            "Target_15_Used": False,
        })
        print(
            f"B1 SOURCE inner fold {fold_id}: train_runs={len(fit_run_ids)} "
            f"| valid_runs={len(val_run_ids)} | after ZV={len(after_zv)} "
            f"| after Pearson={len(after_corr)}"
        )
        del X_inner, y_inner

    after_zv_union = sorted(set.union(*inner_zero_sets))
    statistical_union = sorted(set.union(*inner_feature_sets))
    if not statistical_union or not set(statistical_union).issubset(feature_columns):
        raise RuntimeError("B1 source-only union contains invalid features")
    return {
        "after_zero_variance_union": after_zv_union,
        "statistical_features": statistical_union,
        "selection_frequency": selected_frequency,
        "fold_audit": pd.DataFrame(audit_records),
    }


X_source_full = source_df[full_features].copy()
y_source = source_df["is_attacker"].astype(int).copy()

b1_source_result = fit_source_only_b1_inner_union(
    source_df, full_features, n_folds=N_INNER_FOLDS, seed=RANDOM_STATE
)
after_zv_features = b1_source_result["after_zero_variance_union"]
statistical_features = b1_source_result["statistical_features"]
source_selection_frequency = b1_source_result["selection_frequency"]
source_inner_fold_audit = b1_source_result["fold_audit"]
corr_dropped = sorted(set(after_zv_features) - set(statistical_features))

# Descriptive correlation is calculated after selection, only for the
# machine-readable audit. It DOES NOT determine which features are selected.
source_label_corr = X_source_full.apply(
    lambda col: col.corr(y_source)
).abs().fillna(0.0)

print("\n[3] B1-ALIGNED SOURCE-ONLY STATISTICAL FILTER")
print("Input non-leakage features  :", len(full_features))
print("Inner grouped folds         :", N_INNER_FOLDS)
print("Source training scope       : ALL 1000 X25 RUNS")
print("After ZV (five-fold union)  :", len(after_zv_features))
print("After Pearson (fold union)  :", len(statistical_features))
print("Features excluded by union :", len(corr_dropped))
print("15-node used for filtering  : NO")
print("No assumed 25-feature count: number comes from FULL source run folds")
print("\nStatistically retained source features:")
for i, feature in enumerate(statistical_features, 1):
    print(
        f"{i:02d}. {feature:<48s} "
        f"| fold_frequency={source_selection_frequency[feature]}/{N_INNER_FOLDS}"
    )

source_inner_fold_audit_path = RESULTS_DIR / "25_TO_15_B1_SOURCE_INNER_FOLD_AUDIT.csv"
source_inner_fold_audit.to_csv(source_inner_fold_audit_path, index=False)

missing_stat_target = [f for f in statistical_features if f not in target_df.columns]
if missing_stat_target:
    raise RuntimeError(
        "Source-selected Statistical representation cannot be applied unchanged "
        f"to target. Missing: {missing_stat_target}"
    )

print("\nStatistically retained features:")
for i, feature in enumerate(statistical_features, 1):
    print(
        f"{i:02d}. {feature:<48s} "
        f"|corr_source_label|={source_label_corr[feature]:.6f}"
    )


# =============================================================================
# PROPOSED SIX + REPRESENTATIONS
# =============================================================================

for feature in PROPOSED_FEATURES:
    if feature not in source_df.columns or feature not in target_df.columns:
        raise RuntimeError(f"Proposed feature missing: {feature}")

representations = {
    "Full non-leakage": full_features,
    "Statistical filtering": statistical_features,
    "Proposed 6 features": PROPOSED_FEATURES,
}

print("\n[4] REPRESENTATION SUMMARY")
for name in REPRESENTATION_ORDER:
    print(f"{name:<26s}: {len(representations[name])} features")


# =============================================================================
# MODELS
# =============================================================================

def make_models():
    return [
        (
            "RF",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            False,
        ),
        (
            "DT",
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            False,
        ),
        (
            "SVM",
            SVC(kernel="rbf", class_weight="balanced"),
            True,
        ),
        (
            "KNN",
            KNeighborsClassifier(n_neighbors=5, weights="distance"),
            True,
        ),
        (
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
    ]


def point_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "F1_Macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Specificity": float(specificity),
        "FPR": float(fpr),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def build_run_confusion_table(target_dataframe, y_pred):
    temp = pd.DataFrame({
        "run": target_dataframe["run"].astype(str).to_numpy(),
        "y_true": target_dataframe["is_attacker"].astype(int).to_numpy(),
        "y_pred": np.asarray(y_pred, dtype=int),
    })
    rows = []
    for run_id, group in temp.groupby("run", sort=False):
        tn, fp, fn, tp = confusion_matrix(
            group["y_true"], group["y_pred"], labels=[0, 1]
        ).ravel()
        rows.append({
            "run": run_id,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })
    return pd.DataFrame(rows)


def metrics_from_counts(tn, fp, fn, tp):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return p, r, f1, fpr


# =============================================================================
# COMMON TARGET-RUN BOOTSTRAP
# =============================================================================

target_run_order = target_df["run"].astype(str).drop_duplicates().to_numpy()
n_target_runs = len(target_run_order)

rng_bootstrap = np.random.default_rng(RANDOM_STATE)
COMMON_BOOTSTRAP_INDICES = rng_bootstrap.integers(
    0,
    n_target_runs,
    size=(N_BOOTSTRAP, n_target_runs),
)

print("\n[5] BOOTSTRAP DESIGN")
print("Target runs                    :", n_target_runs)
print("Bootstrap replicates           :", N_BOOTSTRAP)
print("Bootstrap unit                 : COMPLETE TARGET RUN")
print("Same run draws for all 15 tests: YES")
print("Model retraining in bootstrap : NO")


def bootstrap_ci(run_table):
    run_order = run_table["run"].astype(str).to_numpy()
    if not np.array_equal(run_order, target_run_order):
        raise RuntimeError("Target-run ordering changed.")

    counts = run_table[["TN", "FP", "FN", "TP"]].to_numpy(dtype=np.int64)
    boot = np.empty((N_BOOTSTRAP, 4), dtype=float)

    for b in range(N_BOOTSTRAP):
        tn, fp, fn, tp = counts[COMMON_BOOTSTRAP_INDICES[b]].sum(axis=0)
        boot[b] = metrics_from_counts(tn, fp, fn, tp)

    return {
        "Precision": tuple(np.percentile(boot[:, 0], [2.5, 97.5]).astype(float)),
        "Recall": tuple(np.percentile(boot[:, 1], [2.5, 97.5]).astype(float)),
        "F1": tuple(np.percentile(boot[:, 2], [2.5, 97.5]).astype(float)),
        "FPR": tuple(np.percentile(boot[:, 3], [2.5, 97.5]).astype(float)),
    }


# =============================================================================
# MATCHED 3 x 5 BENCHMARK
# =============================================================================

results = []
prediction_frames = []

y_target = target_df["is_attacker"].astype(int).copy()

print("\n" + "=" * 150)
print("RUNNING MATCHED 25 -> 15 EXTERNAL BENCHMARK")
print("=" * 150)

for representation_name in REPRESENTATION_ORDER:
    features = representations[representation_name]

    X_source = source_df[features].copy()
    X_target = target_df[features].copy()

    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)

    print("\n" + "-" * 130)
    print(f"{representation_name} ({len(features)} features)")
    print("Scaler fit on 25-node source only: PASS")
    print("15-node used to fit scaler       : NO")
    print("-" * 130)

    for classifier_name, model, use_scaled in make_models():
        fit_X = X_source_scaled if use_scaled else X_source
        test_X = X_target_scaled if use_scaled else X_target

        model.fit(fit_X, y_source)
        y_pred = model.predict(test_X)

        # -------------------------------------------------------------
        # Machine-readable target-node predictions for reproducibility.
        # One row per target node for each representation/classifier.
        # -------------------------------------------------------------
        prediction_frames.append(
            pd.DataFrame({
                "Transfer": "25→15",
                "Representation": representation_name,
                "Features": len(features),
                "Classifier": classifier_name,
                "run": target_df["run"].astype(str).to_numpy(),
                "host_id_full": target_df["host_id_full"].astype(str).to_numpy(),
                "node_id": target_df["node_id"].astype(int).to_numpy(),
                "run_condition": target_df["run_condition"].astype(str).to_numpy(),
                "y_true": y_target.to_numpy(dtype=int),
                "y_pred": np.asarray(y_pred, dtype=int),
            })
        )

        point = point_metrics(y_target, y_pred)
        run_table = build_run_confusion_table(target_df, y_pred)
        ci = bootstrap_ci(run_table)

        print(
            f"{classifier_name:4s} | "
            f"P={point['Precision']:.6f} [{ci['Precision'][0]:.6f}, {ci['Precision'][1]:.6f}] | "
            f"R={point['Recall']:.6f} [{ci['Recall'][0]:.6f}, {ci['Recall'][1]:.6f}] | "
            f"F1={point['F1']:.6f} [{ci['F1'][0]:.6f}, {ci['F1'][1]:.6f}] | "
            f"FPR={point['FPR']:.6f} [{ci['FPR'][0]:.6f}, {ci['FPR'][1]:.6f}]"
        )

        results.append({
            "Transfer": "25→15",
            "Representation": representation_name,
            "Features": len(features),
            "Classifier": classifier_name,
            **point,
            "Precision_CI_L": ci["Precision"][0],
            "Precision_CI_U": ci["Precision"][1],
            "Recall_CI_L": ci["Recall"][0],
            "Recall_CI_U": ci["Recall"][1],
            "F1_CI_L": ci["F1"][0],
            "F1_CI_U": ci["F1"][1],
            "FPR_CI_L": ci["FPR"][0],
            "FPR_CI_U": ci["FPR"][1],
        })

results_df = pd.DataFrame(results)
if len(results_df) != 15:
    raise RuntimeError("Expected exactly 15 experiments.")

predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True,
)

expected_prediction_rows = (
    len(target_df)
    * len(REPRESENTATION_ORDER)
    * len(CLASSIFIER_ORDER)
)

if len(predictions_df) != expected_prediction_rows:
    raise RuntimeError(
        "Prediction-row audit failed: "
        f"expected {expected_prediction_rows}, "
        f"found {len(predictions_df)}."
    )


# =============================================================================
# KS — PROPOSED SIX ONLY, COMPLETE-RUN BOOTSTRAP
# =============================================================================

def run_matrix(df, feature):
    groups = [
        group[feature].to_numpy(dtype=float)
        for _, group in df.groupby("run", sort=False)
    ]
    lengths = {len(x) for x in groups}
    if len(lengths) != 1:
        raise RuntimeError(f"Inconsistent rows/run for {feature}.")
    return np.vstack(groups)


print("\n" + "=" * 150)
print("KS DISTRIBUTION SHIFT — PROPOSED SIX ONLY")
print("=" * 150)
print("KS bootstrap replicates:", KS_BOOTSTRAP)
print("KS bootstrap unit: COMPLETE SOURCE + TARGET RUN")

ks_rows = []

for feature_index, feature in enumerate(PROPOSED_FEATURES):
    source_matrix = run_matrix(source_df, feature)
    target_matrix = run_matrix(target_df, feature)

    source_values = source_matrix.reshape(-1)
    target_values = target_matrix.reshape(-1)

    point = ks_2samp(
        source_values,
        target_values,
        alternative="two-sided",
        method="auto",
    )

    rng_ks = np.random.default_rng(RANDOM_STATE + 1000 + feature_index)
    n_source_runs = source_matrix.shape[0]
    n_target_runs = target_matrix.shape[0]
    boot_d = np.empty(KS_BOOTSTRAP, dtype=float)

    for b in range(KS_BOOTSTRAP):
        source_idx = rng_ks.integers(0, n_source_runs, size=n_source_runs)
        target_idx = rng_ks.integers(0, n_target_runs, size=n_target_runs)
        source_boot = source_matrix[source_idx].reshape(-1)
        target_boot = target_matrix[target_idx].reshape(-1)
        boot_d[b] = ks_2samp(
            source_boot,
            target_boot,
            alternative="two-sided",
            method="auto",
        ).statistic

    ci_low, ci_high = np.percentile(boot_d, [2.5, 97.5]).astype(float)

    print(
        f"{feature:<42s} D={point.statistic:.6f} "
        f"CI=[{ci_low:.6f}, {ci_high:.6f}] nominal p={point.pvalue:.6g}"
    )

    ks_rows.append({
        "Transfer": "25→15",
        "Feature": feature,
        "KS_D": float(point.statistic),
        "CI_Lower": float(ci_low),
        "CI_Upper": float(ci_high),
        "Nominal_p": float(point.pvalue),
    })

    del source_matrix, target_matrix, source_values, target_values, boot_d
    gc.collect()

ks_df = pd.DataFrame(ks_rows)


# =============================================================================
# RESULTS FORMATTING AND EXPORT
# =============================================================================

def format_ci(point, low, high):
    return f"{point:.4f} [{low:.4f}, {high:.4f}]"


summary_table_df = results_df[[
    "Transfer",
    "Representation",
    "Features",
    "Classifier",
]].copy()

summary_table_df["Precision [95% CI]"] = results_df.apply(
    lambda r: format_ci(r["Precision"], r["Precision_CI_L"], r["Precision_CI_U"]),
    axis=1,
)
summary_table_df["Recall [95% CI]"] = results_df.apply(
    lambda r: format_ci(r["Recall"], r["Recall_CI_L"], r["Recall_CI_U"]),
    axis=1,
)
summary_table_df["F1-score [95% CI]"] = results_df.apply(
    lambda r: format_ci(r["F1"], r["F1_CI_L"], r["F1_CI_U"]),
    axis=1,
)
summary_table_df["FPR [95% CI]"] = results_df.apply(
    lambda r: format_ci(r["FPR"], r["FPR_CI_L"], r["FPR_CI_U"]),
    axis=1,
)

full_results_path = RESULTS_DIR / "25_TO_15_MATCHED_FULL_PRECISION.csv"
summary_table_path = RESULTS_DIR / "25_TO_15_RESULTS_SUMMARY.csv"
ks_path = RESULTS_DIR / "25_TO_15_TABLE9_KS.csv"
stat_path = RESULTS_DIR / "25_TO_15_SOURCE_ONLY_STATISTICAL_FEATURES.csv"
predictions_path = RESULTS_DIR / "25_TO_15_NODE_LEVEL_PREDICTIONS.csv"
audit_path = RESULTS_DIR / "25_TO_15_PROTOCOL_AUDIT.json"

results_df.to_csv(full_results_path, index=False)
summary_table_df.to_csv(summary_table_path, index=False)
ks_df.to_csv(ks_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

pd.DataFrame({
    "Rank": np.arange(1, len(statistical_features) + 1),
    "Feature": statistical_features,
    "Abs_Source_Corr_With_Label": [
        float(source_label_corr[f]) for f in statistical_features
    ],
    "B1_Inner_Fold_Selection_Frequency": [
        source_selection_frequency[f] for f in statistical_features
    ],
}).to_csv(stat_path, index=False)

with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "transfer": "25→15",
            "source_runs": int(source_df["run"].nunique()),
            "source_statistical_selection_method": "B1_5_GROUPED_INNER_FOLDS_UNION",
            "source_statistical_inner_folds": N_INNER_FOLDS,
            "source_statistical_selection_scope": "ALL_1000_SOURCE_RUNS_ONLY",
            "source_inner_fold_audit": str(source_inner_fold_audit_path),
            "target_runs": int(target_df["run"].nunique()),
            "source_raw_features": len(source_raw_features),
            "target_raw_features": len(target_raw_features),
            "full_nonleakage_features": len(full_features),
            "statistical_features": len(statistical_features),
            "proposed_features": len(PROPOSED_FEATURES),
            "classification_bootstrap_replicates": N_BOOTSTRAP,
            "ks_bootstrap_replicates": KS_BOOTSTRAP,
            "prediction_rows": int(len(predictions_df)),
            "predictions_file": str(predictions_path),
            "target_used_for_feature_filtering": False,
            "target_used_for_scaler_fit": False,
            "target_used_for_model_fit": False,
            "target_driven_feature_intersection": False,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\n" + "=" * 190)
print("25 -> 15 FULL-PRECISION RESULTS")
print("=" * 190)
print(
    results_df[[
        "Representation", "Features", "Classifier",
        "Precision", "Recall", "F1", "FPR",
        "TN", "FP", "FN", "TP",
    ]].to_string(index=False, float_format=lambda x: f"{x:.8f}")
)

print("\n" + "=" * 210)
print("25 -> 15 RESULTS SUMMARY")
print("=" * 210)
print(summary_table_df.to_string(index=False))

print("\n" + "=" * 150)
print("25 -> 15 KS TABLE")
print("=" * 150)
print(ks_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\n" + "=" * 150)
print("MACHINE-READABLE NODE-LEVEL PREDICTIONS")
print("=" * 150)
print("Prediction rows                        :", len(predictions_df))
print("Expected prediction rows               :", expected_prediction_rows)
print("Representations × classifiers          :", len(REPRESENTATION_ORDER), "×", len(CLASSIFIER_ORDER))
print("Saved predictions                      :", predictions_path)

prediction_preview = predictions_df.head(20)

if display is not None:
    print("\nJUPYTER VIEW — FULL-PRECISION METRICS")
    display(results_df)

    print("\nJUPYTER VIEW — RESULTS SUMMARY METRICS")
    display(summary_table_df)

    print("\nJUPYTER VIEW — KS RESULTS")
    display(ks_df)

    print("\nJUPYTER VIEW — PREDICTIONS PREVIEW (first 20 rows)")
    display(prediction_preview)
else:
    print("\nPredictions preview (first 20 rows):")
    print(prediction_preview.to_string(index=False))

print("\n" + "=" * 150)
print("FINAL 25 -> 15 PROTOCOL AUDIT")
print("=" * 150)
print("Full non-leakage features             :", len(full_features))
print("Source-only B1 five-fold union feats  :", len(statistical_features))
print("Source-only B1 inner-fold audit        :", source_inner_fold_audit_path)
print("Proposed features                     :", len(PROPOSED_FEATURES))
print("Target used for statistical filtering : NO")
print("Target used to define Proposed-6       : NO")
print("Target used to fit scaler              : NO")
print("Target used for model fitting          : NO")
print("Target-driven feature intersection     : NO")
print("Bootstrap unit                        : COMPLETE TARGET RUN")
print("Node-level bootstrap                   : NO")
print("Model retraining during bootstrap      : NO")
print("Total experiments                     :", len(results_df))
print("Machine-readable predictions           : YES")
print("Prediction rows                       :", len(predictions_df))
print("Predictions visible in Jupyter         : PREVIEW + FULL METRIC TABLES")
print("\n25 -> 15 MATCHED EXTERNAL EVALUATION: PASS")

print("\nSaved outputs:")
for p in [
    full_results_path,
    summary_table_path,
    ks_path,
    stat_path,
    source_inner_fold_audit_path,
    predictions_path,
    audit_path,
]:
    print(p)
