import os
from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
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
    f1_score
)

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# X75 MATCHED CLASSIFICATION EVALUATION
# Compare three feature representations using the 75-node simulation dataset:
# full non-leakage, inner-training statistical filtering and the fixed six
# features previously selected from X7, X25 and X50 (X75 was not involved).
# The first run generates a stratified 70/30 partition of simulation run IDs
# directly from RAW and saves it; later runs reuse the verified CSV manifest.
# The run ID is the indivisible partition unit; no node-level random split.
# The statistical feature union is fitted only within five TRAIN70 inner folds.
# SVM, KNN and MLP use scalers fitted on TRAIN70; RF and DT use raw features.
# HOLDOUT30 labels are used for evaluation and consistency checks only.
# Attack-class F1 and macro-averaged F1 are reported separately.


# Configure paths through environment variables or use repository-relative defaults.
PROJECT_ROOT = Path(os.environ.get('MANET_PROJECT_ROOT', str(Path.cwd()))).expanduser()
DATA_DIR = Path(os.environ.get('MANET_DATA_DIR', str(PROJECT_ROOT / 'data'))).expanduser()
SPLIT_ROOT = Path(os.environ.get('MANET_SPLITS_DIR', str(PROJECT_ROOT / 'splits'))).expanduser()
RAW_PATH = str(DATA_DIR / '75NODE2BLACK500RUN100S.csv')

# Both the outer 70/30 run split and the five statistical inner folds
# use a fixed random seed; statistical filters never use HOLDOUT30 rows.
SPLIT_SEED = 42
TEST_SIZE = 0.30

FIXED_ATTACKER_IDS = [
    'host[20]',
    'host[40]'
]


LEAKAGE_FEATURES = [
    'rreqDropCount:count',
    'rrepSpoofedCount:count',
    'dataDropCount:count'
]


PROPOSED_FEATURES = [
    'numRREQsSent:count',
    'numRREPsSent:count',
    'packetSent:count',
    'numPacketsForwarded:count',
    'arpReplySent:count',
    'linkBroken:count'
]


CORRELATION_THRESHOLD = 0.99


SAVE_DIR = str(Path(os.environ.get('MANET_X75_OUTPUT', str(PROJECT_ROOT / 'results' / 'X75_MATCHED_EVALUATION'))).expanduser())

# X75 was not part of the global six-feature selection (X7/X25/X50 only).
# Create the X75 run-grouped partition locally if its CSV does not exist.
# Reuse and validate the saved CSV on subsequent runs; never overwrite it.
BENCHMARK_SPLIT_PATH = str(SPLIT_ROOT / '75NODE' / '75node_run_split.csv')
N_INNER_FOLDS = 5
EXPECTED_TOTAL_RUNS = 1000
EXPECTED_TRAIN_RUNS = 700
EXPECTED_TEST_RUNS = 300
EXPECTED_TOTAL_NODE_ROWS = 75000
# The statistical feature-union size is derived from five TRAIN70 inner folds.

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


DETAILED_RESULTS_PATH = os.path.join(
    SAVE_DIR,
    'X75_detailed_classification_metrics.csv'
)

MAIN_TABLE_PATH = os.path.join(
    SAVE_DIR,
    'X75_attack_f1_by_representation.csv'
)

RUN_SPLIT_PATH = os.path.join(
    SAVE_DIR,
    'X75_verified_run_split.csv'
)

FEATURE_AUDIT_PATH = os.path.join(
    SAVE_DIR,
    'X75_feature_representation_audit.txt'
)


print("=" * 120)
print("X75 MATCHED CLASSIFICATION EVALUATION")
print("75-NODE SAME-PROTOCOL MATCHED BENCHMARK")
print("FULL NON-LEAKAGE / STATISTICAL FILTERING / PROPOSED SIX")
print("=" * 120)


df_raw = pd.read_csv(
    RAW_PATH
)


print("\n[1] RAW DATA")
print("-" * 100)

print(
    "Raw shape:",
    df_raw.shape
)


df_raw['host_id_full'] = (
    df_raw['module']
    .astype(str)
    .str.extract(
        r'(AODVNetwork\.host\[\d+\])'
    )
)


df_raw = (
    df_raw
    .dropna(
        subset=['host_id_full']
    )
    .copy()
)


print(
    "Rows after host extraction:",
    len(df_raw)
)


df_raw['value'] = pd.to_numeric(
    df_raw['value'],
    errors='coerce'
)


# One row = one node in one simulation run

df_features = (
    df_raw
    .pivot_table(
        index=[
            'run',
            'host_id_full'
        ],
        columns='name',
        values='value'
    )
    .reset_index()
)


df_features = (
    df_features
    .fillna(0)
)


print("\n[2] NODE-LEVEL FEATURE MATRIX")
print("-" * 100)

print(
    "Feature matrix shape:",
    df_features.shape
)


def label_func(row):

    run_name = str(
        row['run']
    )

    if 'attacker' not in run_name.lower():
        return 0

    # Attack simulation:
    # only actual configured Blackhole node = class 1
    return int(
        any(
            attacker_id in row['host_id_full']
            for attacker_id in FIXED_ATTACKER_IDS
        )
    )


df_features['is_attacker'] = (
    df_features
    .apply(
        label_func,
        axis=1
    )
)


print("\nNode-class distribution:")

print(
    df_features[
        'is_attacker'
    ]
    .value_counts()
    .sort_index()
)


df_features['run_condition'] = np.where(
    df_features[
        'run'
    ]
    .astype(str)
    .str.contains(
        'Attacker',
        case=False,
        na=False
    ),
    'Attack',
    'Normal'
)


run_table = (
    df_features[
        [
            'run',
            'run_condition'
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


condition_count_per_run = (
    run_table
    .groupby('run')[
        'run_condition'
    ]
    .nunique()
)


assert (
    condition_count_per_run.max() == 1
), (
    "ERROR: a simulation run belongs "
    "to more than one condition."
)


print("\n[3] SIMULATION RUNS")
print("-" * 100)

print(
    "Total unique runs:",
    len(run_table)
)

print("\nRun-condition distribution:")

print(
    run_table[
        'run_condition'
    ]
    .value_counts()
    .sort_index()
)


# The unique simulation runs, rather than individual node records, are split.
# Attack/Normal stratification is used solely to balance the two partitions.
# X75 is excluded from the globally selected six-feature representation.
# No statistical filters, scalers or classifiers are fitted before splitting.
# A saved manifest is always reused, not regenerated or silently overwritten.

def sha_run_ids(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode('utf-8')).hexdigest()

if not os.path.isfile(BENCHMARK_SPLIT_PATH):
    # Canonical ordering makes first-run partition generation reproducible
    # independently of CSV row ordering or pandas groupby iteration order.
    source_runs = run_table[['run', 'run_condition']].copy()
    source_runs['run'] = source_runs['run'].astype(str)
    source_runs = source_runs.sort_values('run', kind='stable').reset_index(drop=True)
    if len(source_runs) != EXPECTED_TOTAL_RUNS:
        raise RuntimeError('Unexpected number of X75 runs before partitioning')
    if source_runs['run'].duplicated().any() or source_runs['run'].eq('').any():
        raise RuntimeError('Invalid X75 run identifiers before partitioning')
    if source_runs['run_condition'].value_counts().to_dict() != {
        'Normal': 500, 'Attack': 500
    }:
        raise RuntimeError('Expected 500 Normal and 500 Attack runs in X75 RAW')

    train_run_export, test_run_export = train_test_split(
        source_runs,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=source_runs['run_condition'],
    )
    train_run_export = train_run_export.copy()
    test_run_export = test_run_export.copy()
    train_run_export['partition'] = 'TRAIN'
    test_run_export['partition'] = 'TEST'
    new_manifest = pd.concat(
        [train_run_export, test_run_export], ignore_index=True
    )
    if (len(train_run_export), len(test_run_export)) != (
        EXPECTED_TRAIN_RUNS, EXPECTED_TEST_RUNS
    ):
        raise RuntimeError('Unexpected X75 70/30 run split sizes')
    os.makedirs(os.path.dirname(BENCHMARK_SPLIT_PATH), exist_ok=True)
    # Exclusive create prevents accidental replacement of an existing manifest.
    new_manifest.to_csv(
        BENCHMARK_SPLIT_PATH, index=False, encoding='utf-8', mode='x'
    )
    split_status = 'GENERATED_FROM_RAW'
    print('X75 RUN SPLIT GENERATED FROM RAW:', BENCHMARK_SPLIT_PATH)
else:
    split_status = 'REUSED_SAVED_MANIFEST'
    print('X75 EXISTING RUN SPLIT FOUND:', BENCHMARK_SPLIT_PATH)

split_export = pd.read_csv(
    BENCHMARK_SPLIT_PATH, dtype=str, keep_default_na=False
)
required_cols = {'run', 'run_condition', 'partition'}
if not required_cols.issubset(split_export.columns):
    raise RuntimeError(
        f'Missing required saved-manifest columns: {required_cols - set(split_export.columns)}'
    )
split_export = split_export[['run', 'run_condition', 'partition']].copy()
split_export['run'] = split_export['run'].str.strip()
split_export['partition'] = split_export['partition'].str.strip().str.upper()
split_export['run_condition'] = split_export['run_condition'].str.strip().str.lower()

if split_export['run'].eq('').any() or split_export['run'].duplicated().any():
    raise RuntimeError('Saved X75 manifest has blank or duplicate run IDs')
if not split_export['partition'].isin(['TRAIN', 'TEST']).all():
    raise RuntimeError('Unexpected partition label in the saved X75 manifest')
expected_condition = np.where(
    split_export['run'].str.contains('Attacker', case=False, na=False),
    'attack', 'normal'
)
if not np.array_equal(split_export['run_condition'].to_numpy(), expected_condition):
    raise RuntimeError('Manifest run_condition does not agree with the run names')

train_runs = set(split_export.loc[split_export['partition'].eq('TRAIN'), 'run'])
test_runs = set(split_export.loc[split_export['partition'].eq('TEST'), 'run'])
run_overlap = train_runs & test_runs
if (len(train_runs), len(test_runs), len(run_overlap)) != (
    EXPECTED_TRAIN_RUNS, EXPECTED_TEST_RUNS, 0
):
    raise RuntimeError(
        f'X75 saved split counts unexpected: '
        f'TRAIN={len(train_runs)}, TEST={len(test_runs)}, overlap={len(run_overlap)}'
    )
if (train_runs | test_runs) != set(run_table['run']):
    raise RuntimeError('X75 input dataset run IDs differ from saved manifest')

expected_condition_counts = {
    'TRAIN': {'normal': 350, 'attack': 350},
    'TEST': {'normal': 150, 'attack': 150},
}
for partition, expected in expected_condition_counts.items():
    counts = split_export.loc[
        split_export['partition'].eq(partition), 'run_condition'
    ].value_counts().to_dict()
    if counts != expected:
        raise RuntimeError(
            f'X75 {partition} run-condition counts differ: '
            f'expected={expected}, actual={counts}'
        )

print('\n' + '=' * 120)
print('X75 RUN SPLIT AUDIT')
print('=' * 120)
print('Run split manifest :', BENCHMARK_SPLIT_PATH)
print('Split status       :', split_status)
print('Training runs    :', len(train_runs))
print('Testing runs     :', len(test_runs))
print('Run overlap      :', len(run_overlap))
print('TRAIN70 SHA-256  :', sha_run_ids(train_runs))
print('HOLDOUT30 SHA-256:', sha_run_ids(test_runs))
print('X75 excluded from proposed-six feature selection: YES')
print('X75 HOLDOUT30 reserved for final evaluation: PASS')

train_run_table = split_export[split_export['partition'].eq('TRAIN')].copy()
test_run_table = split_export[split_export['partition'].eq('TEST')].copy()


train_split_export = (
    train_run_table
    .copy()
)

train_split_export[
    'partition'
] = 'TRAIN'


test_split_export = (
    test_run_table
    .copy()
)

test_split_export[
    'partition'
] = 'TEST'


split_export = pd.concat(
    [
        train_split_export,
        test_split_export
    ],
    ignore_index=True
)


split_export.to_csv(
    RUN_SPLIT_PATH,
    index=False
)


train_df = (
    df_features[
        df_features[
            'run'
        ]
        .isin(
            train_runs
        )
    ]
    .copy()
)


test_df = (
    df_features[
        df_features[
            'run'
        ]
        .isin(
            test_runs
        )
    ]
    .copy()
)


assert (
    set(
        train_df['run']
    )
    .isdisjoint(
        set(
            test_df['run']
        )
    )
)


print("\n[4] NODE-LEVEL PARTITIONS")
print("-" * 100)

print(
    "Training node rows:",
    len(train_df)
)

print(
    "Testing node rows :",
    len(test_df)
)


print("\nTraining labels:")

print(
    train_df[
        'is_attacker'
    ]
    .value_counts()
    .sort_index()
)


print("\nTesting labels:")

print(
    test_df[
        'is_attacker'
    ]
    .value_counts()
    .sort_index()
)


# 500 Normal + 500 Attack runs, 75 node profiles per run.
# A complete Attack run must contain both designated Blackhole hosts.
# No holdout labels are used for feature selection; these are dataset audits.

expected_conditions = {'normal': 500, 'attack': 500}
actual_conditions = (
    run_table['run_condition'].str.lower().value_counts().to_dict()
)
if actual_conditions != expected_conditions:
    raise RuntimeError(
        f'X75 run condition mismatch: expected {expected_conditions}; '
        f'got {actual_conditions}'
    )

rows_per_run = df_features.groupby('run').size()
if not rows_per_run.eq(75).all():
    raise RuntimeError('X75 expects exactly 75 node profiles per simulation run')

# Each attack run has two labeled Blackhole nodes (host[20] and host[40]);
# all nodes in normal runs have label zero.
attack_per_run = (
    df_features.groupby('run')['is_attacker'].sum().astype(int)
)
run_conditions = run_table.set_index('run')['run_condition'].str.lower()
expected_attack_per_run = run_conditions.map({'normal': 0, 'attack': 2})
if not attack_per_run.sort_index().equals(expected_attack_per_run.sort_index().astype(int)):
    raise RuntimeError('X75 attack label count is not 2 per attack run and 0 per normal run')

train_attack_count = int(train_df['is_attacker'].sum())
test_attack_count = int(test_df['is_attacker'].sum())
if (train_attack_count, test_attack_count) != (700, 300):
    raise RuntimeError(
        f'X75 expected TRAIN/HOLDOUT attacker rows (700, 300); '
        f'got ({train_attack_count}, {test_attack_count})'
    )

print('X75 two-Blackhole attacker audit: PASS')
print('Train/holdout attacker rows:', train_attack_count, test_attack_count)


METADATA_COLUMNS = {
    'run',
    'host_id_full',
    'is_attacker',
    'run_condition'
}


raw_predictive_features = [

    column

    for column in df_features.columns

    if (
        column not in METADATA_COLUMNS
        and
        pd.api.types.is_numeric_dtype(
            df_features[column]
        )
    )
]


print("\n[5] RAW PREDICTIVE FEATURE UNIVERSE")
print("-" * 100)

print(
    "Raw numeric predictive features:",
    len(raw_predictive_features)
)


# The matched evaluation requires 59 raw numeric features and retains
# 56 features after removing the three predefined leakage variables.
if len(raw_predictive_features) != 59:
    raise RuntimeError(
        f"RAW FEATURE SCHEMA MISMATCH: got {len(raw_predictive_features)}, "
        "expected 59. Verify that this is the harmonized OMNeT++ dataset "
        "used for this matched evaluation."
    )

if len(df_features) != EXPECTED_TOTAL_NODE_ROWS or len(run_table) != EXPECTED_TOTAL_RUNS:
    raise RuntimeError(
        f'Unexpected X75 dataset size: node profiles={len(df_features)}, '
        f'runs={len(run_table)}; expected {EXPECTED_TOTAL_NODE_ROWS} and {EXPECTED_TOTAL_RUNS}'
    )

# Remove ONLY the three predefined leakage variables.

full_non_leakage_features = [

    feature

    for feature in raw_predictive_features

    if feature not in LEAKAGE_FEATURES
]


removed_leakage = [

    feature

    for feature in LEAKAGE_FEATURES

    if feature in raw_predictive_features
]


print("\n" + "=" * 120)
print("REPRESENTATION 1 — FULL NON-LEAKAGE")
print("=" * 120)

print(
    "Raw numeric features      :",
    len(raw_predictive_features)
)

print(
    "Leakage removed           :",
    removed_leakage
)

print(
    "Full non-leakage features :",
    len(full_non_leakage_features)
)


if len(full_non_leakage_features) != 56 or len(removed_leakage) != 3:
    raise RuntimeError(
        f'Non-leakage schema mismatch: features={len(full_non_leakage_features)}, '
        f'removed-leakage={removed_leakage}'
    )

# Zero-variance and absolute Pearson-correlation (>=0.99) filters are fitted
# independently within the training portion of five TRAIN70 grouped folds.
# Target-correlation tie-breaking uses inner-training labels only.
# The final statistical representation is the union across these five fits.
# No inner-validation or HOLDOUT30 data fit either feature filter.
# MI, PFI and TreeSHAP are not computed for this statistical baseline.

def fit_statistical_filters(X_inner_train, y_inner_train):
    selector = VarianceThreshold(threshold=0.0)
    X_var_np = selector.fit_transform(X_inner_train)
    after_zv = X_inner_train.columns[selector.get_support()].tolist()
    if not after_zv:
        raise RuntimeError('Statistical filtering removed all features at the zero-variance stage')
    X_zv = pd.DataFrame(X_var_np, columns=after_zv, index=X_inner_train.index)
    corr_matrix = X_zv.corr().abs()
    target_corr = X_zv.apply(lambda col: col.corr(y_inner_train)).abs().fillna(0.0)
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    auto_drop = set()
    for col in upper_tri.columns:
        correlated_features = upper_tri.index[
            upper_tri[col] >= CORRELATION_THRESHOLD
        ].tolist()
        for feature in correlated_features:
            if target_corr[col] >= target_corr[feature]:
                auto_drop.add(feature)
            else:
                auto_drop.add(col)
    after_corr = [f for f in after_zv if f not in auto_drop]
    if not after_corr:
        raise RuntimeError('Statistical filtering removed all features at the correlation stage')
    return after_zv, after_corr

# Five stratified inner folds over unique TRAIN70 simulation runs.
run_series = train_df['run'].astype(str).reset_index(drop=True)
unique_inner_runs = pd.DataFrame({'run': run_series}).drop_duplicates().reset_index(drop=True)
unique_inner_runs['condition'] = unique_inner_runs['run'].str.contains(
    'Attacker', case=False, na=False
).astype(int)

inner_cv = StratifiedKFold(
    n_splits=N_INNER_FOLDS, shuffle=True, random_state=SPLIT_SEED
)
inner_sets = []
inner_zero_sets = []
inner_audit_rows = []

for inner_id, (run_fit_idx, run_validation_idx) in enumerate(
    inner_cv.split(
        np.zeros(len(unique_inner_runs)), unique_inner_runs['condition']
    ), start=1
):
    inner_fit_runs = set(unique_inner_runs.iloc[run_fit_idx]['run'])
    inner_validation_runs = set(unique_inner_runs.iloc[run_validation_idx]['run'])
    if inner_fit_runs & inner_validation_runs:
        raise RuntimeError('Inner-fold run overlap')
    # Validation runs are NOT used in filtering or tie-break.
    inner_train_mask = run_series.isin(inner_fit_runs).to_numpy()
    X_inner = train_df.iloc[np.flatnonzero(inner_train_mask)][full_non_leakage_features].copy()
    y_inner = train_df.iloc[np.flatnonzero(inner_train_mask)]['is_attacker'].copy()
    zv_features, corr_features = fit_statistical_filters(X_inner, y_inner)
    inner_sets.append(set(corr_features))
    inner_zero_sets.append(set(zv_features))
    inner_audit_rows.append({
        'Inner_Fold': inner_id,
        'Inner_Training_Runs': len(inner_fit_runs),
        'Inner_Validation_Runs': len(inner_validation_runs),
        'Run_Overlap': len(inner_fit_runs & inner_validation_runs),
        'After_Zero_Variance': len(zv_features),
        'After_Correlation': len(corr_features),
        'Selected_Features': '|'.join(corr_features),
    })
    print(
        f'Statistical inner fold {inner_id}: TRAIN runs={len(inner_fit_runs)} | '
        f'VALIDATION runs={len(inner_validation_runs)} | '
        f'after zero-variance={len(zv_features)} | '
        f'after Pearson={len(corr_features)}'
    )

if len(inner_sets) != 5:
    raise RuntimeError('Missing statistical-filtering inner folds')
statistical_features = sorted(set.union(*inner_sets))
zero_variance_union = sorted(set.union(*inner_zero_sets))
if not statistical_features or not set(statistical_features).issubset(full_non_leakage_features):
    raise RuntimeError('Invalid statistical feature set')
# X75 statistical features are learned from its own TRAIN70 runs;
# the union size is not assumed to equal another scenario's size.

print('\nX75 STATISTICAL FILTERING - PASS')
print('Full non-leakage features:', len(full_non_leakage_features))
print('Five inner-fold union   :', len(statistical_features))
print('Selected statistical features:')
for i, feature in enumerate(statistical_features, 1):
    print(f'  {i:02d}. {feature}')

pd.DataFrame(inner_audit_rows).to_csv(
    os.path.join(SAVE_DIR, 'X75_statistical_inner_folds_audit.csv'), index=False
)
pd.DataFrame({'Statistical_Feature': statistical_features}).to_csv(
    os.path.join(SAVE_DIR, 'X75_statistical_selected_features.csv'), index=False
)


missing_proposed = [

    feature

    for feature in PROPOSED_FEATURES

    if feature not in df_features.columns
]


if missing_proposed:

    raise ValueError(
        f"Missing proposed features: "
        f"{missing_proposed}"
    )


assert (
    set(
        PROPOSED_FEATURES
    )
    .isdisjoint(
        set(
            LEAKAGE_FEATURES
        )
    )
)


print("\n" + "=" * 120)
print("REPRESENTATION 3 — PROPOSED SIX FEATURES")
print("=" * 120)

print(
    "Number of proposed features:",
    len(PROPOSED_FEATURES)
)


for i, feature in enumerate(
    PROPOSED_FEATURES,
    1
):

    print(
        f"{i}. {feature}"
    )


REPRESENTATIONS = {

    'Full non-leakage':
        full_non_leakage_features,

    'Statistical filtering':
        statistical_features,

    'Proposed 6 features':
        PROPOSED_FEATURES
}


with open(
    FEATURE_AUDIT_PATH,
    'w',
    encoding='utf-8'
) as f:

    f.write(
        "X75 MATCHED CLASSIFICATION EVALUATION\n"
    )

    f.write(
        "75-NODE MATCHED BENCHMARK\n"
    )

    f.write(
        "=" * 100 + "\n\n"
    )

    f.write(
        f"Raw features: "
        f"{len(raw_predictive_features)}\n"
    )

    f.write(
        f"Leakage removed: "
        f"{removed_leakage}\n\n"
    )

    for representation_name, features in REPRESENTATIONS.items():

        f.write(
            f"{representation_name}\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            f"Feature count: "
            f"{len(features)}\n"
        )

        for feature in features:

            f.write(
                f"{feature}\n"
            )

        f.write("\n")


def compute_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )


    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )


    macro_precision = precision_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, average='macro', zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average='macro',
        zero_division=0
    )


    return {

        'Accuracy':
            accuracy,

        'Precision':
            precision,

        'Recall':
            recall,

        'F1':
            f1,

        'Macro_Precision': macro_precision,
        'Macro_Recall': macro_recall,
        'Macro_F1':
            macro_f1,

        'Specificity':
            specificity,

        'FPR':
            fpr,

        'TN':
            tn,

        'FP':
            fp,

        'FN':
            fn,

        'TP':
            tp
    }


def build_models():

    return {

        'RF':
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ),

        'DT':
            DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced'
            ),

        'SVM':
            SVC(
                kernel='rbf',
                class_weight='balanced'
            ),

        'KNN':
            KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),

        'MLP':
            MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            )
    }


results = []


for representation_name, feature_list in REPRESENTATIONS.items():

    print("\n\n" + "#" * 120)

    print(
        "REPRESENTATION:",
        representation_name
    )

    print(
        "FEATURE COUNT:",
        len(feature_list)
    )

    print("#" * 120)


    # Reuse the same node-level train/test partitions for all representations

    X_train = (
        train_df[
            feature_list
        ]
        .copy()
    )


    X_test = (
        test_df[
            feature_list
        ]
        .copy()
    )


    y_train = (
        train_df[
            'is_attacker'
        ]
        .copy()
    )


    y_test = (
        test_df[
            'is_attacker'
        ]
        .copy()
    )


    # Exclude labels and run/node metadata from classifier predictors

    forbidden = {
        'run',
        'host_id_full',
        'run_condition',
        'is_attacker'
    }


    assert (
        len(
            forbidden
            .intersection(
                X_train.columns
            )
        )
        == 0
    )


    # Training-only scaling

    scaler = StandardScaler()


    X_train_scaled = (
        scaler
        .fit_transform(
            X_train
        )
    )


    X_test_scaled = (
        scaler
        .transform(
            X_test
        )
    )


    models = build_models()


    for classifier_name, model in models.items():

        # Scale SVM, KNN and MLP using training-fitted standardization

        if classifier_name in {
            'SVM',
            'KNN',
            'MLP'
        }:

            model.fit(
                X_train_scaled,
                y_train
            )

            y_pred = model.predict(
                X_test_scaled
            )

        else:

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_test
            )


        metrics = compute_metrics(
            y_test,
            y_pred
        )


        results.append(
            {

                'Representation':
                    representation_name,

                'Feature_Count':
                    len(feature_list),

                'Classifier':
                    classifier_name,

                **metrics
            }
        )


results_df = pd.DataFrame(
    results
)


representation_order = [
    'Full non-leakage',
    'Statistical filtering',
    'Proposed 6 features'
]


classifier_order = [
    'RF',
    'DT',
    'SVM',
    'KNN',
    'MLP'
]


results_df[
    'Representation'
] = pd.Categorical(
    results_df[
        'Representation'
    ],
    categories=representation_order,
    ordered=True
)


results_df[
    'Classifier'
] = pd.Categorical(
    results_df[
        'Classifier'
    ],
    categories=classifier_order,
    ordered=True
)


results_df = (
    results_df
    .sort_values(
        [
            'Representation',
            'Classifier'
        ]
    )
    .reset_index(drop=True)
)


print("\n\n" + "=" * 160)
print("DETAILED MATCHED HOLDOUT RESULTS - 75 NODES")
print("=" * 160)


detailed_columns = [
    'Representation',
    'Feature_Count',
    'Classifier',
    'Accuracy',
    'Precision',
    'Recall',
    'F1',
    'Macro_Precision',
    'Macro_Recall',
    'Macro_F1',
    'Specificity',
    'FPR',
    'TN',
    'FP',
    'FN',
    'TP'
]


print(
    results_df[
        detailed_columns
    ]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# 20A. MACRO-AVERAGED METRICS ON THE FINAL 30% HOLDOUT
# Export macro F1, macro precision and macro recall summaries. They differ
# from the positive-class F1 values in the separate attack-class summary.

macro_f1_table = results_df.pivot(
    index='Representation', columns='Classifier', values='Macro_F1'
).reindex(index=representation_order, columns=classifier_order)
macro_f1_table.insert(0, 'Features', [
    len(REPRESENTATIONS[name]) for name in representation_order
])
macro_f1_table.to_csv(
    os.path.join(SAVE_DIR, 'X75_macro_f1_holdout30.csv')
)
print('\n' + '=' * 120)
print('X75 - MACRO F1 SCORES ON VERIFIED HOLDOUT30')
print('=' * 120)
print(macro_f1_table.to_string(float_format=lambda v: f'{v:.6f}'))

for metric in ('Macro_Precision', 'Macro_Recall'):
    tbl = results_df.pivot(
        index='Representation', columns='Classifier', values=metric
    ).reindex(index=representation_order, columns=classifier_order)
    tbl.insert(0, 'Features', [
        len(REPRESENTATIONS[name]) for name in representation_order
    ])
    tbl.to_csv(os.path.join(SAVE_DIR, f'X75_{metric.upper()}_HOLDOUT30.csv'))

# One row per feature representation and one column per classifier.
# This is positive-class (Blackhole) F1, not macro F1.

attack_f1_table = (
    results_df[
        [
            'Representation',
            'Classifier',
            'F1'
        ]
    ]
    .pivot(
        index='Representation',
        columns='Classifier',
        values='F1'
    )
    .reindex(
        index=representation_order,
        columns=classifier_order
    )
)


feature_counts = {

    'Full non-leakage':
        len(full_non_leakage_features),

    'Statistical filtering':
        len(statistical_features),

    'Proposed 6 features':
        len(PROPOSED_FEATURES)
}


attack_f1_table.insert(
    0,
    'Features',
    [
        feature_counts[
            representation
        ]
        for representation in representation_order
    ]
)


print("\n\n" + "=" * 120)
print("ATTACK-CLASS F1 SUMMARY")
print("SAME-PROTOCOL MATCHED BENCHMARK — 75 NODES")
print("CELL VALUES = ATTACK-CLASS F1-SCORE")
print("=" * 120)


print(
    attack_f1_table
    .to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print("=" * 120)


attack_f1_percent = (
    attack_f1_table
    .copy()
)


for classifier in classifier_order:

    attack_f1_percent[
        classifier
    ] = (
        attack_f1_percent[
            classifier
        ]
        * 100
    )


print("\n" + "=" * 120)
print("ATTACK-CLASS F1 SUMMARY (%)")
print("=" * 120)


print(
    attack_f1_percent
    .to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# Compare all three feature representations across five classifiers.

METRICS_TO_PRINT = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1',
    'Macro_F1',
    'Specificity',
    'FPR'
]


for metric in METRICS_TO_PRINT:

    metric_table = (
        results_df[
            [
                'Representation',
                'Classifier',
                metric
            ]
        ]
        .pivot(
            index='Representation',
            columns='Classifier',
            values=metric
        )
        .reindex(
            index=representation_order,
            columns=classifier_order
        )
    )


    print("\n")
    print("-" * 120)

    print(
        f"{metric.upper()} TABLE"
    )

    print("-" * 120)


    print(
        metric_table
        .to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


results_df.to_csv(
    DETAILED_RESULTS_PATH,
    index=False
)


attack_f1_table.to_csv(
    MAIN_TABLE_PATH
)


percent_path = os.path.join(
    SAVE_DIR,
    'X75_attack_f1_percent_by_representation.csv'
)


attack_f1_percent.to_csv(
    percent_path
)


print("\n\n" + "=" * 120)
print("FINAL X75 MATCHED-EVALUATION PROTOCOL AUDIT")
print("=" * 120)


print(
    "Scenario                              : 75 nodes"
)

print(
    f"Raw predictive features               : "
    f"{len(raw_predictive_features)}"
)

print(
    f"Full non-leakage features             : "
    f"{len(full_non_leakage_features)}"
)

print(
    f"Statistically filtered features       : "
    f"{len(statistical_features)}"
)

print(
    f"Proposed features                     : "
    f"{len(PROPOSED_FEATURES)}"
)

print(
    f"Statistical inner-CV random_state     : "
    f"{SPLIT_SEED}"
)

print(
    "Train/Test ratio                      : 70/30"
)

print(
    "Statistical splitting unit            : SIMULATION RUN"
)

print(
    f"Run split source/status               : {split_status}"
)

print(
    "TRAIN/HOLDOUT run IDs verified       : YES"
)

print(
    f"Saved split path                       : {BENCHMARK_SPLIT_PATH}"
)

print(
    f"Train/Test run overlap                : "
    f"{len(run_overlap)}"
)

print(
    "Node-level random split               : NO"
)

print(
    "Run ID used as predictor              : NO"
)

print(
    "Node ID used as predictor             : NO"
)

print(
    "Run condition used as predictor       : NO"
)

print(
    "Leakage variables used                : NO"
)

print(
    "Zero variance fitted on training only : YES"
)

print(
    "Pearson fitted on training only       : YES"
)

print(
    "Target tie-break uses training only   : YES"
)

print(
    "Test used in feature selection        : NO"
)

print(
    "Scaler fitted on training only        : YES"
)

print(
    "Fixed five-classifier configuration    : YES"
)

print(
    "Common metrics across representations : YES"
)

print(
    "MI/PFI/TreeSHAP recomputed here       : NO"
)

print(
    "Statistical filtering                : 5 inner fits + union"
)


assert (
    len(results_df) == 15
)


assert (
    results_df
    .groupby(
        'Representation',
        observed=True
    )[
        'Classifier'
    ]
    .nunique()
    .eq(5)
    .all()
)


assert (
    attack_f1_table.shape
    ==
    (3, 6)
)


print("\n" + "=" * 120)
print("ATTACK-CLASS F1 TABLE STRUCTURE: PASS")
print("15 MATCHED EXPERIMENTS: PASS")
print("MATCHED BENCHMARK INTEGRITY: PASS")
print("=" * 120)


# List every generated output, including statistical filtering and macro metrics.
saved_output_paths = [
    BENCHMARK_SPLIT_PATH,
    DETAILED_RESULTS_PATH,
    MAIN_TABLE_PATH,
    percent_path,
    RUN_SPLIT_PATH,
    FEATURE_AUDIT_PATH,
    os.path.join(SAVE_DIR, 'X75_statistical_inner_folds_audit.csv'),
    os.path.join(SAVE_DIR, 'X75_statistical_selected_features.csv'),
    os.path.join(SAVE_DIR, 'X75_macro_f1_holdout30.csv'),
    os.path.join(SAVE_DIR, 'X75_MACRO_PRECISION_HOLDOUT30.csv'),
    os.path.join(SAVE_DIR, 'X75_MACRO_RECALL_HOLDOUT30.csv'),
]
print("\nFILES SAVED:")
for index, saved_path in enumerate(saved_output_paths, 1):
    print(f"{index}. {saved_path}")