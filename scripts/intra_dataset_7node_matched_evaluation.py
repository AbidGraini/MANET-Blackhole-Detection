import os
from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
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


# X7 MATCHED CLASSIFICATION EVALUATION
# All models use the same saved simulation-run-grouped TRAIN70/HOLDOUT30 IDs.
# Statistical filtering is fitted on inner-training folds of TRAIN70 only;
# scaling for SVM, KNN and MLP is fitted on TRAIN70 only.


# 1. CONFIGURATION

# Paths default to folders in the repository; environment variables allow local overrides.
PROJECT_ROOT = Path(os.environ.get("MANET_PROJECT_ROOT", str(Path.cwd()))).expanduser()
DATA_DIR = Path(os.environ.get("MANET_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
SPLIT_ROOT = Path(os.environ.get("MANET_SPLITS_DIR", str(PROJECT_ROOT / "splits"))).expanduser()
SELECTION_OUTPUT_DIR = Path(os.environ.get("MANET_SELECTION_OUTPUT", str(PROJECT_ROOT / "selection_outputs"))).expanduser()
RAW_PATH = str(DATA_DIR / "7NODE1BLACK300RUN600S.csv")

# SPLIT_SEED is used only by the stratified statistical-filtering inner folds.
SPLIT_SEED = 42

FIXED_ATTACKER_IDS = [
    'host[5]'
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


SAVE_DIR = str(Path(os.environ.get(
    "MANET_X7_OUTPUT", str(PROJECT_ROOT / "results" / "X7_MATCHED_EVALUATION")
)).expanduser())

BENCHMARK_SPLIT_PATH = str(SPLIT_ROOT / "7NODE" / "7node_run_split.csv")
FROZEN_ID_ROOT = str(SELECTION_OUTPUT_DIR / "00_FIXED_70_30_RUN_IDS")
FROZEN_TRAIN_PATH = os.path.join(FROZEN_ID_ROOT, 'X7_DEVELOPMENT_TRAIN_70_RUN_IDS.csv')
FROZEN_TEST_PATH = os.path.join(FROZEN_ID_ROOT, 'X7_FINAL_CLASSIFICATION_HOLDOUT_30_RUN_IDS.csv')
EXPECTED_TRAIN_HASH = '8b51ab9ea8ba69877323b850835605e1e26b680de107955995a68dad84baa7fc'
EXPECTED_TEST_HASH = '153032e574b00fb3fdf8c28c507e92fba6e12b262ee1e5cc857f1ee38b0724e9'
N_INNER_FOLDS = 5
EXPECTED_X7_STATISTICAL_FEATURE_COUNT = 22

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


DETAILED_RESULTS_PATH = os.path.join(
    SAVE_DIR,
    'X7_detailed_classification_metrics.csv'
)

MAIN_TABLE_PATH = os.path.join(
    SAVE_DIR,
    'X7_attack_f1_by_representation.csv'
)

RUN_SPLIT_PATH = os.path.join(
    SAVE_DIR,
    'X7_verified_run_split.csv'
)

FEATURE_AUDIT_PATH = os.path.join(
    SAVE_DIR,
    'X7_feature_representation_audit.txt'
)


print("=" * 120)
print("X7 MATCHED CLASSIFICATION EVALUATION")
print("7-NODE SAME-PROTOCOL MATCHED BENCHMARK")
print("FULL NON-LEAKAGE / STATISTICAL FILTERING / PROPOSED SIX")
print("=" * 120)


# 2. LOAD RAW DATA

df_raw = pd.read_csv(
    RAW_PATH
)


print("\n[1] RAW DATA")
print("-" * 100)

print(
    "Raw shape:",
    df_raw.shape
)


# 3. EXTRACT NODE IDENTIFIER

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


# 4. NUMERIC VALUE

df_raw['value'] = pd.to_numeric(
    df_raw['value'],
    errors='coerce'
)


# 5. NODE-LEVEL MATRIX
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


# 6. LABEL

def label_func(row):

    run_name = str(
        row['run']
    )

    if 'attacker' not in run_name.lower():
        return 0

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


# 7. RUN CONDITION

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


# 8. LOAD AND VALIDATE THE SAVED RAW-DERIVED 70/30 RUN SPLIT

def sha_run_ids(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()

for required_path in (BENCHMARK_SPLIT_PATH, FROZEN_TRAIN_PATH, FROZEN_TEST_PATH):
    if not os.path.isfile(required_path):
        raise FileNotFoundError(
            f"Missing required saved run-ID file: {required_path}\n"
            "Run the RAW-based feature-selection split export before classification."
        )

split_export = pd.read_csv(
    BENCHMARK_SPLIT_PATH, dtype=str, keep_default_na=False
)
required_cols = {'run', 'run_condition', 'partition'}
if not required_cols.issubset(split_export.columns):
    raise RuntimeError(f"Invalid saved run-split manifest: {required_cols - set(split_export.columns)}")

split_export = split_export[['run','run_condition','partition']].copy()
split_export['partition'] = split_export['partition'].str.strip().str.upper()
split_export['run_condition'] = split_export['run_condition'].str.strip().str.lower()
if split_export['run'].eq('').any() or split_export['run'].duplicated().any():
    raise RuntimeError('Empty or duplicated run IDs in saved manifest')
if not split_export['partition'].isin(['TRAIN','TEST']).all():
    raise RuntimeError('Unexpected partition label in saved manifest')
expected_condition = np.where(
    split_export['run'].str.contains('Attacker', case=False, na=False),
    'attack', 'normal'
)
if not np.array_equal(split_export['run_condition'].to_numpy(), expected_condition):
    raise RuntimeError('Run condition in manifest does not match run IDs')

train_runs = set(split_export.loc[split_export['partition'].eq('TRAIN'), 'run'])
test_runs = set(split_export.loc[split_export['partition'].eq('TEST'), 'run'])
run_overlap = train_runs & test_runs

if (len(train_runs), len(test_runs), len(run_overlap)) != (420, 180, 0):
    raise RuntimeError(
        f"Expected exactly 420 TRAIN, 180 TEST, 0 overlap; "
        f"got {(len(train_runs), len(test_runs), len(run_overlap))}"
    )

for role, frozen_path, expected in (
    ('TRAIN70', FROZEN_TRAIN_PATH, train_runs),
    ('HOLDOUT30', FROZEN_TEST_PATH, test_runs),
):
    frozen = pd.read_csv(frozen_path, dtype=str, keep_default_na=False)
    if 'run' not in frozen.columns or frozen['run'].duplicated().any():
        raise RuntimeError(f'Invalid {role} run-ID CSV: {frozen_path}')
    if set(frozen['run']) != expected:
        raise RuntimeError(f'{role} differs from frozen feature-selection run IDs')

if sha_run_ids(train_runs) != EXPECTED_TRAIN_HASH:
    raise RuntimeError('TRAIN70 SHA-256 differs from the saved reference partition')
if sha_run_ids(test_runs) != EXPECTED_TEST_HASH:
    raise RuntimeError('HOLDOUT30 SHA-256 differs from the saved reference partition')

current_runs = set(run_table['run'])
if current_runs != (train_runs | test_runs):
    raise RuntimeError('Dataset run IDs differ from the saved split manifest')

print("\n" + "=" * 120)
print('FROZEN RUN SPLIT AUDIT - 7 NODES')
print("=" * 120)
print('Saved RAW-derived manifest:', BENCHMARK_SPLIT_PATH)
print('Training runs    :', len(train_runs))
print('Testing runs     :', len(test_runs))
print('Run overlap      :', len(run_overlap))
print('TRAIN70 SHA-256  :', sha_run_ids(train_runs))
print('HOLDOUT30 SHA-256:', sha_run_ids(test_runs))
print('Same 30% excluded from feature selection: PASS')

train_run_table = split_export[split_export['partition'].eq('TRAIN')].copy()
test_run_table = split_export[split_export['partition'].eq('TEST')].copy()

# 9. EXPORT A COPY OF THE VERIFIED 70/30 RUN SPLIT

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


# 10. NODE-LEVEL TRAIN / TEST

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


# 11. RAW PREDICTIVE FEATURES

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


if len(raw_predictive_features) != 59:
    raise RuntimeError(
        f"RAW FEATURE SCHEMA MISMATCH: got {len(raw_predictive_features)}, "
        "expected 59. Verify that this is the harmonized OMNeT++ dataset "
        "used for this matched evaluation."
    )

if len(df_features) != 4200 or len(run_table) != 600:
    raise RuntimeError(
        f'Unexpected X7 dataset size: node profiles={len(df_features)}, '
        f'runs={len(run_table)}; expected 4200 and 600'
    )

# 12. REPRESENTATION 1 - FULL NON-LEAKAGE

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
print("REPRESENTATION 1 - FULL NON-LEAKAGE")
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

# 13. REPRESENTATION 2 - STATISTICAL FILTERING
# Fit the zero-variance and Pearson (absolute correlation >= 0.99) filters
# independently in the training portion of each of five run-grouped folds
# within TRAIN70. Correlation tie-breaking uses those training labels only.
# The statistical representation is the union of the five retained sets.
# Neither HOLDOUT30 nor inner-validation data are used to fit the filters.

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
if len(statistical_features) != EXPECTED_X7_STATISTICAL_FEATURE_COUNT:
    raise RuntimeError(
        f'X7 statistical feature-count audit failed: expected '
        f'{EXPECTED_X7_STATISTICAL_FEATURE_COUNT}, got {len(statistical_features)}. '
        'Check the dataset schema and saved run-ID manifests.'
    )

print('\nX7 STATISTICAL FILTERING - PASS')
print('Full non-leakage features:', len(full_non_leakage_features))
print('Five inner-fold union   :', len(statistical_features))
print('Selected statistical features:')
for i, feature in enumerate(statistical_features, 1):
    print(f'  {i:02d}. {feature}')

pd.DataFrame(inner_audit_rows).to_csv(
    os.path.join(SAVE_DIR, 'X7_statistical_inner_folds_audit.csv'), index=False
)
pd.DataFrame({'Statistical_Feature': statistical_features}).to_csv(
    os.path.join(SAVE_DIR, 'X7_statistical_selected_features.csv'), index=False
)

# 14. REPRESENTATION 3 - PROPOSED SIX

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
print("REPRESENTATION 3 - PROPOSED SIX FEATURES")
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


# 15. REPRESENTATION DICTIONARY

REPRESENTATIONS = {

    'Full non-leakage':
        full_non_leakage_features,

    'Statistical filtering':
        statistical_features,

    'Proposed 6 features':
        PROPOSED_FEATURES
}


# 16. FEATURE AUDIT FILE

with open(
    FEATURE_AUDIT_PATH,
    'w',
    encoding='utf-8'
) as f:

    f.write(
        "X7 MATCHED CLASSIFICATION EVALUATION\n"
    )

    f.write(
        "7-NODE MATCHED BENCHMARK\n"
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


# 17. METRICS

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


# 18. CLASSIFIERS

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


# 19. EVALUATE THREE REPRESENTATIONS WITH FIVE CLASSIFIERS

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


# 20. DETAILED RESULT TABLE

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
print("DETAILED MATCHED HOLDOUT RESULTS - 7 NODES")
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

macro_f1_table = results_df.pivot(
    index='Representation', columns='Classifier', values='Macro_F1'
).reindex(index=representation_order, columns=classifier_order)
macro_f1_table.insert(0, 'Features', [
    len(REPRESENTATIONS[name]) for name in representation_order
])
macro_f1_table.to_csv(
    os.path.join(SAVE_DIR, 'X7_macro_f1_holdout30.csv')
)
print('\n' + '=' * 120)
print('X7 - MACRO F1 SCORES ON VERIFIED HOLDOUT30')
print('=' * 120)
print(macro_f1_table.to_string(float_format=lambda v: f'{v:.6f}'))

for metric in ('Macro_Precision', 'Macro_Recall'):
    tbl = results_df.pivot(
        index='Representation', columns='Classifier', values=metric
    ).reindex(index=representation_order, columns=classifier_order)
    tbl.insert(0, 'Features', [
        len(REPRESENTATIONS[name]) for name in representation_order
    ])
    tbl.to_csv(os.path.join(SAVE_DIR, f'X7_{metric.upper()}_HOLDOUT30.csv'))

# 21. ATTACK-CLASS F1 SUMMARY
# These are positive-class (Blackhole) F1 scores, not macro F1 scores.

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
print("SAME-PROTOCOL MATCHED BENCHMARK - 7 NODES")
print("CELL VALUES = ATTACK-CLASS F1-SCORE")
print("=" * 120)


print(
    attack_f1_table
    .to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print("=" * 120)


# 22. ATTACK-CLASS F1 SCORES IN PERCENT

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


# 23. ADDITIONAL METRIC TABLES

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


# 24. SAVE TABLES

results_df.to_csv(
    DETAILED_RESULTS_PATH,
    index=False
)


attack_f1_table.to_csv(
    MAIN_TABLE_PATH
)


percent_path = os.path.join(
    SAVE_DIR,
    'X7_attack_f1_percent_by_representation.csv'
)


attack_f1_percent.to_csv(
    percent_path
)


# 25. FINAL PROTOCOL AUDIT

print("\n\n" + "=" * 120)
print("FINAL X7 MATCHED-EVALUATION PROTOCOL AUDIT")
print("=" * 120)


print(
    "Scenario                              : 7 nodes"
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
    "Saved RAW-derived split reused         : YES"
)

print(
    "Same train run IDs                    : YES"
)

print(
    "Same test run IDs                     : YES"
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


# 26. INTEGRITY CHECK

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


print("\nFILES SAVED:")

print(
    "1.",
    DETAILED_RESULTS_PATH
)

print(
    "2.",
    MAIN_TABLE_PATH
)

print(
    "3.",
    percent_path
)

print(
    "4.",
    RUN_SPLIT_PATH
)

print(
    "5.",
    FEATURE_AUDIT_PATH
)
