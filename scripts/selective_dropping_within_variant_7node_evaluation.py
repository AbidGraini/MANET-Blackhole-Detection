# SELECTIVE PACKET DROPPING — WITHIN-VARIANT EVALUATION (SECTION 4.4.2)
# Three independent intra-scenario experiments (Drop_P100%, Drop_P50%, Drop_P25%)
# Six fixed features, node-level pivot, and grouped 70/30 splitting;
# stratified by run condition, random_state=42, with five classifiers.

import os
from pathlib import Path
import gc
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Repository-relative defaults; environment variables allow local overrides.
PROJECT_ROOT = Path(os.environ.get('MANET_PROJECT_ROOT', str(Path.cwd()))).expanduser()
DATA_DIR = Path(os.environ.get('MANET_DATA_DIR', str(PROJECT_ROOT / 'data'))).expanduser()
OUTPUT_ROOT = Path(os.environ.get(
    'MANET_SELECTIVE_DROP_OUTPUT',
    str(PROJECT_ROOT / 'results' / '7NODE_SELECTIVE_DROPPING_WITHIN_VARIANT')
)).expanduser()
RAW_FILES = {
    'Drop_P100%': DATA_DIR / '7NODE1BLACK300RUN600S_100_PROB.csv',
    'Drop_P50%': DATA_DIR / '7NODE1BLACK300RUN600S_50_PROB.csv',
    'Drop_P25%': DATA_DIR / '7NODE1BLACK300RUN600S25PROB.csv',
}
SELECT_VARIANTS = ['Drop_P100%', 'Drop_P50%', 'Drop_P25%']
EXPECTED_REPETITIONS = 300
EXPECTED_NODES = 7
TEST_SIZE = 0.30
SPLIT_SEED = 42

# Fixed feature names and order used for all three variants.
SELECTED_FEATURES = [
    'numRREQsSent:count',
    'numRREPsSent:count',
    'packetSent:count',
    'numPacketsForwarded:count',
    'arpReplySent:count',
    'linkBroken:count'
]

# Accepted simulation configuration names for class labeling.
# A positive label requires an attack run and the attacker node ID.
NORMAL_CONFIGS = {'DatasetNormal', 'BH7_Normal'}
ATTACK_CONFIGS = {
    'Drop_P100%': {'DatasetAttacker', 'BH7_Drop100'},
    'Drop_P50%': {'BH7_Drop50'},
    'Drop_P25%': {'BH7_Drop25'},
}
ATTACKER_NODE = 5

def parse_run(run):
    match = re.match(r'^([A-Za-z][A-Za-z0-9_]*)-#?(\d+)(?:-|$)', str(run))
    if not match:
        raise ValueError(f'Unrecognized run identifier: {run}')
    return match.group(1), int(match.group(2))

def evaluate_one(variant):
    input_path = RAW_FILES[variant]
    if not input_path.is_file():
        raise FileNotFoundError(f'{variant} CSV not found: {input_path}')
    out = OUTPUT_ROOT / variant
    out.mkdir(parents=True, exist_ok=True)

    print('\n' + '='*104)
    print(f'{variant} | SIMULATION-RUN-GROUPED 70/30 EVALUATION')
    print('='*104)
    print('Input :', input_path)
    print('Size MB:', round(input_path.stat().st_size / (1024*1024), 2))

    # Read raw scalar records and extract node identifiers.
    df_raw = pd.read_csv(input_path, low_memory=False)
    print('\n[1] RAW:', df_raw.shape)
    must_have = {'run', 'module', 'name', 'value'}
    if not must_have.issubset(df_raw.columns):
        raise ValueError('Missing required CSV columns: ' + repr(must_have - set(df_raw.columns)))

    df_raw['host_id_full'] = df_raw['module'].str.extract(
        r'(AODVNetwork\.host\[\d+\])', expand=False)
    df_raw = df_raw.dropna(subset=['host_id_full']).copy()
    df_raw['value'] = pd.to_numeric(df_raw['value'], errors='coerce')

    # Pivot node records, averaging duplicate scalar keys with the
    # default mean aggregation, then replacing missing values with zero.
    df_features = (
        df_raw.pivot_table(
            index=['run', 'host_id_full'],
            columns='name',
            values='value'
        ).reset_index()
    )
    df_features.columns.name = None
    df_features = df_features.fillna(0)
    del df_raw
    gc.collect()

    # The pivot orders node profiles by run and host identifier.
    # Preserve this order when constructing the run-level partition.
    missing = [c for c in SELECTED_FEATURES if c not in df_features.columns]
    if missing:
        raise ValueError('Missing six-feature columns: ' + repr(missing))

    # Determine the run condition from the configured scenario name;
    # Drop_P50% and Drop_P25% run names do not contain "Attacker".
    run_table = df_features[['run']].drop_duplicates().reset_index(drop=True)
    parsed = run_table['run'].map(parse_run)
    run_table['config'] = parsed.map(lambda pair: pair[0])
    run_table['repetition'] = parsed.map(lambda pair: pair[1])

    allowed = NORMAL_CONFIGS | ATTACK_CONFIGS[variant]
    unknown = sorted(set(run_table['config']) - allowed)
    if unknown:
        raise ValueError(f'{variant}: unknown config(s) {unknown}; refusing automatic Normal label')

    run_table['run_condition'] = np.where(
        run_table['config'].isin(ATTACK_CONFIGS[variant]), 'Attack', 'Normal')

    if len(run_table) != 2 * EXPECTED_REPETITIONS:
        raise ValueError(f'Expected {2*EXPECTED_REPETITIONS} runs, got {len(run_table)}')
    if run_table.duplicated(['run_condition', 'repetition']).any():
        raise ValueError('Duplicate repetition within a run condition')
    for cond in ('Normal', 'Attack'):
        ids = set(run_table.loc[run_table['run_condition'].eq(cond), 'repetition'])
        if ids != set(range(EXPECTED_REPETITIONS)):
            raise ValueError(f'{cond} run repetitions incomplete; got {len(ids)} of {EXPECTED_REPETITIONS}')

    # Preserve node-profile order while appending run metadata.
    df_features = df_features.merge(
        run_table[['run', 'config', 'repetition', 'run_condition']],
        how='left', on='run', sort=False, validate='many_to_one')

    # Derive the run order used for splitting from the pivot output,
    # where run names are sorted lexicographically, not from raw CSV order.
    baseline_run_order = df_features[['run', 'run_condition']].drop_duplicates().reset_index(drop=True)
    if not baseline_run_order['run'].equals(run_table['run']):
        raise RuntimeError('Run order was modified; this changes the run-grouped partition')

    df_features['node_id'] = pd.to_numeric(
        df_features['host_id_full'].str.extract(r'host\[(\d+)\]', expand=False),
        errors='raise').astype(int)

    node_sets = df_features.groupby('run')['node_id'].agg(['count', lambda s: set(s)])
    bad_nodes = node_sets[
        (node_sets['count'] != EXPECTED_NODES)
        | (node_sets['<lambda_0>'] != set(range(EXPECTED_NODES)))
    ]
    if not bad_nodes.empty:
        raise ValueError('Incomplete 7-node runs: ' + repr(bad_nodes.head().to_dict()))

    df_features['is_attacker'] = (
        df_features['run_condition'].eq('Attack')
        & df_features['node_id'].eq(ATTACKER_NODE)
    ).astype(int)
    expected_normal = EXPECTED_REPETITIONS * (EXPECTED_NODES + EXPECTED_NODES - 1)
    got_classes = df_features['is_attacker'].value_counts().to_dict()
    if got_classes.get(0, 0) != expected_normal or got_classes.get(1, 0) != EXPECTED_REPETITIONS:
        raise ValueError(f'Incorrect node labeling: {got_classes}')

    print('\n[2] DATA AUDIT: node profiles', len(df_features), '| labels', got_classes)
    print(pd.crosstab(df_features['run_condition'], df_features['is_attacker']).to_string())

    # Stratified 70/30 partitioning by individual simulation run.
    # Stratify the 600 simulation runs by condition using seed 42.
    train_run_table, test_run_table = train_test_split(
        baseline_run_order,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=baseline_run_order['run_condition']
    )
    train_runs = set(train_run_table['run'])
    test_runs = set(test_run_table['run'])
    if train_runs & test_runs:
        raise RuntimeError('Run-level leakage: TRAIN/TEST overlap')

    train_df = df_features.loc[df_features['run'].isin(train_runs)].copy()
    test_df = df_features.loc[df_features['run'].isin(test_runs)].copy()
    if len(train_runs) != 420 or len(test_runs) != 180:
        raise ValueError('Incorrect number of TRAIN/TEST simulation runs')
    if len(train_df) != 2940 or len(test_df) != 1260:
        raise ValueError('Incorrect number of TRAIN/TEST node profiles')

    print('\n[3] SIMULATION-RUN SPLIT AUDIT')
    print('TRAIN/TEST runs:', len(train_runs), '/', len(test_runs))
    print('TRAIN/TEST node profiles:', len(train_df), '/', len(test_df))
    print('RUN overlap:', len(train_runs & test_runs))
    print('TRAIN run conditions:', train_run_table['run_condition'].value_counts().to_dict())
    print('TEST  run conditions:', test_run_table['run_condition'].value_counts().to_dict())
    print('TRAIN labels:', train_df['is_attacker'].value_counts().sort_index().to_dict())
    print('TEST  labels:', test_df['is_attacker'].value_counts().sort_index().to_dict())


    X_train = train_df[SELECTED_FEATURES].copy()
    y_train = train_df['is_attacker'].copy()
    X_test = test_df[SELECTED_FEATURES].copy()
    y_test = test_df['is_attacker'].copy()
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError('Unexpected missing values after pivot_table + fillna(0)')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use the same classifier configurations for all drop-rate variants.
    models = [
        ('Random Forest', RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'), False),
        ('SVM', SVC(kernel='rbf', class_weight='balanced'), True),
        ('Decision Tree (DT)', DecisionTreeClassifier(
            random_state=42, class_weight='balanced'), False),
        ('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance'), True),
        ('MLP', MLPClassifier(hidden_layer_sizes=(100, 50),
                              max_iter=1000, random_state=42), True)
    ]

    results = []
    predictions = test_df[
        ['run', 'config', 'repetition', 'run_condition',
         'node_id', 'is_attacker']
    ].reset_index(drop=True).copy()

    for model_name, model, do_scale in models:
        model.fit(X_train_scaled if do_scale else X_train, y_train)
        pred = model.predict(X_test_scaled if do_scale else X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        results.append({
            'Variant': variant,
            'Classifier': model_name,
            'Accuracy': accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred, zero_division=0),
            'Recall (Sens.)': recall_score(y_test, pred, zero_division=0),
            'F1-Score': f1_score(y_test, pred, zero_division=0),
            'Specificity': tn / (tn + fp) if tn+fp else np.nan,
            'FPR': fp / (tn + fp) if tn+fp else np.nan,
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
        })
        predictions[model_name] = pred
        print('\n' + '='*76 + '\n' + model_name)
        print(classification_report(y_test, pred, labels=[0, 1],
            target_names=['Benign','Blackhole'], digits=4, zero_division=0))
        print('Confusion matrix:', [[int(tn), int(fp)], [int(fn), int(tp)]])

    summary = pd.DataFrame(results)
    profile_path = out / f'WITHIN_VARIANT_{variant}_NODE_PROFILES.csv'
    compact_path = out / f'WITHIN_VARIANT_{variant}_SIX_FEATURES_LABELED.csv'
    audit_path = out / f'WITHIN_VARIANT_{variant}_RUN_PARTITIONS.csv'
    metric_path = out / f'WITHIN_VARIANT_{variant}_METRICS.csv'
    pred_path = out / f'WITHIN_VARIANT_{variant}_TEST30_PREDICTIONS.csv'

    df_features[['run', 'config', 'repetition', 'run_condition', 'node_id',
                 'host_id_full', *SELECTED_FEATURES, 'is_attacker']].assign(
        partition=lambda p: np.where(p['run'].isin(train_runs), 'TRAIN70', 'TEST30')
    ).to_csv(profile_path, index=False)
    df_features[SELECTED_FEATURES + ['is_attacker']].to_csv(compact_path, index=False)
    run_table.assign(partition=np.where(run_table['run'].isin(train_runs),
                                        'TRAIN70', 'TEST30')).to_csv(audit_path, index=False)
    summary.to_csv(metric_path, index=False)
    predictions.to_csv(pred_path, index=False)

    print('\n' + '='*104)
    print(f'{variant} | FIVE-CLASSIFIER SUMMARY')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('Simulation-run TRAIN70/TEST30 separation: PASS')
    print('Saved to:', out)
    del df_features, train_df, test_df, X_train, X_test
    gc.collect()
    return summary, run_table.assign(
        partition=np.where(run_table['run'].isin(train_runs), 'TRAIN70', 'TEST30'))

for variant in SELECT_VARIANTS:
    if variant not in RAW_FILES:
        raise ValueError(f'Unknown variant: {variant}')

all_results = []
partition_maps = {}
for variant in SELECT_VARIANTS:
    summary, audit = evaluate_one(variant)
    all_results.append(summary)
    partition_maps[variant] = audit[['repetition','run_condition','partition']].copy()

if all_results:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = OUTPUT_ROOT / 'WITHIN_VARIANT_THREE_DROP_PROBABILITIES_COMPARISON.csv'
    combined.to_csv(combined_path, index=False)
    print('\n' + '='*104)
    print('THREE-PROBABILITY COMPARISON | SIMULATION-RUN GROUPED SPLIT')
    print(combined[['Variant', 'Classifier', 'Precision', 'Recall (Sens.)',
                    'F1-Score', 'FP', 'FN']].to_string(
                        index=False, float_format=lambda x: f'{x:.4f}'))
    print('Saved comparison:', combined_path)

if len(partition_maps) > 1:
    # Compare partition membership for matching condition/repetition pairs.
    # Report differences without modifying the assigned partitions.
    first_name = next(iter(partition_maps))
    first_map = partition_maps[first_name].rename(columns={'partition':'base_partition'})
    for name, other in partition_maps.items():
        if name == first_name:
            continue
        merged = first_map.merge(other, on=['repetition','run_condition'],
                                 how='outer', validate='one_to_one')
        mismatches = int((merged['base_partition'] != merged['partition']).sum())
        print(f'Partition membership mismatches: {first_name} vs {name}: {mismatches}')
        if mismatches:
            raise RuntimeError(f'Inconsistent TRAIN70/TEST30 membership: {first_name} vs {name}')

print('\nCOMPLETED: SIMULATION-RUN-GROUPED 70/30 | NO CHARTS CREATED')
print('Evaluation: WITHIN-VARIANT (Section 4.4.2)')