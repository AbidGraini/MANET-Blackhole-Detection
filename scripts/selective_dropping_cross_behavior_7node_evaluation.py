# =============================================================================
# Section 4.4.4 — Cross-Behavior Evaluation under Different Packet-Dropping Probabilities
# Train on Drop_P100% and Drop_P25% TRAIN70, counting normal controls once.
# Evaluate on Drop_P50% TEST30; exclude test data from fitting and scaling.
# Partition by simulation run (70/30), stratified by condition, seed 42.
# Six fixed behavioral features and five classifiers.
# Outputs metrics, predictions, partitions and fitted models; inputs stay unchanged.
# =============================================================================

from pathlib import Path
import os
import gc
import re
import warnings
import joblib
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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# 1) Repository-relative defaults; environment overrides affect locations only.
PROJECT_ROOT = Path(os.environ.get('MANET_PROJECT_ROOT', str(Path.cwd()))).expanduser()
DATA_DIR = Path(os.environ.get('MANET_DATA_DIR', str(PROJECT_ROOT / 'data'))).expanduser()
OUTPUT_DIR = Path(os.environ.get(
    'MANET_MIXED_DROP_OUTPUT',
    str(PROJECT_ROOT / 'results' / 'CROSS_BEHAVIOR_Drop_P100%_Drop_P25%_TO_Drop_P50%')
)).expanduser()
RAW_FILES = {
    'Drop_P100%': DATA_DIR / '7NODE1BLACK300RUN600S_100_PROB.csv',
    'Drop_P25%':  DATA_DIR / '7NODE1BLACK300RUN600S25PROB.csv',
    'Drop_P50%':  DATA_DIR / '7NODE1BLACK300RUN600S_50_PROB.csv',
}
TEST_SIZE = 0.30
SPLIT_SEED = 42
EXPECTED_REPETITIONS = 300
EXPECTED_NODES = 7
ATTACKER_NODE = 5

# Fixed feature names and ordering
SELECTED_FEATURES = [
    'numRREQsSent:count',
    'numRREPsSent:count',
    'packetSent:count',
    'numPacketsForwarded:count',
    'arpReplySent:count',
    'linkBroken:count',
]
NORMAL_CONFIGS = {'DatasetNormal', 'BH7_Normal'}
ATTACK_CONFIGS = {
    'Drop_P100%': {'BH7_Drop100'},
    'Drop_P25%': {'BH7_Drop25'},
    'Drop_P50%': {'BH7_Drop50'},
}

for variant, path in RAW_FILES.items():
    if not path.is_file():
        raise FileNotFoundError(f'{variant}: missing input CSV: {path}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_run_name(run):
    m = re.match(r'^([A-Za-z][A-Za-z0-9_]*)-#?(\d+)(?:-|$)', str(run))
    if not m:
        raise ValueError(f'Unrecognized OMNeT++ run name: {run!r}')
    return m.group(1), int(m.group(2))


def load_profiles(variant):
    """Read each raw CSV once; form node profiles using pivot_table and fillna(0)."""
    path = RAW_FILES[variant]
    print(f'\n[{variant}] Reading {path.name} ({path.stat().st_size / 1048576:.2f} MB)')
    raw = pd.read_csv(path, low_memory=False)
    required = {'run', 'module', 'name', 'value'}
    if not required.issubset(raw.columns):
        raise ValueError(f'{variant}: missing raw CSV columns: {required - set(raw.columns)}')
    raw['host_id_full'] = raw['module'].str.extract(
        r'(AODVNetwork\.host\[\d+\])', expand=False)
    raw = raw.dropna(subset=['host_id_full']).copy()
    raw['value'] = pd.to_numeric(raw['value'], errors='coerce')
    profiles = raw.pivot_table(
        index=['run', 'host_id_full'], columns='name', values='value'
    ).reset_index()
    profiles.columns.name = None
    del raw
    gc.collect()

    absent = [f for f in SELECTED_FEATURES if f not in profiles.columns]
    if absent:
        raise ValueError(f'{variant}: missing required features: {absent}')
    # All six measurements must be available for each node.
    if profiles[SELECTED_FEATURES].isna().any().any():
        raise ValueError(f'{variant}: a selected feature is absent for a node')
    profiles = profiles.fillna(0)
    if not np.isfinite(profiles[SELECTED_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f'{variant}: non-finite six-feature values')

    # Preserve the pivot-derived lexicographical run order for the seeded split.
    runs = profiles[['run']].drop_duplicates().reset_index(drop=True)
    parsed = runs['run'].map(parse_run_name)
    runs['config'] = parsed.map(lambda p: p[0])
    runs['repetition'] = parsed.map(lambda p: p[1])
    permitted = NORMAL_CONFIGS | ATTACK_CONFIGS[variant]
    unknown = set(runs['config']) - permitted
    if unknown:
        raise ValueError(f'{variant}: unexpected run config(s): {sorted(unknown)}')
    runs['run_condition'] = np.where(
        runs['config'].isin(ATTACK_CONFIGS[variant]), 'Attack', 'Normal')

    if len(runs) != 2*EXPECTED_REPETITIONS:
        raise ValueError(f'{variant}: expected 600 runs; got {len(runs)}')
    if runs.duplicated(['run_condition', 'repetition']).any():
        raise ValueError(f'{variant}: repeated run-condition/repetition keys')
    for cond in ['Normal', 'Attack']:
        reps = set(runs.loc[runs['run_condition'].eq(cond), 'repetition'])
        if reps != set(range(EXPECTED_REPETITIONS)):
            raise ValueError(f'{variant}: incomplete repetition IDs for {cond}')

    profiles = profiles.merge(runs, on='run', how='left', sort=False,
                              validate='many_to_one')
    if not profiles[['run']].drop_duplicates().reset_index(drop=True)['run'].equals(runs['run']):
        raise RuntimeError(f'{variant}: pivot-derived run order changed')
    profiles['node_id'] = profiles['host_id_full'].str.extract(
        r'host\[(\d+)\]', expand=False).astype(int)
    counts = profiles.groupby('run')['node_id'].agg(['size','nunique'])
    if not (counts['size'].eq(EXPECTED_NODES) &
            counts['nunique'].eq(EXPECTED_NODES)).all():
        raise ValueError(f'{variant}: incomplete 7-node run profiles')
    if not profiles['node_id'].between(0, EXPECTED_NODES-1).all():
        raise ValueError(f'{variant}: invalid host index')
    profiles['is_attacker'] = (
        profiles['run_condition'].eq('Attack') &
        profiles['node_id'].eq(ATTACKER_NODE)).astype(int)
    if len(profiles) != 4200 or profiles['is_attacker'].value_counts().to_dict() != {0:3900,1:300}:
        raise ValueError(f'{variant}: invalid profile/label count')
    print('   RUNS=600 | NODE_PROFILES=4200 | NORMAL=3900 | BLACKHOLE=300: PASS')
    return profiles, runs


def normal_control(profiles):
    return profiles.loc[profiles['run_condition'].eq('Normal'),
        ['repetition','node_id', *SELECTED_FEATURES]
    ].sort_values(['repetition','node_id']).reset_index(drop=True)


def ensure_same_normal(base, other, name):
    if not base[['repetition','node_id']].equals(other[['repetition','node_id']]):
        raise RuntimeError(f'{name}: Normal-control node identity mismatch')
    diff = base[SELECTED_FEATURES].to_numpy() != other[SELECTED_FEATURES].to_numpy()
    if diff.any():
        raise RuntimeError(f'{name}: Normal controls differ in {int(diff.sum())} feature values')
    print(f'   Normal controls vs Drop_P100%: PASS | 2100 node profiles, 12600 feature values identical')


def add_partition(runs, partition_map, variant):
    audit = runs.merge(partition_map, on=['run_condition','repetition'],
                       how='left', validate='one_to_one', sort=False)
    if audit['partition'].isna().any():
        raise RuntimeError(f'{variant}: partition alignment incomplete')
    if audit['partition'].value_counts().to_dict() != {'TRAIN70':420,'TEST30':180}:
        raise RuntimeError(f'{variant}: incorrect split size')
    if audit.groupby(['run_condition','partition']).size().to_dict() != {
        ('Attack','TEST30'):90, ('Attack','TRAIN70'):210,
        ('Normal','TEST30'):90, ('Normal','TRAIN70'):210}:
        raise RuntimeError(f'{variant}: incorrect stratified counts')
    audit.to_csv(OUTPUT_DIR / f'CROSS_BEHAVIOR_{variant}_RUN_PARTITIONS.csv', index=False)
    return audit


print('=' * 110)
print('MIXED TRAIN: Drop_P100% TRAIN70 + Drop_P25% TRAIN70 -> UNSEEN TEST: Drop_P50% TEST30')
print('Protocol: SIMULATION-RUN-GROUPED 70/30, stratified by condition, seed=42')
print('=' * 110)

# 2) Derive the run-grouped split from Drop_P100%.
p100, r100 = load_profiles('Drop_P100%')
run_order = r100[['run','run_condition']].copy()
train_runs_table, test_runs_table = train_test_split(
    run_order, test_size=TEST_SIZE, random_state=SPLIT_SEED,
    stratify=run_order['run_condition'])
train_runs100 = set(train_runs_table['run'])
test_runs100 = set(test_runs_table['run'])
if train_runs100 & test_runs100:
    raise RuntimeError('Leakage inside the Drop_P100% run-grouped split')
r100['partition'] = np.where(r100['run'].isin(train_runs100), 'TRAIN70', 'TEST30')
partition_map = r100[['run_condition','repetition','partition']].copy()
if partition_map.duplicated(['run_condition','repetition']).any():
    raise RuntimeError('Duplicate source partition keys')
if r100['partition'].value_counts().to_dict() != {'TRAIN70':420,'TEST30':180}:
    raise RuntimeError('Incorrect Drop_P100% partition sizes')
r100.to_csv(OUTPUT_DIR / 'CROSS_BEHAVIOR_Drop_P100%_RUN_PARTITIONS.csv', index=False)
partition_map.to_csv(OUTPUT_DIR / 'CROSS_BEHAVIOR_ALIGNED_PARTITION_KEYS.csv', index=False)

normal100 = normal_control(p100)
training100 = p100.loc[p100['run'].isin(train_runs100)].copy()
if len(training100) != 2940 or training100['is_attacker'].value_counts().to_dict() != {0:2730,1:210}:
    raise RuntimeError('Drop_P100% TRAIN70 profile counts incorrect')
training100['source_variant'] = 'Drop_P100%'

# 3) Include the Drop_P25% attack runs assigned to TRAIN70.
# Avoid duplicating 210 identical normal-control training runs.
p25, r25 = load_profiles('Drop_P25%')
ensure_same_normal(normal100, normal_control(p25), 'Drop_P25%')
audit25 = add_partition(r25, partition_map, 'Drop_P25%')
train_attack25_runs = set(audit25.loc[
    audit25['partition'].eq('TRAIN70') & audit25['run_condition'].eq('Attack'), 'run'])
training25_attack = p25.loc[p25['run'].isin(train_attack25_runs)].copy()
if len(train_attack25_runs) != 210 or len(training25_attack) != 1470:
    raise RuntimeError('Expected 210 Attack runs and 1470 profiles from Drop_P25% TRAIN70')
if training25_attack['is_attacker'].value_counts().to_dict() != {0:1260,1:210}:
    raise RuntimeError('Invalid Drop_P25% Attack TRAIN70 labels')
training25_attack['source_variant'] = 'Drop_P25%'

# Attack-run repetition IDs are shared between the source variants.
# The aligned Drop_P50% TEST30 repetition IDs are excluded from training.
training = pd.concat([training100, training25_attack], ignore_index=True)
# 210 normal runs (1,470 profiles) + 420 attack runs (2,940 profiles).
if len(training) != 4410 or training['is_attacker'].value_counts().to_dict() != {0:3990,1:420}:
    raise RuntimeError('Incorrect mixed training counts')
if len(training.loc[training['run_condition'].eq('Normal')]) != 1470:
    raise RuntimeError('Duplicate pure-Normal runs entered mixed training')
if training.loc[training['run_condition'].eq('Normal'),
                ['repetition','node_id']].duplicated().any():
    raise RuntimeError('Duplicated pure-Normal node profile in mixed training')

# Retain the source variant and run keys for each training profile.
training['partition'] = 'TRAIN70'
train_run_keys = set(zip(training['run_condition'], training['repetition']))
training[[
    'source_variant','run','run_condition','repetition','node_id',
    *SELECTED_FEATURES,'is_attacker','partition'
]].to_csv(OUTPUT_DIR / 'CROSS_BEHAVIOR_TRAIN70_FEATURES_PROVENANCE.csv', index=False)
print('\n[MIXED TRAIN AUDIT]')
print('Drop_P100%: 210 Normal runs + 210 Attack runs')
print('Drop_P25% : 0 Normal runs (exact duplicates excluded) + 210 Attack runs')
print('Unique simulation runs in mixed train: 630')
print('Mixed training profiles:', len(training), '| labels:',
      training['is_attacker'].value_counts().sort_index().to_dict())
print('Normal controls included once: PASS')

# Release source profile tables before loading Drop_P50%.
del p100, p25, r100, r25, training100, training25_attack, audit25

gc.collect()

# 4) Evaluate only on the held-out Drop_P50% test partition.
p50, r50 = load_profiles('Drop_P50%')
ensure_same_normal(normal100, normal_control(p50), 'Drop_P50%')
audit50 = add_partition(r50, partition_map, 'Drop_P50%')
test50_runs = set(audit50.loc[audit50['partition'].eq('TEST30'),'run'])
test50 = p50.loc[p50['run'].isin(test50_runs)].copy()
if len(test50_runs) != 180 or len(test50) != 1260:
    raise RuntimeError('Expected Drop_P50% TEST30: 180 runs / 1260 node profiles')
if test50['is_attacker'].value_counts().to_dict() != {0:1170,1:90}:
    raise RuntimeError('Incorrect Drop_P50% TEST30 label counts')

test_run_keys = set(zip(test50['run_condition'], test50['repetition']))
if train_run_keys & test_run_keys:
    raise RuntimeError('LEAKAGE: source TRAIN / Drop_P50% TEST (condition,repetition) overlap')
if (training['run_condition'].eq('Attack') & training['source_variant'].eq('Drop_P50%')).any():
    raise RuntimeError('LEAKAGE: Drop_P50% Attack profiles were used in training')
if len(train_run_keys & test_run_keys) != 0:
    raise RuntimeError('Unexpected TRAIN/TEST key overlap')

print('\n[EXTERNAL TEST AUDIT]')
print('Target: Drop_P50% TEST30 only | runs:', len(test50_runs),
      '| profiles:',len(test50),'| labels:',
      test50['is_attacker'].value_counts().sort_index().to_dict())
print('Source TRAIN / target TEST overlap for (condition, repetition): 0 | PASS')
print('Aligned run-wise TEST30 membership: PASS')
print('Drop_P50% used to fit/select any features, scaler, model or threshold: NO')

# 5) Fit classifiers after checking training/test separation.
X_train = training[SELECTED_FEATURES].copy()
y_train = training['is_attacker'].copy()
test50 = test50.sort_values(['run','node_id']).reset_index(drop=True)
X_test = test50[SELECTED_FEATURES].copy()
y_test = test50['is_attacker'].copy()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = [
    ('Random Forest', RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'), False),
    ('SVM', SVC(kernel='rbf', class_weight='balanced'), True),
    ('Decision Tree (DT)', DecisionTreeClassifier(
        random_state=42, class_weight='balanced'), False),
    ('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance'), True),
    ('MLP', MLPClassifier(hidden_layer_sizes=(100, 50),
                          max_iter=1000, random_state=42), True),
]

all_results = []
predictions = test50[['run','config','repetition','run_condition',
                       'node_id','is_attacker']].copy()

for name, model, uses_scaler in models:
    print('\n' + '=' * 90)
    print('TRAIN MIXED Drop_P100%+Drop_P25% TRAIN70 -> TEST Drop_P50% TEST30 |', name)
    model.fit(X_train_scaled if uses_scaler else X_train, y_train)
    y_pred = model.predict(X_test_scaled if uses_scaler else X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    all_results.append({
        'Train': 'Drop_P100% + Drop_P25% TRAIN70 (shared Normal once)',
        'Test': 'Drop_P50% TEST30',
        'Classifier': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1_Blackhole': f1_score(y_test, y_pred, zero_division=0),
        'Macro_F1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'Specificity': tn / (tn + fp) if (tn+fp) else np.nan,
        'FPR': fp / (tn + fp) if (tn+fp) else np.nan,
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    })
    predictions[name] = y_pred
    print(classification_report(y_test, y_pred, labels=[0,1],
                                target_names=['Benign','Blackhole'],
                                digits=4, zero_division=0))
    print('Confusion matrix [0,1]:', [[int(tn),int(fp)],[int(fn),int(tp)]])
    safe_name = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_')
    joblib.dump(model, OUTPUT_DIR / f'CROSS_BEHAVIOR_{safe_name}_MODEL.joblib')

joblib.dump(scaler, OUTPUT_DIR / 'CROSS_BEHAVIOR_TRAIN70_SCALER.joblib')
summary = pd.DataFrame(all_results)
summary.to_csv(OUTPUT_DIR / 'CROSS_BEHAVIOR_Drop_P100%_Drop_P25%_TO_Drop_P50%_METRICS.csv', index=False)
predictions.to_csv(OUTPUT_DIR / 'CROSS_BEHAVIOR_Drop_P50%_TEST30_PREDICTIONS.csv', index=False)

print('\n' + '=' * 124)
print('FINAL CROSS-BEHAVIOR EVALUATION | Drop_P100% + Drop_P25% TRAIN70 -> Drop_P50% TEST30')
print('=' * 124)
print(summary[['Classifier','Precision','Recall','F1_Blackhole',
               'Macro_F1','FP','FN','TP','TN']].to_string(
      index=False,float_format=lambda x:f'{x:.4f}'))
print('\nCHECK: exact six features; fixed classifier settings; run-grouped split seed 42')
print('TRAIN source attacks: Drop_P100% 210 runs + Drop_P25% 210 runs')
print('TRAIN Normal: Drop_P100% 210 runs (Drop_P25% duplicate normals excluded)')
print('TEST: Drop_P50% 90 Normal + 90 Attack runs | no target fit/scale/tuning')
print('NO target-trained model | NO figures created')
print('IMPORTANT: mixed training has 420 Blackhole profiles versus 210 in')
print('the Drop_P100%-only baseline. A gain by itself does not isolate diversity')
print('from sample-count effects. Model behavior remains within the same AODV RREP family.')
print('Saved outputs:', OUTPUT_DIR)