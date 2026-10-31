# NAMING AND OUTPUT CONTRACT
# DEVELOPMENT_TRAIN_70 = 70% of unique simulation runs for feature selection.
# FINAL_CLASSIFICATION_HOLDOUT_30 = the remaining 30% of runs, never used in selection.
# OUTER_FOLD_HOLDOUT = grouped CV holdout *inside* DEVELOPMENT_TRAIN_70.
# The fixed, stratified 70/30 run assignment is generated from RAW input data,
# saved to splits/{7NODE,25NODE,50NODE}, and validated on later executions.
# Exact holdout run IDs are also exported for separate classifier evaluation.
# This script performs feature selection; it does not evaluate final classifiers.
# Five outer and five inner run-grouped folds use MI, PFI and TreeSHAP,
# followed by consensus, stability filtering and weighted-MI synthesis.
# The selected feature algorithms, thresholds, and inner/outer folds are unchanged.

# =============================================================================
# TRAINING-ONLY FEATURE SELECTION AND AUDIT EXPORTS
# This script exports fold-level Global Top-6 and article-level feature lists,
# audit records and saved run partitions; it does not produce classifier metrics.
# =============================================================================
# ============================================================================
# STABILITY-DRIVEN FEATURE SELECTION FOR BLACKHOLE DETECTION IN MANETs
#
# FINAL METHODOLOGY:
#   - Development scenarios: 7 / 25 / 50 nodes
#   - 75-node scenario excluded from feature-selection inputs
#   - Outer CV grouped by simulation run
#   - Inner CV grouped by simulation run
#   - Zero-variance filtering fitted on INNER-TRAINING data only
#   - Correlation filtering fitted on INNER-TRAINING data only
#   - Majority downsampling performed on INNER-TRAINING data only
#   - MI + PFI + TreeSHAP evidence
#   - Top 40% (minimum 10) per evidence source
#   - 2-of-3 consensus voting
#   - Stability threshold >= 0.80 across five inner folds
#   - Each outer fold retains the full stable candidate pool for synthesis
#   - Cross-scenario synthesis uses sample-proportional weighted MI
#   - Final global feature budget = 6
# ============================================================================

import pandas as pd
import numpy as np
import shap
import warnings
import os
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'left')


# ============================================================================
# 1. NORMALIZATION FUNCTION
# ============================================================================

def normalize(x):
    x = np.array(x)

    if np.max(x) == np.min(x):
        return np.ones(len(x))

    return (x - np.min(x)) / (np.max(x) - np.min(x))


# ============================================================================
# 2. GROUPED STRATIFIED SPLITTING BY SIMULATION RUN
# ============================================================================
#
# StratifiedKFold splits unique run identifiers, not node-level records.
# The run assignments are then mapped back to all corresponding node rows.
#
# Stratification is performed according to run type:
#   0 = Normal run
#   1 = Attacker run
#
# All nodes from one run remain in exactly one partition.
# ============================================================================

def grouped_stratified_run_split(
    run_series,
    n_splits=5,
    shuffle=True,
    random_state=42
):
    """
    Generate row-level train/test indices after splitting at simulation-run level.

    Parameters
    ----------
    run_series : pd.Series
        Simulation-run identifier corresponding to each node-level row.

    Returns
    -------
    row_train_idx, row_test_idx, train_runs, test_runs
    """

    run_series = run_series.astype(str)

    # ------------------------------------------------------------------------
    # Build one row per unique simulation run
    # ------------------------------------------------------------------------

    run_table = pd.DataFrame({
        'run': run_series
    })

    run_table['run_condition'] = (
        run_table['run']
        .str.contains('Attacker', case=False, na=False)
        .astype(int)
    )

    run_table = (
        run_table[
            ['run', 'run_condition']
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------------
    # Audit run distribution
    # ------------------------------------------------------------------------

    run_counts = (
        run_table['run_condition']
        .value_counts()
        .sort_index()
    )

    if len(run_counts) < 2:
        raise RuntimeError(
            "Both Normal and Attacker simulation runs are required."
        )

    if run_counts.min() < n_splits:
        raise RuntimeError(
            f"Not enough simulation runs for {n_splits}-fold "
            f"stratification.\n{run_counts}"
        )

    # ------------------------------------------------------------------------
    # Stratification is applied to unique simulation runs.
    # ------------------------------------------------------------------------

    run_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    dummy_X = np.zeros(len(run_table))

    for run_train_idx, run_test_idx in run_cv.split(
        dummy_X,
        run_table['run_condition']
    ):

        train_runs = set(
            run_table.iloc[run_train_idx]['run']
        )

        test_runs = set(
            run_table.iloc[run_test_idx]['run']
        )

        # --------------------------------------------------------------------
        # Hard leakage audit
        # --------------------------------------------------------------------

        overlap = train_runs.intersection(test_runs)

        if len(overlap) != 0:
            raise RuntimeError(
                f"RUN-LEVEL LEAKAGE DETECTED: {overlap}"
            )

        # --------------------------------------------------------------------
        # Convert run partitions back to node-row positions
        # --------------------------------------------------------------------

        train_mask = (
            run_series
            .isin(train_runs)
            .to_numpy()
        )

        test_mask = (
            run_series
            .isin(test_runs)
            .to_numpy()
        )

        row_train_idx = np.flatnonzero(train_mask)
        row_test_idx = np.flatnonzero(test_mask)

        yield (
            row_train_idx,
            row_test_idx,
            train_runs,
            test_runs
        )


# ============================================================================
# 3. INPUT SCENARIOS
# ============================================================================
#
# The 75-node scenario is not an input to the feature-selection procedure.
#
# Development scenarios:
#   7 nodes
#   25 nodes
#   50 nodes
#
# 75-node scenario is NOT loaded and therefore cannot influence
# the feature-selection process.
# ============================================================================

# Local Windows RAW dataset directory (fixed, independent of Jupyter working folder).
# Files X7, X25, X50 are read from this directory; no external split files required.
DATA_DIR = Path(r"F:/2026/RAWDATASET")
OUTPUT_DIR = Path(os.environ.get("MANET_SELECTION_OUTPUT", str(Path.cwd() / "selection_outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N_OUTER_FOLDS = 5
FINAL_FEATURE_BUDGET = 6


scenario_files = [

    {
        'name': '7 Nodes',
        'path': DATA_DIR / '7NODE1BLACK300RUN600S.csv',
        'attacker': ['host[5]']
    },

    {
        'name': '25 Nodes',
        'path': DATA_DIR / '25NODE1BLACK500RUN600S.csv',
        'attacker': ['host[15]']
    },

    {
        'name': '50 Nodes',
        'path': DATA_DIR / '50NOD2BLACK400RUN100S.csv',
        'attacker': [
            'host[20]',
            'host[35]'
        ]
    }

]


# ============================================================================
# REPRODUCIBLE RUN-GROUPED 70/30 PARTITION — CREATED FROM THE RAW SCENARIOS
# This exactly follows the matched benchmark's split method:
#   pivot_table (sorted run IDs) -> one row per run -> stratified train_test_split
#   test_size=0.30, random_state=42, stratify=run_condition (Normal/Attack).
# The actual run ID, not a node record, is the unit of partitioning.
# On subsequent runs, the saved manifest must match the regenerated assignment.
# Only TRAIN runs enter the unchanged nested 5 x 5 feature-selection pipeline.
# ============================================================================

SPLIT_ROOT = Path(
    os.environ.get("MANET_SPLITS_DIR", str(Path.cwd() / "splits"))
)

SPLIT_FILES = {
    "7 Nodes": SPLIT_ROOT / "7NODE" / "7node_run_split.csv",
    "25 Nodes": SPLIT_ROOT / "25NODE" / "25node_run_split.csv",
    "50 Nodes": SPLIT_ROOT / "50NODE" / "50node_run_split.csv",
}

RUN_SPLIT_TEST_SIZE = 0.30
RUN_SPLIT_RANDOM_STATE = 42

# These SHA-256 fingerprints come from the retained, original matched-benchmark
# TRAIN/TEST run-ID manifests, and guard against accidental partition changes.
EXPECTED_RUN_SPLIT_HASHES = {
    "7 Nodes": (
        "8b51ab9ea8ba69877323b850835605e1e26b680de107955995a68dad84baa7fc",
        "153032e574b00fb3fdf8c28c507e92fba6e12b262ee1e5cc857f1ee38b0724e9",
    ),
    "25 Nodes": (
        "fca32ab4da28036c844e3ae3bfcc2605aa70b17e809c2cec0bbfe6bdbb52821e",
        "4ebcdca93a23001623fcf8d66c0225f01901ea20549b6be052ddeb9368965543",
    ),
    "50 Nodes": (
        "a288a24b8634f5f8578bb2a0a93003bf6b9d1e871f505be6d301c89931180f06",
        "23e2af454e50b12572eb48e524644f267fcfa751fce7e34dfcb658fc7f17b666",
    ),
}
EXPECTED_RUN_COUNTS = {"7 Nodes": 600, "25 Nodes": 1000, "50 Nodes": 800}


def _run_hash(run_ids):
    """SHA-256 fingerprint of the exact names of run IDs in a partition."""
    import hashlib
    joined = "\n".join(sorted(map(str, run_ids))).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def get_strict_training_run_manifest(scenario_name, raw_run_ids):
    """Generate the deterministic benchmark split from RAW run IDs and save it."""
    path = SPLIT_FILES[scenario_name]
    # pandas pivot_table (in the original benchmark) sorts the run index;
    # sorting RAW unique IDs here reproduces precisely that input ordering.
    run_ids = sorted(set(map(str, raw_run_ids)))
    if len(run_ids) != EXPECTED_RUN_COUNTS[scenario_name]:
        raise RuntimeError(
            f"Unexpected number of RAW runs in {scenario_name}: "
            f"{len(run_ids)} rather than {EXPECTED_RUN_COUNTS[scenario_name]}"
        )

    run_table = pd.DataFrame({'run': run_ids})
    run_table['run_condition'] = np.where(
        run_table['run'].str.contains('Attacker', case=False, na=False),
        'Attack', 'Normal'
    )
    if run_table['run_condition'].nunique() != 2:
        raise RuntimeError(f"Both Normal and Attack runs are required: {scenario_name}")

    train_table, test_table = train_test_split(
        run_table,
        test_size=RUN_SPLIT_TEST_SIZE,
        random_state=RUN_SPLIT_RANDOM_STATE,
        stratify=run_table['run_condition'],
    )
    train_table = train_table.copy()
    test_table = test_table.copy()
    train_table['partition'] = 'TRAIN'
    test_table['partition'] = 'TEST'
    generated = pd.concat([train_table, test_table], ignore_index=True)

    train_ids = set(train_table['run'])
    test_ids = set(test_table['run'])
    if train_ids & test_ids or len(train_ids | test_ids) != len(run_ids):
        raise RuntimeError(f"Invalid run-grouped split: {scenario_name}")
    expected_train_hash, expected_test_hash = EXPECTED_RUN_SPLIT_HASHES[scenario_name]
    if _run_hash(train_ids) != expected_train_hash or _run_hash(test_ids) != expected_test_hash:
        raise RuntimeError(
            f"The regenerated RAW split for {scenario_name} does not match "
            "the original benchmark's frozen train/test run IDs. "
            "Check RAW input file, simulation run names, and split method. "
            "No split was saved or replaced."
        )

    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {'run', 'run_condition', 'partition'}
        if not required.issubset(existing.columns):
            raise RuntimeError(f"Saved split lacks {required - set(existing.columns)}: {path}")
        existing = existing[['run', 'run_condition', 'partition']].copy()
        existing['run_condition'] = existing['run_condition'].str.strip().str.lower()
        existing['partition'] = existing['partition'].str.strip().str.upper()
        expected_assignments = {
            str(r.run): (str(r.run_condition).lower(), str(r.partition))
            for r in generated.itertuples(index=False)
        }
        saved_assignments = {
            str(r.run): (str(r.run_condition), str(r.partition))
            for r in existing.itertuples(index=False)
        }
        if (len(existing) != len(expected_assignments)
                or saved_assignments != expected_assignments):
            raise RuntimeError(
                f"Saved split differs from deterministic RAW split: {path}. "
                "Refusing to replace the earlier run assignments."
            )
        print(f"EXISTING RUN SPLIT VERIFIED FROM RAW: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        generated.to_csv(path, index=False, encoding='utf-8')
        print(f"RUN SPLIT GENERATED FROM RAW: {path}")

    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    fields = {'run', 'run_condition', 'partition'}
    if not fields.issubset(m.columns):
        raise RuntimeError(f"Bad run split {path}: missing {fields - set(m.columns)}")
    m = m[['run', 'run_condition', 'partition']].copy()
    m['partition'] = m['partition'].str.upper().str.strip()
    m['run_condition'] = m['run_condition'].str.strip().str.lower()
    if not m['partition'].isin(['TRAIN', 'TEST']).all():
        raise RuntimeError(f"Bad partition in {path}")
    if not m['run_condition'].isin(['normal', 'attack']).all():
        raise RuntimeError(f"Bad condition in {path}")
    if not m['run'].ne('').all() or m['run'].duplicated().any():
        raise RuntimeError(f"Empty or duplicated run ID in {path}")
    expected_condition = m['run'].str.contains('Attacker', case=False, na=False)
    declared_condition = m['run_condition'].eq('attack')
    if not expected_condition.equals(declared_condition):
        raise RuntimeError(f"Run names and condition labels disagree in {path}")
    verified_train = set(m.loc[m['partition'].eq('TRAIN'), 'run'])
    verified_test = set(m.loc[m['partition'].eq('TEST'), 'run'])
    if verified_train != train_ids or verified_test != test_ids:
        raise RuntimeError(f"Reloaded run split differs from RAW assignment: {path}")
    if not 0.68 <= len(train_ids) / len(m) <= 0.72:
        raise RuntimeError(f"Not a 70/30 run split: {path}")
    print(f"  {scenario_name}: TRAIN={len(train_ids)}, TEST={len(test_ids)}, "
          f"overlap=0, original SHA-256 matched")
    return path, m, train_ids, test_ids

# ============================================================================
# EXPLICIT, REUSABLE 70/30 RUN-ID RECORDS
# Run identifiers are copied from the newly generated or verified manifests.
# TEST IDs are exported as explicit strings for separate classification.
# Existing manifests cannot silently be replaced with different partitions.
# ============================================================================

FIXED_PARTITIONS_DIR = OUTPUT_DIR / '00_FIXED_70_30_RUN_IDS'
FIXED_PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)


def _save_csv_without_repartition(path, expected_df):
    """Write once; on later execution, require exact content equivalence."""
    expected_df = expected_df.reset_index(drop=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(existing.columns) != list(expected_df.columns) or not existing.equals(expected_df):
            raise RuntimeError(
                f'SAVED RUN IDs CHANGED: {path}\n'
                'Refusing to overwrite an existing partition. '
                'Investigate the difference before proceeding.'
            )
    else:
        expected_df.to_csv(path, index=False, encoding='utf-8')

    reread = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not reread.equals(expected_df):
        raise RuntimeError(f'RUN-ID round-trip mismatch: {path}')
    return str(path)


def save_exact_run_id_manifests(scenario_name, manifest, train_ids, test_ids):
    """Export actual string run identifiers—not only hashes or percentages."""
    key = 'X' + scenario_name.split()[0]  # X7, X25, X50
    base_cols = ['run', 'run_condition', 'partition']
    complete = manifest[base_cols].sort_values('run', kind='stable').reset_index(drop=True)
    train_df_ids = complete.loc[complete['partition'].eq('TRAIN')].reset_index(drop=True)
    test_df_ids = complete.loc[complete['partition'].eq('TEST')].reset_index(drop=True)

    if (set(train_df_ids['run']) != train_ids or
            set(test_df_ids['run']) != test_ids or
            set(train_df_ids['run']) & set(test_df_ids['run'])):
        raise RuntimeError(f'Manifest run-ID mismatch before export: {scenario_name}')
    if len(complete) != len(train_ids) + len(test_ids):
        raise RuntimeError(f'Manifest contains duplicates: {scenario_name}')

    files = {
        'full_manifest_csv': FIXED_PARTITIONS_DIR / f'{key}_FIXED_70_30_MANIFEST.csv',
        'train70_ids_csv': FIXED_PARTITIONS_DIR / f'{key}_DEVELOPMENT_TRAIN_70_RUN_IDS.csv',
        'test30_ids_csv': FIXED_PARTITIONS_DIR / f'{key}_FINAL_CLASSIFICATION_HOLDOUT_30_RUN_IDS.csv',
    }
    paths = {
        'full_manifest_csv': _save_csv_without_repartition(files['full_manifest_csv'], complete),
        'train70_ids_csv': _save_csv_without_repartition(files['train70_ids_csv'], train_df_ids),
        'test30_ids_csv': _save_csv_without_repartition(files['test30_ids_csv'], test_df_ids),
    }
    # Verify the exact saved TEST IDs are never in the selection TRAIN data.
    saved_train = pd.read_csv(paths['train70_ids_csv'], dtype=str, keep_default_na=False)
    saved_test = pd.read_csv(paths['test30_ids_csv'], dtype=str, keep_default_na=False)
    if set(saved_train['run']) != train_ids or set(saved_test['run']) != test_ids:
        raise RuntimeError(f'Run-ID identity check failed after export: {scenario_name}')
    if set(saved_test['run']) & set(saved_train['run']):
        raise RuntimeError(f'Train/test overlap found in saved IDs: {scenario_name}')

    print(f'FIXED PARTITION FILES SAVED FOR {scenario_name}:')
    print('  DEVELOPMENT_TRAIN_70 IDs:', paths['train70_ids_csv'])
    print('  FINAL_CLASSIFICATION_HOLDOUT_30 IDs:', paths['test30_ids_csv'])
    print('  MANIFEST :', paths['full_manifest_csv'])
    print('  TEST run IDs validated against supplied manifest:', len(test_ids))
    return paths


HOLDOUT_AUDITS = []


# ============================================================================
# 4. PREDEFINED LEAKAGE FEATURES
# ============================================================================

leakage_features = [
    'rreqDropCount:count',
    'rrepSpoofedCount:count',
    'dataDropCount:count'
]


# ============================================================================
# 5. GLOBAL REGISTRIES
# ============================================================================

scenario_local_mi_registry = {}

scenario_actual_samples = {}

scenario_audit_trail = {}

# Physical schema retained only for final compatibility auditing.
scenario_physical_schema = {}
scenario_leakage_clean_schemas = {}
scenario_fold_selected_mi = {}
scenario_fold_ranked = {}
scenario_fold_diagnostics = []
inner_feature_filter_audits = []



print("=" * 110)

print(
    "STARTING GROUPED CUMULATIVE MI PIPELINE "
    "(SIMULATION-RUN LEVEL SPLITTING)"
)

print("=" * 110)

print(
    "\nFEATURE-SELECTION INPUT SCENARIOS:"
)

for s in scenario_files:
    print(f"  - {s['name']}")

missing_inputs = [str(sc['path']) for sc in scenario_files if not sc['path'].is_file()]
if missing_inputs:
    raise FileNotFoundError(
        'Missing source development CSVs. Set MANET_DATA_DIR or edit DATA_DIR.\n'
        + '\n'.join(missing_inputs)
    )

print(
    "\n75 Nodes: EXCLUDED FROM FEATURE SELECTION"
)

print("=" * 110)


# ============================================================================
# 6. PROCESS EACH SCENARIO
# ============================================================================

for scenario in scenario_files:

    scen_name = scenario['name']

    print(
        "\n" + "=" * 110
    )

    print(
        f"PROCESSING SCENARIO: {scen_name}"
    )

    print(
        "=" * 110
    )


    # ========================================================================
    # 6.1 LOAD DATA
    # ========================================================================

    try:

        df_raw = pd.read_csv(
            scenario['path']
        )

        # Create or verify the fixed 70/30 split from RAW run names BEFORE
        # any feature filtering or selection. Final TEST runs are excluded.
        if 'run' not in df_raw.columns:
            raise RuntimeError(f"No run column: {scenario['path']}")
        raw_run_ids = set(df_raw['run'].astype(str).unique())
        split_path, manifest, train_run_ids, test_run_ids = (
            get_strict_training_run_manifest(scen_name, raw_run_ids)
        )
        manifest_run_ids = train_run_ids | test_run_ids
        if raw_run_ids != manifest_run_ids:
            raise RuntimeError(
                f"RAW/SPLIT RUN MISMATCH for {scen_name}: "
                f"missing-in-manifest={len(raw_run_ids - manifest_run_ids)}, "
                f"missing-in-raw={len(manifest_run_ids - raw_run_ids)}. "
                "Do NOT create another split or continue."
            )
        before_rows = len(df_raw)
        df_raw = df_raw.loc[
            df_raw['run'].astype(str).isin(train_run_ids)
        ].copy()
        actual_training_runs = set(df_raw['run'].astype(str).unique())
        if actual_training_runs != train_run_ids:
            raise RuntimeError(f"Training-run retention failed: {scen_name}")
        if actual_training_runs.intersection(test_run_ids):
            raise RuntimeError(f"TEST runs entered selection: {scen_name}")
        saved_id_files = save_exact_run_id_manifests(
            scen_name, manifest, train_run_ids, test_run_ids
        )
        report = {
            'scenario': scen_name,
            'benchmark_split_path': str(split_path),
            'explicit_saved_run_id_files': saved_id_files,
            'all_runs': len(manifest_run_ids),
            'train_runs': len(train_run_ids),
            'test_runs': len(test_run_ids),
            'train_rows_long_form': len(df_raw),
            'all_rows_long_form': before_rows,
            'train_ids_sha256': _run_hash(train_run_ids),
            'test_ids_sha256': _run_hash(test_run_ids),
            'train_test_intersection': 0,
            'test_used_in_selection': False,
        }
        HOLDOUT_AUDITS.append(report)
        print('\n' + '='*90)
        print('STRICT TRAIN-ONLY SELECTION —', scen_name)
        print('Input 70/30 run-split manifest:', split_path)
        print('Run counts: TRAIN=', len(train_run_ids),
              '| TEST=', len(test_run_ids))
        print('Final TEST runs excluded from selection: YES')
        print('Rows loaded / rows retained:', before_rows, '/', len(df_raw))
        print('='*90)

    except Exception as e:

        raise RuntimeError(
            f"Cannot complete strict training-only selection for "
            f"{scen_name}: {e}"
        ) from e


    # ========================================================================
    # 6.2 EXTRACT HOST IDENTIFIER
    # ========================================================================

    df_raw['host_id_full'] = (
        df_raw['module']
        .str.extract(r'(host\[\d+\])')
    )

    df_raw = df_raw.dropna(
        subset=['host_id_full']
    )

    df_raw['value'] = pd.to_numeric(
        df_raw['value'],
        errors='coerce'
    )


    # ========================================================================
    # 6.3 NODE-LEVEL MATRIX
    # ========================================================================

    df = (
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
        .fillna(0)
    )

    # Audit only: physical columns available in this development scenario.
    scenario_physical_schema[
        scen_name
    ] = set(
        df.columns
    )


    # ========================================================================
    # 6.4 LABEL ATTACKER NODES
    # ========================================================================

    attacker_ids = scenario['attacker']

    df['is_attacker'] = df.apply(
        lambda r:
            1
            if (
                "Attacker" in str(r['run'])
                and
                r['host_id_full'] in attacker_ids
            )
            else 0,
        axis=1
    )


    # ========================================================================
    # 6.5 KEEP RUN IDENTIFIER FOR GROUPED CV
    # ========================================================================
    #
    # run is NOT a classifier feature.
    # It is used only to guarantee run-level independence.
    # ========================================================================

    run_groups = (
        df['run']
        .astype(str)
        .copy()
    )


    # ========================================================================
    # RUN-LEVEL DATA AUDIT
    # ========================================================================

    unique_runs = (
        run_groups
        .nunique()
    )

    normal_runs = (
        df.loc[
            ~df['run']
            .astype(str)
            .str.contains(
                'Attacker',
                case=False,
                na=False
            ),
            'run'
        ]
        .nunique()
    )

    attack_runs = (
        df.loc[
            df['run']
            .astype(str)
            .str.contains(
                'Attacker',
                case=False,
                na=False
            ),
            'run'
        ]
        .nunique()
    )


    print(
        "\nRUN-LEVEL AUDIT"
    )

    print(
        "-" * 80
    )

    print(
        f"Unique simulation runs : "
        f"{unique_runs}"
    )

    print(
        f"Normal runs            : "
        f"{normal_runs}"
    )

    print(
        f"Attack runs            : "
        f"{attack_runs}"
    )

    print(
        f"Node-level rows        : "
        f"{len(df)}"
    )

    print(
        "-" * 80
    )


    # ========================================================================
    # STEP 0: RAW FEATURES
    # ========================================================================

    raw_features = [

        c

        for c in df.select_dtypes(
            include=['number']
        ).columns

        if c != 'is_attacker'

    ]


    print(
        f"\n[STEP 0] Original raw features "
        f"(total: {len(raw_features)}):"
    )

    print(
        raw_features
    )


    X_scen = (
        df[
            raw_features
        ]
        .copy()
    )

    y_scen = (
        df[
            'is_attacker'
        ]
        .copy()
    )


    # ========================================================================
    # STEP 1: REMOVE PREDEFINED DATA-LEAKAGE FEATURES
    # ========================================================================

    removed_leakage = [

        lf

        for lf in leakage_features

        if lf in X_scen.columns

    ]


    if removed_leakage:

        X_scen = (
            X_scen
            .drop(
                columns=removed_leakage
            )
        )


    features_after_leakage = (
        list(
            X_scen.columns
        )
    )


    scenario_leakage_clean_schemas[scen_name] = set(features_after_leakage)
    scenario_fold_selected_mi[scen_name] = {}
    scenario_fold_ranked[scen_name] = {}

    print(
        f"\n[STEP 1] After predefined leakage-feature removal "
        f"(remaining: {len(features_after_leakage)}):"
    )

    print(
        f"Removed ({len(removed_leakage)}): "
        f"{removed_leakage}"
    )

    print(
        f"Remaining "
        f"({len(features_after_leakage)}): "
        f"{features_after_leakage}"
    )


    # ========================================================================
    # STEP 2: ZERO-VARIANCE AND CORRELATION FILTERS FITTED IN EACH INNER TRAINING FOLD
    # ========================================================================
    #
    # IMPORTANT LEAKAGE-CONTROL NOTE:
    # Both filters are fitted independently on each INNER-TRAINING partition.
    # Target-correlation tie breaking uses INNER-TRAINING labels only; neither
    # INNER-VALIDATION nor OUTER-HOLDOUT data are used for fitting either filter.
    #
    # Outer-training data provide the input universe, not a pre-fitted filter.
    # ========================================================================

    # Preserve the scenario-size weighting used later in the global
    # meta-synthesis. Inner-fold filtering does not change this row count.
    scenario_actual_samples[
        scen_name
    ] = X_scen.shape[0]

    # Audit-only registries. They do not alter feature selection.
    zero_var_retained_sets = []
    zero_var_dropped_set = set()
    zero_var_feature_counts = []


    # ========================================================================
    # STEP 3: GROUPED NESTED FEATURE-SELECTION PIPELINE
    # ========================================================================

    scenario_local_mi_means = {}

    collinear_dropped_set = set()

    # AUDIT ONLY: retain the actual post-correlation feature count
    # for each outer fold. This does not alter feature selection.
    post_corr_feature_counts = []


    # ========================================================================
    # OUTER CV — GROUPED BY SIMULATION RUN
    # ========================================================================

    outer_cv = grouped_stratified_run_split(
        run_groups,
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    for fold_idx, (
        outer_train_idx,
        outer_test_idx,
        outer_train_runs,
        outer_test_runs
    ) in enumerate(
        outer_cv,
        1
    ):


        # --------------------------------------------------------------------
        # AUDIT: TRAIN AND OUTER HOLDOUT MUST HAVE ZERO RUN OVERLAP
        # --------------------------------------------------------------------

        outer_overlap = (
            outer_train_runs
            .intersection(
                outer_test_runs
            )
        )


        print(
            "\n" + "-" * 100
        )

        print(
            f"OUTER FOLD {fold_idx}/5"
        )

        print(
            f"Training runs : "
            f"{len(outer_train_runs)}"
        )

        print(
            f"Holdout runs  : "
            f"{len(outer_test_runs)}"
        )

        print(
            f"Run overlap   : "
            f"{len(outer_overlap)}"
        )


        if len(outer_overlap) != 0:

            raise RuntimeError(
                "OUTER RUN-LEVEL LEAKAGE DETECTED."
            )


        print(
            "Outer grouped split: PASS"
        )


        # --------------------------------------------------------------------
        # OUTER TRAINING DATA
        # --------------------------------------------------------------------

        X_outer_train = (
            X_scen
            .iloc[
                outer_train_idx
            ]
            .copy()
        )

        y_outer_train = (
            y_scen
            .iloc[
                outer_train_idx
            ]
            .copy()
        )

        # Run groups corresponding exactly to outer-training rows.
        run_outer_train = (
            run_groups
            .iloc[
                outer_train_idx
            ]
            .copy()
        )


        # ====================================================================
        # STRICT INNER-FOLD FEATURE FILTERING
        # ====================================================================
        # IMPORTANT: neither feature filtering nor target-correlation tie
        # breaking can inspect labels of the current INNER-VALIDATION fold.
        # Start from all predefined leakage-cleaned outer-training features.
        # A feature absent after filtering in an inner fold receives no vote,
        # and a zero importance contribution in that fold. This makes the
        # 5-fold stability denominator and global MI score well-defined.

        feature_universe = list(X_outer_train.columns)
        voting_stability_freq = {f: 0 for f in feature_universe}
        perm_accum = {f: [] for f in feature_universe}
        mi_accum = {f: [] for f in feature_universe}
        shap_accum = {f: [] for f in feature_universe}
        inner_zero_sets = []
        inner_corr_sets = []
        inner_filter_audit_rows = []

        inner_cv = grouped_stratified_run_split(
            run_outer_train, n_splits=5, shuffle=True, random_state=42
        )

        for inner_id, (inner_train_idx, inner_val_idx,
                       inner_train_runs, inner_val_runs) in enumerate(inner_cv, 1):
            if inner_train_runs.intersection(inner_val_runs):
                raise RuntimeError("INNER RUN-LEVEL LEAKAGE DETECTED")

            X_in_train_raw = X_outer_train.iloc[inner_train_idx].copy()
            y_in_train = y_outer_train.iloc[inner_train_idx].copy()
            X_in_val_raw = X_outer_train.iloc[inner_val_idx].copy()
            y_in_val = y_outer_train.iloc[inner_val_idx].copy()

            # FIT BOTH FILTERS ON INNER TRAINING ONLY.
            var_filter = VarianceThreshold(threshold=0.0)
            var_filter.fit(X_in_train_raw)
            after_var = X_in_train_raw.columns[var_filter.get_support()].tolist()
            if not after_var:
                raise RuntimeError("No feature survived inner zero-variance filter")

            X_in_train_zv = X_in_train_raw[after_var]
            corr_matrix = X_in_train_zv.corr().abs()
            target_corr = X_in_train_zv.apply(
                lambda col: col.corr(y_in_train)
            ).abs().fillna(0.0)
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            auto_drop = set()
            for col in upper_tri.columns:
                for feature in upper_tri.index[upper_tri[col] >= 0.99].tolist():
                    if target_corr[col] >= target_corr[feature]:
                        auto_drop.add(feature)
                    else:
                        auto_drop.add(col)

            retained_inner = [f for f in after_var if f not in auto_drop]
            if not retained_inner:
                raise RuntimeError("No feature survived inner correlation filter")

            # Apply the FITTED feature names without refitting on validation.
            X_in_train = X_in_train_raw[retained_inner].copy()
            X_in_val = X_in_val_raw[retained_inner].copy()
            inner_zero_sets.append(set(after_var))
            inner_corr_sets.append(set(retained_inner))
            inner_filter_audit_rows.append({
                'Scenario': scen_name, 'Outer_Fold': fold_idx,
                'Inner_Fold': inner_id,
                'Inner_Training_Runs': len(inner_train_runs),
                'Inner_Validation_Runs': len(inner_val_runs),
                'After_ZeroVariance': len(after_var),
                'After_Correlation': len(retained_inner),
                'Retained_Features': '|'.join(retained_inner),
                'Label_Scope': 'INNER_TRAIN_ONLY',
            })

            pos_idx = y_in_train[y_in_train == 1].index
            neg_idx = y_in_train[y_in_train == 0].sample(
                n=len(pos_idx), random_state=42
            ).index
            balanced_idx = pos_idx.union(neg_idx)
            X_in_train_bal = X_in_train.loc[balanced_idx]
            y_in_train_bal = y_in_train.loc[balanced_idx]

            in_model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            in_model.fit(X_in_train_bal, y_in_train_bal)
            perm_res = permutation_importance(
                in_model, X_in_val, y_in_val, scoring='f1',
                n_repeats=2, random_state=42, n_jobs=-1
            )
            fold_pfi = dict(zip(
                retained_inner, np.maximum(0, perm_res.importances_mean)
            ))
            fold_mi_scores = mutual_info_classif(
                X_in_train_bal, y_in_train_bal, random_state=42
            )
            fold_mi = dict(zip(retained_inner, [
                float(z) if z >= 0.0400 else 0.0 for z in fold_mi_scores
            ]))
            explainer = shap.TreeExplainer(in_model)
            shap_values = explainer.shap_values(X_in_train_bal)
            if isinstance(shap_values, list):
                shap_array = np.asarray(shap_values[1])
            elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
                shap_array = np.asarray(shap_values[:, :, 1])
            else:
                shap_array = np.asarray(shap_values)
            fold_shap = dict(zip(retained_inner,
                np.ravel(np.mean(np.abs(shap_array), axis=0))
            ))

            # Features absent after filtering in an inner fold contribute zero evidence
            # and receive no consensus vote in that fold.
            for f in feature_universe:
                perm_accum[f].append(float(fold_pfi.get(f, 0.0)))
                mi_accum[f].append(float(fold_mi.get(f, 0.0)))
                shap_accum[f].append(float(fold_shap.get(f, 0.0)))

            k = min(max(10, int(len(retained_inner) * 0.40)), len(retained_inner))
            perm_selected = set(sorted(retained_inner,
                key=lambda f: (-float(fold_pfi[f]), f))[:k])
            mi_selected = set(sorted(retained_inner,
                key=lambda f: (-float(fold_mi[f]), f))[:k])
            shap_selected = set(sorted(retained_inner,
                key=lambda f: (-float(fold_shap[f]), f))[:k])
            for f in retained_inner:
                if (int(f in perm_selected) + int(f in mi_selected)
                        + int(f in shap_selected)) >= 2:
                    voting_stability_freq[f] += 1

            print(f"Inner {inner_id}: zero-var={len(after_var)}, "
                  f"correlation={len(retained_inner)}, "
                  f"run-overlap=0 [PASS]")

        # Diagnostic count per OUTER fold: union of features surviving the inner filters.
        # This is a union of inner-fold results, not a separate fitted filter.
        retained_after_var_fold = sorted(set.union(*inner_zero_sets))
        retained_after_corr_fold = sorted(set.union(*inner_corr_sets))
        zero_variance_dropped_fold = sorted(
            set(feature_universe) - set(retained_after_var_fold)
        )
        zero_var_retained_sets.append(set(retained_after_var_fold))
        zero_var_dropped_set.update(zero_variance_dropped_fold)
        zero_var_feature_counts.append(len(retained_after_var_fold))
        collinear_dropped_set.update(
            set(retained_after_var_fold) - set(retained_after_corr_fold)
        )
        post_corr_feature_counts.append(len(retained_after_corr_fold))
        X_outer_train = X_outer_train[retained_after_corr_fold].copy()

        inner_stability_df = pd.DataFrame({
            'Feature': retained_after_corr_fold,
            'Normalized_Stability': [
                voting_stability_freq[f] / 5.0 for f in retained_after_corr_fold
            ],
            'Mean_MI': [np.mean(mi_accum[f]) for f in retained_after_corr_fold],
            'Mean_SHAP': [np.mean(shap_accum[f]) for f in retained_after_corr_fold],
        })
        stable_pool_df = inner_stability_df.loc[
            inner_stability_df['Normalized_Stability'] >= 0.8
        ].copy()
        if stable_pool_df.empty:
            raise RuntimeError("No feature survived 4-of-5 inner-fold consensus")
        stable_pool_df = stable_pool_df.sort_values('Mean_MI', ascending=False)
        stable_pool_df['InterimScore'] = normalize(
            stable_pool_df['Mean_SHAP'].to_numpy()
        )
        final_ranked_inner_df = stable_pool_df.sort_values(
            'InterimScore', ascending=False
        ).reset_index(drop=True)
        selected_features_at_fold = final_ranked_inner_df['Feature'].tolist()
        scenario_fold_selected_mi[scen_name][fold_idx] = {
            f: float(np.mean(mi_accum[f])) for f in selected_features_at_fold
        }
        scenario_fold_ranked[scen_name][fold_idx] = list(selected_features_at_fold)
        scenario_fold_diagnostics.append({
            'Scenario': scen_name, 'Outer_Fold': fold_idx,
            'After_ZeroVariance': len(retained_after_var_fold),
            'After_Correlation': len(retained_after_corr_fold),
            'Stable_Pool': len(selected_features_at_fold),
            'Local_Features': '|'.join(selected_features_at_fold),
            'Inner_Fit_Independent': True,
        })
        for f in selected_features_at_fold:
            scenario_local_mi_means.setdefault(f, []).append(
                float(np.mean(mi_accum[f]))
            )
        # Save evidence of each inner fold's fitted selection scope.
        inner_feature_filter_audits.extend(inner_filter_audit_rows)

    # ========================================================================
    # FOLD-SPECIFIC ZERO-VARIANCE SUMMARY (UNION OF INNER FITS)
    # ========================================================================

    zero_var_feature_counts_np = np.asarray(
        zero_var_feature_counts,
        dtype=int
    )

    if len(zero_var_feature_counts_np) != 5:
        raise RuntimeError(
            f"Expected 5 outer-fold zero-variance counts for {scen_name}, "
            f"found {len(zero_var_feature_counts_np)}."
        )

    zero_variance_dropped = sorted(
        zero_var_dropped_set
    )

    # Intersection across the five OUTER folds of the UNION of their retained
    # INNER-fold zero-variance feature sets. AUDIT ONLY; not a fitted filter.
    retained_after_var = sorted(
        set.intersection(
            *zero_var_retained_sets
        )
    ) if zero_var_retained_sets else []

    zero_var_sets_identical = (
        all(
            s == zero_var_retained_sets[0]
            for s in zero_var_retained_sets[1:]
        )
        if zero_var_retained_sets
        else True
    )

    print(
        "\n" + "=" * 100
    )
    print(
        "TRAINING-ONLY ZERO-VARIANCE SUMMARY"
    )
    print(
        "-" * 100
    )
    print(
        "Union of inner-fit retained features per outer fold : "
        + ", ".join(
            map(str, zero_var_feature_counts_np.tolist())
        )
    )
    print(
        f"Retained sets identical across folds      : "
        f"{'YES' if zero_var_sets_identical else 'NO'}"
    )
    print(
        f"In union of inner fits in all 5 outer folds : "
        f"{len(retained_after_var)}"
    )
    print(
        f"Features absent from all inner fits in some outer fold : "
        f"{zero_variance_dropped}"
    )
    print(
        "=" * 100
    )


    # ========================================================================
    # AUDIT ONLY: FOLD-SPECIFIC CORRELATION FILTERING SUMMARY
    # ========================================================================

    post_corr_feature_counts_np = np.asarray(
        post_corr_feature_counts,
        dtype=int
    )

    if len(post_corr_feature_counts_np) != 5:
        raise RuntimeError(
            f"Expected 5 outer-fold post-correlation counts for {scen_name}, "
            f"found {len(post_corr_feature_counts_np)}."
        )

    post_corr_mean = post_corr_feature_counts_np.mean()
    post_corr_std = post_corr_feature_counts_np.std(ddof=1)
    post_corr_min = post_corr_feature_counts_np.min()
    post_corr_max = post_corr_feature_counts_np.max()

    print(
        "\n" + "=" * 100
    )
    print(
        "FOLD-SPECIFIC CORRELATION FILTERING SUMMARY"
    )
    print(
        "-" * 100
    )
    print(
        "Union of inner-correlation retained features per outer fold : "
        + ", ".join(
            map(str, post_corr_feature_counts_np.tolist())
        )
    )
    print(
        f"Mean ± SD                        : "
        f"{post_corr_mean:.2f} ± {post_corr_std:.2f}"
    )
    print(
        f"Range                            : "
        f"{post_corr_min}–{post_corr_max}"
    )
    print(
        "=" * 100
    )


    # ========================================================================
    # REQUIREMENT:
    # FEATURE MUST HAVE VALUES FROM ALL FIVE OUTER FOLDS
    # ========================================================================

    scenario_local_mi_registry[
        scen_name
    ] = {

        f:
            np.mean(
                ml
            )

        for f, ml
        in scenario_local_mi_means.items()

        if len(ml) == 5

    }


    # ========================================================================
    # COLLINEARITY AUDIT SUMMARY (NOT A GLOBAL CORRELATION FILTER FIT)
    # ========================================================================

    collinear_dropped_list = (
        list(
            collinear_dropped_set
        )
    )


    features_after_collinear = [

        f

        for f in retained_after_var

        if f not in
        collinear_dropped_list

    ]


    print(
        f"\n[STEP 3] After Collinearity "
        f"Filter >= 0.99:"
    )


    print(
        f"Features observed as "
        f"collinearity-dropped "
        f"({len(collinear_dropped_list)}): "
        f"{collinear_dropped_list}"
    )


    print(
        f"Features remaining "
        f"({len(features_after_collinear)}): "
        f"{features_after_collinear}"
    )


    # ========================================================================
    # SCENARIO AUDIT
    # ========================================================================

    all_dropped_scenario = list(

        set(

            removed_leakage

            + zero_variance_dropped

            + collinear_dropped_list

        )

    )


    scenario_audit_trail[
        scen_name
    ] = {

        'total_raw':
            len(
                raw_features
            ),

        'retained_count':
            len(
                scenario_local_mi_registry[
                    scen_name
                ]
            ),

        'dropped_count':
            len(
                all_dropped_scenario
            ),

        'dropped_list':
            all_dropped_scenario,

        'retained_ranked':

            sorted(

                scenario_local_mi_registry[
                    scen_name
                ].items(),

                key=lambda x:
                    x[1],

                reverse=True

            ),

        'unique_runs':
            unique_runs,

        'normal_runs':
            normal_runs,

        'attack_runs':
            attack_runs
    }


# ============================================================================
# FOLD-SPECIFIC GLOBAL TOP-6 FEATURE SYNTHESIS
# ============================================================================
# Fold-level candidates are restricted to the shared leakage-cleaned
# feature schema of X7, X25 and X50 before weighted-MI ranking.
# The article-level 5-of-5 registry is computed separately below.

COMMON_EVAL_SCHEMA = set.intersection(*[
    scenario_leakage_clean_schemas[sc['name']]
    for sc in scenario_files
])

if len(COMMON_EVAL_SCHEMA) < FINAL_FEATURE_BUDGET:
    raise RuntimeError('The common leakage-cleaned schema has fewer than six features.')

weighted_rows = sum(scenario_actual_samples[sc['name']] for sc in scenario_files)
scenario_weights = {
    sc['name']: scenario_actual_samples[sc['name']] / weighted_rows
    for sc in scenario_files
}

outer_fold_global_rows = []
outer_fold_top6_rows = []
outer_fold_ranked = {}

for outer_fold in range(1, N_OUTER_FOLDS + 1):
    fold_candidate_mi = {
        name: scenario_fold_selected_mi[name][outer_fold]
        for name in scenario_fold_selected_mi
    }
    all_candidates = set().union(*(set(v) for v in fold_candidate_mi.values()))
    all_candidates.intersection_update(COMMON_EVAL_SCHEMA)
    if len(all_candidates) < FINAL_FEATURE_BUDGET:
        raise RuntimeError(f'Outer fold {outer_fold}: fewer than six globally eligible candidates.')

    rows = []
    for feat in all_candidates:
        row = {'Fold': outer_fold, 'Feature': feat}
        global_score = 0.0
        occurrence = 0
        for sc in scenario_files:
            name = sc['name']
            local_score = fold_candidate_mi[name].get(feat, 0.0)
            present = feat in fold_candidate_mi[name]
            occurrence += int(present)
            row[f'Present_{name}'] = int(present)
            row[f'MI_{name}'] = local_score if present else np.nan
            global_score += scenario_weights[name] * local_score
        row['Scenario_Occurrence'] = occurrence
        row['Global_MI'] = global_score
        rows.append(row)

    ranked = pd.DataFrame(rows).sort_values(by='Global_MI', ascending=False).reset_index(drop=True)
    ranked['Global_Rank'] = ranked.index + 1
    ranked['Final_Top6'] = (ranked.index < FINAL_FEATURE_BUDGET).astype(int)
    selected = ranked['Feature'].head(FINAL_FEATURE_BUDGET).tolist()
    outer_fold_ranked[outer_fold] = selected
    outer_fold_global_rows.extend(ranked.to_dict('records'))
    for rank, feat in enumerate(selected, 1):
        outer_fold_top6_rows.append({
            'Fold': outer_fold, 'Rank': rank, 'Feature': feat,
            'Global_MI': float(ranked.loc[rank-1, 'Global_MI']),
        })
    print(f'OUTER FOLD {outer_fold} GLOBAL TOP-6 (COMMON FEATURE SCHEMA): {selected}')

pd.DataFrame(outer_fold_global_rows).to_csv(OUTPUT_DIR / '01_outer_fold_global_MI_all_candidates.csv', index=False)
pd.DataFrame(outer_fold_top6_rows).to_csv(OUTPUT_DIR / '02_outer_fold_global_top6.csv', index=False)
pd.DataFrame(scenario_fold_diagnostics).to_csv(OUTPUT_DIR / '03_scenario_fold_stable_pools.csv', index=False)
pd.DataFrame(inner_feature_filter_audits).to_csv(OUTPUT_DIR / '04_inner_training_only_filter_audit.csv', index=False)

# Article-level 5-of-5 selection is computed separately below.

# ============================================================================
# 7. PRINT INDIVIDUAL SCENARIO AUDIT MATRICES
# ============================================================================

print(
    "\n" + "=" * 110
)

print(
    "INDIVIDUAL SCENARIO AUDIT MATRICES "
    "(GROUPED BY SIMULATION RUN)"
)

print(
    "=" * 110
)


for scenario in scenario_files:

    scen_name = scenario['name']


    if (
        scen_name
        not in scenario_audit_trail
    ):

        continue


    audit = (
        scenario_audit_trail[
            scen_name
        ]
    )


    print(
        f"\n>>> SCENARIO: {scen_name}"
    )


    print(
        f"Unique runs: "
        f"{audit['unique_runs']} "
        f"| Normal runs: "
        f"{audit['normal_runs']} "
        f"| Attack runs: "
        f"{audit['attack_runs']}"
    )


    print(
        f"Raw Features: "
        f"{audit['total_raw']} "
        f"| Five-fold registry: "
        f"{audit['retained_count']} "
        f"| Filtered-out union: "
        f"{audit['dropped_count']}"
    )


    print(
        "-" * 110
    )


    retained_rows = [

        {

            'Rank':
                str(
                    idx + 1
                ),

            'Feature':
                f,

            'Local MI Score':
                f"{score:.4f}",

            'Status':
                'Retained (✓)'

        }

        for idx, (
            f,
            score
        )

        in enumerate(
            audit[
                'retained_ranked'
            ]
        )

    ]


    df_retained = (
        pd.DataFrame(
            retained_rows
        )
    )


    dropped_rows = [

        {

            'Rank':
                'N/A',

            'Feature':
                f,

            'Local MI Score':
                'N/A',

            'Status':
                'Dropped (✕)'

        }

        for f
        in audit[
            'dropped_list'
        ]

    ]


    df_dropped = (
        pd.DataFrame(
            dropped_rows
        )
    )


    df_scenario_full = (
        pd.concat(
            [
                df_retained,
                df_dropped
            ],
            ignore_index=True
        )
    )


    print(
        df_scenario_full
        .to_string(
            index=False,
            justify='left'
        )
    )


    print(
        "=" * 110
    )


# ============================================================================
# 8. GLOBAL META-SYNTHESIS
# ============================================================================
#
# IMPORTANT:
# Scenario weighting rule:
#
#     scenario node-row count / total node-row count
#
# This scenario-size weighting rule is part of the final methodology.
# ============================================================================

print(
    "\n" + "=" * 110
)

print(
    "STAGE 4: CROSS-SCENARIO GLOBAL SYNTHESIS MATRIX "
    "(GLOBAL CUMULATIVE MI)"
)

print(
    "DEVELOPMENT SCENARIOS = 7 + 25 + 50 NODES"
)

print(
    "75-NODE SCENARIO = EXCLUDED"
)

print(
    "=" * 110
)


global_total_samples = sum(
    scenario_actual_samples.values()
)


global_cumulative_ig_registry = {}


all_candidate_features = set()


for winners in (
    scenario_local_mi_registry.values()
):

    all_candidate_features.update(
        winners.keys()
    )


# Final-methodology safeguard:
# at least six candidates must remain across the scenarios' five-fold registries.
if len(all_candidate_features) < 6:

    raise RuntimeError(
        "Fewer than six candidates survived the "
        "5/5 scenario aggregation."
    )


formatted_matrix_rows = []


for feat in (
    all_candidate_features
):

    row_data = {
        'Feature': feat
    }


    cum_score = 0.0


    for scenario in (
        scenario_files
    ):

        scen_name = (
            scenario[
                'name'
            ]
        )


        local_winners = (
            scenario_local_mi_registry
            .get(
                scen_name,
                {}
            )
        )


        actual_domain_size = (
            scenario_actual_samples
            .get(
                scen_name,
                0
            )
        )


        dynamic_domain_weight = (

            actual_domain_size
            /
            global_total_samples

            if global_total_samples > 0

            else 0

        )


        if feat in (
            local_winners
        ):

            row_data[
                scen_name
            ] = '✓'


            cum_score += (

                local_winners[
                    feat
                ]

                *

                dynamic_domain_weight

            )


        else:

            row_data[
                scen_name
            ] = '✕'


    row_data[
        'Global Score'
    ] = cum_score


    global_cumulative_ig_registry[
        feat
    ] = cum_score


    formatted_matrix_rows.append(
        row_data
    )


# ============================================================================
# 9. GLOBAL MATRIX
# ============================================================================

matrix_df = (
    pd.DataFrame(
        formatted_matrix_rows
    )
)


if matrix_df.empty:

    raise RuntimeError(
        "No feature survived the grouped "
        "feature-selection procedure."
    )


matrix_df = (

    matrix_df

    .sort_values(
        by='Global Score',
        ascending=False
    )

    .reset_index(
        drop=True
    )

)


# ============================================================================
# 10. FINAL TOP-K = 6
# ============================================================================

TOP_K_SELECTED = 6


matrix_df[
    'Final'
] = [

    '✓'
    if i < TOP_K_SELECTED
    else '✕'

    for i in range(
        len(
            matrix_df
        )
    )

]


matrix_df[
    'Global Score'
] = (

    matrix_df[
        'Global Score'
    ]
    .map(
        lambda x:
            f"{x:.6f}"
    )

)


scen_cols = [

    scenario[
        'name'
    ]

    for scenario
    in scenario_files

]


ordered_cols = (

    ['Feature']

    + scen_cols

    + [
        'Global Score',
        'Final'
    ]

)


matrix_df = (
    matrix_df[
        ordered_cols
    ]
)


print(
    "\n"
    +
    matrix_df
    .to_string(
        index=False,
        justify='center'
    )
)


# ============================================================================
# 11. PUBLICATION MATRIX
# ============================================================================

print(
    "\n" + "=" * 110
)


print(
    "FINAL PUBLICATION MATRIX "
    "(GROUPED BY SIMULATION RUN)"
)


print(
    "-" * 110
)


pipeline_results = (

    matrix_df[
        [
            'Feature',
            'Global Score'
        ]
    ]
    .copy()

)


pipeline_results.columns = [

    'Feature_Name',
    'Cumulative_IG_Score'

]


pipeline_results[
    'Global_Rank'
] = (

    pipeline_results.index
    + 1

)


print(

    pipeline_results[
        [
            'Global_Rank',
            'Feature_Name',
            'Cumulative_IG_Score'
        ]
    ]
    .to_string(
        index=False
    )

)


print(
    "-" * 110
)


# ============================================================================
# 12. FINAL SIX FEATURES
# ============================================================================

final_consensus_publication_set = (

    matrix_df
    .head(6)[
        'Feature'
    ]
    .tolist()

)


# Final-methodology safeguard:
# the final common representation must exist unchanged in every
# development scenario.
for scenario in scenario_files:

    scen_name = scenario['name']

    missing_final_features = [

        feature

        for feature
        in final_consensus_publication_set

        if feature
        not in scenario_physical_schema[
            scen_name
        ]

    ]

    if missing_final_features:

        raise RuntimeError(
            f"Final Top-6 incompatible with {scen_name}: "
            f"{missing_final_features}"
        )


# ============================================================================
# ARTICLE-LEVEL FIVE-FOLD FEATURE REGISTRY AND AUDIT OUTPUTS
# ============================================================================
# The article-level synthesis does not apply a common-schema prefilter.
# The selected six features are checked against every development schema.
article_rows = []
for _, row in matrix_df.iterrows():
    item = {'Feature': row['Feature'], 'Global_Rank': len(article_rows) + 1}
    item['Global_MI'] = float(global_cumulative_ig_registry[row['Feature']])
    item['Final_Top6'] = int(row['Feature'] in final_consensus_publication_set)
    for sc in scenario_files:
        name = sc['name']
        item[f'Present_{name}'] = int(row['Feature'] in scenario_local_mi_registry[name])
        item[f'MI_{name}'] = scenario_local_mi_registry[name].get(row['Feature'], np.nan)
    article_rows.append(item)

pd.DataFrame(article_rows).to_csv(OUTPUT_DIR / '04_article_final_global_MI.csv', index=False)
registry_rows = [
    {'Scenario': sc['name'], 'Feature': f, 'Mean_MI_Across_5_Outer_Folds': val}
    for sc in scenario_files
    for f,val in scenario_local_mi_registry[sc['name']].items()
]
pd.DataFrame(registry_rows).to_csv(OUTPUT_DIR / '05_article_scenario_5of5_registry.csv', index=False)

if len(HOLDOUT_AUDITS) != 3:
    raise RuntimeError('Not all three development scenarios passed the TRAIN-only audit.')
with (OUTPUT_DIR / '00_strict_train70_split_audit.json').open('w', encoding='utf-8') as holdout_f:
    json.dump(HOLDOUT_AUDITS, holdout_f, indent=2, ensure_ascii=False)

with (OUTPUT_DIR / '06_selection_audit.json').open('w', encoding='utf-8') as audit_f:
    json.dump({
        'method': 'STABILITY_DRIVEN_FEATURE_SELECTION_TRAIN70',
        'benchmark_train70_only': True,
        'benchmark_test_runs_excluded': True,
        'original_split_audits': HOLDOUT_AUDITS,
        'development_scenarios': [sc['name'] for sc in scenario_files],
        'common_schema_size': len(COMMON_EVAL_SCHEMA),
        'outer_fold_global_top6': {str(k): v for k,v in outer_fold_ranked.items()},
        'article_final_top6': final_consensus_publication_set,
        'scenario_5of5_counts': {k:len(v) for k,v in scenario_local_mi_registry.items()},
        'weights': scenario_weights,
        'eval_predictions_computed': False,
    }, audit_f, indent=2, ensure_ascii=False)

print('\nLOCAL FEATURE-SELECTION OUTPUT DIRECTORY:', OUTPUT_DIR.resolve())
print('ARTICLE FINAL TOP-6:', final_consensus_publication_set)
print('EXACT FINAL_CLASSIFICATION_HOLDOUT_30 RUN-ID CSVs:')
for audit in HOLDOUT_AUDITS:
    print(' ', audit['scenario'], '->', audit['explicit_saved_run_id_files']['test30_ids_csv'])


print(
    "\n" + "=" * 110
)


print(
    "FINAL TOP-6 BEHAVIORAL FEATURES "
    "AFTER GROUPED-RUN FEATURE SELECTION"
)


print(
    "INPUT DOMAINS: 7 + 25 + 50 NODES"
)


print(
    "75-NODE DOMAIN: COMPLETELY EXCLUDED"
)


print(
    "=" * 110
)


for i, feature in enumerate(
    final_consensus_publication_set,
    1
):

    print(
        f"{i:02d}. {feature}"
    )


print(
    "-" * 110
)


# ============================================================================
# 13. FINAL FEATURE-SELECTION AUDIT
# ============================================================================

print(
    "\n" + "=" * 110
)

print(
    "FINAL FEATURE-SELECTION AUDIT"
)

print(
    "=" * 110
)

print(
    "75-node scenario used in feature selection : NO   [PASS]"
)

print(
    "Outer CV grouped by simulation run         : YES  [PASS]"
)

print(
    "Zero variance and Pearson fitted on inner training only: YES  [PASS]"
)

print(
    "Inner CV grouped by simulation run         : YES  [PASS]"
)


print(
    "Node ID used as predictive feature         : NO   [PASS]"
)

print(
    "Run ID used as predictive feature          : NO   [PASS]"
)

print(
    "Permutation importance n_repeats           : 2"
)

print(
    "MI threshold                               : 0.0400"
)

print(
    "Stability threshold                        : 0.80"
)

print(
    "Voting rule                                : 2 of 3"
)

print(
    "RF estimators                              : 100"
)

print(
    "Local selection rule                       : retain full stable pool"
)

print(
    "Global scenario weighting                  : node-row count"
)

print(
    "Candidates after 5/5 aggregation           : >= 6 [PASS]"
)

print(
    "Final Top-6 schema compatibility           : PASS"
)

print(
    "Final number of features                   : 6"
)

print(
    "=" * 110
)