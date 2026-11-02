# MANET Blackhole Detection

Reproducible simulation and machine-learning workflows for studying Blackhole behavior in mobile ad hoc networks (MANETs) using AODV, OMNeT++/INET, and behavioral feature selection.

This repository contains simulation configuration/source files and Python scripts for feature selection, within-scenario classification, cross-scenario transfer, and selective packet-dropping experiments. The scripts are research implementations; reproducing numerical results requires the corresponding simulation-generated CSV datasets and a compatible software environment.

## Repository structure

```text
MANET-Blackhole-Detection/
├── README.md
├── scripts/
│   ├── stability_driven_feature_selection.py
│   ├── intra_dataset_7node_matched_evaluation.py
│   ├── intra_dataset_25node_matched_evaluation.py
│   ├── intra_dataset_50node_matched_evaluation.py
│   ├── intra_dataset_75node_matched_evaluation.py
│   ├── intra_dataset_100node_matched_evaluation.py
│   ├── cross_scenario_25_to_15_evaluation.py
│   ├── cross_scenario_100_to_75_evaluation.py
│   ├── selective_packet_dropping_7node_evaluation.py
│   └── selective_dropping_cross_behavior_7node_evaluation.py
└── simulation/
    ├── README.md
    ├── SHA256SUMS.txt
    ├── networks/
    │   └── AODVNetwork.ned
    ├── inet_aodv/
    │   ├── full_drop/
    │   └── selective_dropping/
    └── configs/
        ├── X7/
        ├── X15/
        ├── X25/
        ├── X50/
        ├── X75/
        ├── X100/
        └── BH7_selective_dropping/
```

The `simulation/` layout shown above follows the previously documented repository organization and has **not been independently rechecked against the live GitHub tree** during this README revision. For simulator setup, source provenance and validation limits, consult [simulation/README.md](simulation/README.md). Use the matching `Aodv.cc`, `Aodv.h` and `Aodv.ned` files together for the intended dropping behavior; switching modified INET implementations requires rebuilding INET. Keep the original upstream INET licensing notices.

## Research workflow

| Study | Script | Evaluation design |
|---|---|---|
| Stability-driven feature selection | `stability_driven_feature_selection.py` | Develop the fixed six-feature representation using X7, X25, and X50; group partitions by simulation run and fit selection stages on training data only. |
| Within-scenario X7 | `intra_dataset_7node_matched_evaluation.py` | Matched comparison of full non-leakage, statistical filtering, and proposed six-feature representations. |
| Within-scenario X25 | `intra_dataset_25node_matched_evaluation.py` | Same three-representation / five-classifier evaluation. |
| Within-scenario X50 | `intra_dataset_50node_matched_evaluation.py` | Same three-representation / five-classifier evaluation. |
| Within-scenario X75 | `intra_dataset_75node_matched_evaluation.py` | Run-grouped TRAIN70/HOLDOUT30; X75 is not used to derive the proposed global six features. |
| Within-scenario X100 | `intra_dataset_100node_matched_evaluation.py` | Run-grouped TRAIN70/HOLDOUT30; X100 is not used to derive the proposed global six features. |
| Cross-scenario 25 → 15 | `cross_scenario_25_to_15_evaluation.py` | Train on 25-node source, evaluate on 15-node target; includes compact-data preparation, grouped source-only statistical filtering, bootstrap CIs, and KS tests. |
| Cross-scenario 100 → 75 | `cross_scenario_100_to_75_evaluation.py` | Train on 100-node source, evaluate on 75-node target; compact data are prepared using chunked, SQLite-backed aggregation when rebuilding is needed. |
| Within-variant selective dropping | `selective_packet_dropping_7node_evaluation.py` | Independent 7-node evaluations for packet-dropping probabilities of **100%, 50%, and 25%**. |
| Cross-Behavior Evaluation under Different Packet-Dropping Probabilities (Section 4.4.4) | `selective_dropping_cross_behavior_7node_evaluation.py` | Mixed-rate training on **100% and 25%** dropping variants (TRAIN70), followed by evaluation at the untrained **50%** variant (TEST30). |

`Drop100`, `Drop50`, and `Drop25` are internal configuration identifiers corresponding to packet-dropping probabilities of **100%**, **50%**, and **25%**. They are not network sizes.

## Simulation scenarios

| Scenario | Nodes | Configured Blackhole nodes | Intended use |
|---|---:|---|---|
| X7 | 7 | `host[5]` | Feature development and within-scenario evaluation |
| X15 | 15 | `host[8]` | 25 → 15 transfer target |
| X25 | 25 | `host[15]` | Feature development, within-scenario evaluation, and transfer source |
| X50 | 50 | `host[20]`, `host[35]` | Feature development and within-scenario evaluation |
| X75 | 75 | `host[20]`, `host[40]` | Within-scenario evaluation and 100 → 75 transfer target |
| X100 | 100 | `host[30]`, `host[60]`, `host[90]` | Within-scenario evaluation and 100 → 75 transfer source |

The 7-node selective-dropping experiments are separate variants of the 7-node attack configuration. Refer to the `simulation/` configuration files for the exact simulator parameters used in each study.

## Methods and leakage safeguards

- **Unit of partitioning:** simulation run, not individual node records.
- **Within-scenario evaluation:** stratified 70/30 run-grouped TRAIN/HOLDOUT splits, using deterministically generated and verified manifests for X7/X25/X50 and scenario-specific manifest handling for X75/X100.
- **Feature sets:** full non-leakage representation (56 behavioral features for the harmonized X25/X50/X75/X100 data), training-only statistical filtering, and the fixed proposed six features.
- **Statistical filtering:** zero-variance removal followed by Pearson correlation pruning (`|r| >= 0.99`), with label-dependent tie-breaking restricted to the respective training portion.
- **Models:** Random Forest (RF), Decision Tree (DT), support vector machine (SVM), k-nearest neighbors (KNN), and multilayer perceptron (MLP).
- **Scaling:** standardization fitted on training/source data only for the models that use scaling.
- **External transfer:** source-only feature fitting and model training; target labels are reserved for evaluation and dataset-integrity audits.
- **Uncertainty and shift assessment:** cross-scenario scripts compute run-level bootstrap confidence intervals and Kolmogorov–Smirnov (KS) statistics with complete-run bootstrap resampling; the implementations specify 2,000 replicates.

### Proposed six behavioral features

These are the original scalar feature-column names used by the Python scripts:

```text
numRREQsSent:count
numRREPsSent:count
packetSent:count
numPacketsForwarded:count
arpReplySent:count
linkBroken:count
```

## Python environment

The scripts use Python 3 and the following packages:

```bash
python -m pip install numpy pandas scipy scikit-learn shap joblib
```

IPython/Jupyter is optional for displaying some results interactively. For exact numerical reproducibility, record the Python and package versions used in the original experiments; installing the latest versions alone does not establish numerical equivalence.

## Raw simulation datasets (GitHub Releases)

The earlier repository README identifies **[Raw Simulation Datasets — Version 1 (`raw-data-v1`)](https://github.com/AbidGraini/MANET-Blackhole-Detection/releases/tag/raw-data-v1)** as the release containing nine losslessly compressed `.7z` archives and a `SHA256SUMS.txt` checksum manifest. Its documented coverage is the six node-count scenarios (7, 15, 25, 50, 75 and 100 nodes) plus the three 7-node selective packet-dropping variants (100%, 50% and 25%). Download the appropriate archive and extract its CSV files with 7-Zip before running the relevant script. Verify the archive checksums using the provided manifest.

**Verification scope:** this release URL and asset inventory are retained from the user-provided earlier README; the live Releases page could not be fetched during this revision. Confirm current availability and asset names on GitHub before relying on them.

The raw exports are scalar CSVs, typically including `run`, `module`, `name` and `value`. **Raw archives and run-partition manifests are different artifacts.** The feature-selection script **generates** the original X7/X25/X50 70/30 run-partition manifests deterministically from matching raw run IDs, checks their embedded SHA-256 fingerprints, and saves them under `splits/`. It also exports the frozen training and holdout run-ID lists into `selection_outputs/00_FIXED_70_30_RUN_IDS/`. These generated files are not separate prerequisites for a first run of feature selection. Archive-integrity verification is a separate, user-performed step; the feature-selection script itself validates raw run counts, deterministic TRAIN/TEST assignments, and the SHA-256 fingerprints of run identifiers before creating or reusing the manifests. Existing manifests are validated and reused, never silently replaced.

## SHA-256 verification: two distinct checks

**1. Downloaded-archive integrity (`SHA256SUMS.txt`).** The release checksum manifest applies to the downloaded `.7z` archives. Check the SHA-256 of each downloaded archive against its entry in the manifest **before extracting it**. On Windows PowerShell, for example:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'path/to/downloaded-archive.7z'
```

Compare the printed hash with the corresponding filename entry in the release's `SHA256SUMS.txt`. Replace the example path with the actual archive path and use the real asset filename. **The Python feature-selection script does not perform this `.7z` archive-checksum comparison.** Do not confuse the archive checksum with the run-split fingerprints.

**2. Frozen TRAIN70/HOLDOUT30 split identity (embedded reference hashes).** For X7, X25, and X50, `scripts/stability_driven_feature_selection.py` regenerates a deterministic, stratified run-level split from the raw CSV run identifiers (`test_size=0.30`, `random_state=42`). It calculates SHA-256 fingerprints of the *sorted run-ID strings* separately for TRAIN and TEST and compares them with the six reference hashes in `EXPECTED_RUN_SPLIT_HASHES`. The code also validates the expected counts of 600, 1,000, and 800 simulation runs, respectively. These hashes **are not checksums of the raw CSV contents**.

- **Match:** the script saves the new `splits/<scenario>/...run_split.csv` manifest, or verifies and reuses an existing identical manifest. Its output includes `original SHA-256 matched`.
- **Mismatch:** it stops with an error and does not save or overwrite an incompatible split. Check the input CSV, original simulation-run names, and whether the source data match the published scenarios; do **not** change the reference hashes merely to silence an error.
- **Scope:** the embedded split-hash checks apply to X7/X25/X50 in the feature-selection script. Other scripts may implement additional partition checks, but the six embedded reference hashes do not establish archive integrity, numerical equivalence across software versions, or full paper-wide reproduction.

**Verification status:** the presence and logic of these checks have been inspected in source code. A clean-room run using the downloaded release archives, followed by comparison with every reported table, is still required to confirm end-to-end reproduction.

## Dataset placement and portable paths

The Python scripts support environment-based paths. By default, `MANET_PROJECT_ROOT` is the current working directory and raw CSVs are read from its `data/` folder. Run commands from the repository root, or set `MANET_PROJECT_ROOT` explicitly.

Common settings:

| Environment variable | Purpose |
|---|---|
| `MANET_PROJECT_ROOT` | Base directory for default relative paths |
| `MANET_DATA_DIR` | Raw OMNeT++ scalar CSV datasets |
| `MANET_SPLITS_DIR` | Saved run-grouped partition manifests |
| `MANET_SELECTION_OUTPUT` | Feature-selection outputs, including frozen run-ID audits |
| `MANET_X7_OUTPUT` … `MANET_X100_OUTPUT` | Within-scenario result directories (for scenarios with corresponding scripts) |
| `MANET_CROSS_25_15_COMPACT` | Prepared 25/15 compact datasets |
| `MANET_CROSS_25_15_OUTPUT` | 25 → 15 evaluation outputs |
| `MANET_CROSS_100_75_COMPACT` | Prepared 100/75 compact datasets |
| `MANET_CROSS_100_75_OUTPUT` | 100 → 75 evaluation outputs |
| `MANET_SELECTIVE_DROP_OUTPUT` | Within-variant selective-dropping outputs |
| `MANET_SELECTIVE_CROSS_BEHAVIOR_OUTPUT` | Cross-behavior selective-dropping outputs |

On Windows PowerShell, for example:

```powershell
$env:MANET_PROJECT_ROOT = 'F:/JUPITERABID'
$env:MANET_DATA_DIR = 'F:/2026/RAWDATASET'
$env:MANET_SPLITS_DIR = 'F:/JUPITERABID/splits'
$env:MANET_SELECTION_OUTPUT = 'F:/JUPITERABID/selection_outputs'
```

The paths above are **examples**, not required locations. Output directories can be changed with the variables in the table. Input CSV filenames must match those expected by the scripts, or be adjusted deliberately to the user's dataset organization.

## Running the scripts

From the repository root, after making the required CSV inputs available:

```bash
python scripts/stability_driven_feature_selection.py
python scripts/intra_dataset_7node_matched_evaluation.py
python scripts/intra_dataset_25node_matched_evaluation.py
python scripts/intra_dataset_50node_matched_evaluation.py
python scripts/intra_dataset_75node_matched_evaluation.py
python scripts/intra_dataset_100node_matched_evaluation.py
python scripts/cross_scenario_25_to_15_evaluation.py
python scripts/cross_scenario_100_to_75_evaluation.py
python scripts/selective_packet_dropping_7node_evaluation.py
python scripts/selective_dropping_cross_behavior_7node_evaluation.py
```

**Recommended order for reproducing the development benchmarks:** first run `stability_driven_feature_selection.py` with the matching X7, X25 and X50 raw CSVs. It creates and verifies the run-grouped manifests in `splits/` and exports the exact frozen run-ID lists in `selection_outputs/00_FIXED_70_30_RUN_IDS/`. Then run the X7, X25 and X50 within-scenario scripts, which consume those saved files and validate the original partition identifiers/hashes. The 75- and 100-node within-scenario scripts independently create their deterministic run-grouped manifests if absent and reuse them afterward. The cross-scenario scripts contain compact-data preparation and external evaluation in the same script; the 100 → 75 preparation uses SQLite-backed streaming when rebuilding large raw datasets. The two selective-dropping scripts require their respective variant-specific raw CSVs.

**The listed commands are individual entry points, not one compulsory batch.** Choose the experiments you wish to reproduce. The feature-selection step is a prerequisite only when running the development benchmarks that depend on its frozen run-ID exports.

### X7/X25/X50 run-partition generation

The feature-selection script constructs the benchmark splits from *sorted unique raw simulation-run IDs*, using `train_test_split(test_size=0.30, random_state=42, stratify=run_condition)`. It compares both TRAIN and TEST run-ID SHA-256 fingerprints with the reference hashes embedded in the program **before writing** any new manifest. This run-ID verification is separate from checking downloaded `.7z` archives against `SHA256SUMS.txt`. If the raw dataset or run naming differs, it raises an error rather than saving an incompatible split. Existing manifests must match the regenerated assignments exactly.

Files generated or verified automatically include:

```text
splits/
├── 7NODE/7node_run_split.csv
├── 25NODE/25node_run_split.csv
└── 50NODE/50node_run_split.csv

selection_outputs/
└── 00_FIXED_70_30_RUN_IDS/
    └── ... exact scenario-specific TRAIN70 and HOLDOUT30 run-ID CSVs
```

The exact names of the additional run-ID exports are defined in `save_exact_run_id_manifests()` in the feature-selection script. Do not manually generate substitute splits; preserve the original CSV run names and the generated manifests for subsequent classifier evaluations.

## Interpretation of the evaluation protocols

**Within-scenario** means training and testing in the same node-count scenario using disjoint simulation runs. **Cross-scenario** means source-to-target transfer between different node counts (25 → 15 or 100 → 75), with source-trained models evaluated on a distinct target scenario. **Cross-behavior** means training on two packet-dropping probabilities (100% and 25%) and testing on a held-out third probability (50%) in the 7-node setting; this is not a change in node count.

The cross-behavior mixed training set includes normal-control profiles once rather than duplicating identical normal runs. The mixed training set has more Blackhole profiles than a single-variant baseline, so a performance gain by itself cannot isolate behavioral diversity from sample-count effects.

## Reproducibility notes

- Preserve the original raw CSV data, run IDs, selected-feature names, and the auto-generated, SHA-256-verified split manifests. X7/X25/X50 splits are created by the feature-selection script; they need not be separately downloaded on first execution.
- Do not use target datasets to select source features, fit scalers, or train external-transfer classifiers.
- Some scripts check expected run counts, attacker labels, feature schemas, and train/test disjointness and will stop if those checks fail.
- The scripts produce numerical outputs and protocol audits in configurable local output folders; the repository's code alone does not establish the article's reported metrics.
- No blanket claim is made that every script was rerun after its path/documentation cleanup. Reproducing a paper table requires the original datasets and recorded experimental environment.
- The ten published Python scripts do **not** constitute a separate executable implementation of every ablation configuration reported in the study; full ablation reproduction requires the corresponding additional implementation and inputs.
- When reporting results, identify the repository commit, script, input scenario, software versions and run-partition protocol.

## Citation and license

If you use this work, please cite the corresponding research article when its final bibliographic details are available. Licensing and third-party OMNeT++/INET dependencies should be checked against the repository's published license and upstream licenses; this README does not grant additional rights.
