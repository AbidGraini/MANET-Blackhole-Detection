# MANET Blackhole Detection

This repository contains simulation source files and Python scripts for **node-level detection of Blackhole attacks in mobile ad hoc networks (MANETs)**. The experiments use AODV in **OMNeT++ 6.1 / INET 4.5.4** and evaluate a compact behavioral representation with supervised machine-learning classifiers.

The released code covers stability-driven feature selection, within-dataset classification, cross-scenario evaluation, and selective packet-dropping experiments. It is an **offline experimental evaluation**, not a deployed intrusion-prevention system.

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
│   ├── selective_drop_within_variant_evaluation.py
│   └── selective_drop_mixed_rate_evaluation.py
└── simulation/
    ├── README.md
    ├── SHA256SUMS.txt
    ├── networks/AODVNetwork.ned
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

The `simulation/` folder includes AODV modifications for complete and probabilistic dropping, network definitions, and the corresponding configuration profiles. See [simulation/README.md](simulation/README.md) for setup details, source provenance, and validation limits. Use the matching `Aodv.cc`, `Aodv.h`, and `Aodv.ned` implementation together; rebuilding INET is necessary when switching implementations. The original INET licensing notices are retained in the source files.

## Behavioral representation

Feature selection uses the **7-, 25-, and 50-node** development datasets. The fixed six-feature representation evaluated in the classification scripts is:

- `numPacketsForwarded`
- `packetSent`
- `numRREPsSent`
- `numRREQsSent`
- `arpReplySent`
- `linkBroken`

The OMNeT++ scalar CSV exports use measurement names such as `numRREPsSent:count`; the scripts expect the exact column names defined in their source code.

## Evaluation scripts

| Script | Experiment |
| --- | --- |
| `stability_driven_feature_selection.py` | Training-only, nested run-grouped selection using mutual information (MI), permutation feature importance (PFI), TreeSHAP, consensus voting, stability filtering, and cross-scenario synthesis on the 7/25/50-node development data. |
| `intra_dataset_7node_matched_evaluation.py` | 7-node within-dataset matched classification. |
| `intra_dataset_25node_matched_evaluation.py` | 25-node within-dataset matched classification. |
| `intra_dataset_50node_matched_evaluation.py` | 50-node within-dataset matched classification. |
| `intra_dataset_75node_matched_evaluation.py` | 75-node within-dataset matched classification. |
| `intra_dataset_100node_matched_evaluation.py` | 100-node within-dataset matched classification. |
| `cross_scenario_25_to_15_evaluation.py` | Train on the 25-node source and test on the separate 15-node target. |
| `cross_scenario_100_to_75_evaluation.py` | Train on the 100-node source and test on the separate 75-node target. |
| `selective_drop_within_variant_evaluation.py` | Train/test separately at dropping probabilities of 100%, 50%, and 25% in the 7-node scenario. |
| `selective_drop_mixed_rate_evaluation.py` | Train using Drop100 and Drop25 TRAIN70; test at the untrained Drop50 TEST30 rate. |

The classification experiments use **Random Forest, Decision Tree, SVM, KNN, and MLP**, with settings specified in each script. The within-dataset benchmarks compare the full non-leakage, statistically filtered, and proposed six-feature representations.

## Evaluation protocol and leakage controls

- The **within-dataset** holdout split is **70% training / 30% testing by simulation run**, rather than by individual node record. The 7/25/50-node analyses reuse saved partitions; the 75/100-node evaluation scripts can generate their deterministic run-level partitions when missing.
- Feature selection for the proposed six features is performed using development training data only. Nested inner and outer validation folds are grouped by simulation run; final holdout records do not participate in selection.
- **Cross-scenario** tests fit models and scalers on source data only. Target data are reserved for evaluation; distribution shifts are examined using two-sample Kolmogorov–Smirnov statistics and run-level bootstrap procedures as implemented in the scripts.
- **Selective dropping** uses probabilities 1.00, 0.50, and 0.25 while retaining the forged-RREP attack strategy. In the mixed-rate experiment, normal-control records that are identical across the two training variants are counted once, and Drop50 is not used for fitting or tuning.

## Software and input data

The Python scripts use `numpy`, `pandas`, `scipy`, `scikit-learn`, `shap`, and `joblib`. Install the main dependencies in a suitable Python environment:

```bash
python -m pip install numpy pandas scipy scikit-learn shap joblib
```

To regenerate the simulation exports, use a compatible OMNeT++ 6.1 / INET 4.5.4 setup and the matching files under `simulation/`. The Python evaluations operate on the resulting scalar CSV files (including fields such as `run`, `module`, `name`, and `value`).

**Large raw simulation CSVs are not included in the source-code folders.** Supply the matching exports locally. Before running an evaluation script, inspect its input/output path constants: several scripts contain machine-specific Windows locations (for example, `F:/2026/RAWDATASET/` and `G:/FINAL_BEHAVIORAL_DATASETS/`). Changing those paths is a filesystem adjustment, not a change to the evaluation protocol.

The feature-selection program uses `MANET_DATA_DIR`, `MANET_SPLITS_DIR`, and `MANET_SELECTION_OUTPUT` environment variables, defaulting respectively to local `data/`, `splits/`, and `selection_outputs/` directories. It requires the original saved run-partition manifests at:

```text
splits/7NODE/7node_run_split.csv
splits/25NODE/25node_run_split.csv
splits/50NODE/50node_run_split.csv
```

These manifests must provide `run`, `run_condition` (`normal` or `attack`), and `partition` (`TRAIN` or `TEST`). **The feature-selection script does not regenerate missing saved partitions.** The corresponding 7/25/50-node classification scripts also validate frozen run identifiers and reference hashes; supply their required split exports and audit files as specified in those scripts.

## Suggested reproduction sequence

1. Prepare a compatible software environment, the raw scalar CSVs, and the required saved run partitions.
2. Run `scripts/stability_driven_feature_selection.py` on the 7/25/50-node development inputs.
3. Run the five `scripts/intra_dataset_*_matched_evaluation.py` programs with their corresponding data and run-split inputs.
4. Run both `scripts/cross_scenario_*_evaluation.py` programs using their source and target datasets.
5. Run `scripts/selective_drop_within_variant_evaluation.py`, then `scripts/selective_drop_mixed_rate_evaluation.py`, with the three selective-dropping exports.

The scripts generate experiment-specific metrics, predictions, feature records, and/or validation outputs under their configured output paths. They do not write to one common results folder.

## Reproducibility scope

The repository provides the **ten scripts listed above** and the `simulation/` source/configuration folder. Complete numerical reproduction also depends on the matching raw datasets, saved partition files, and compatible software versions. Their availability should not be inferred from the presence of the code. The published scripts **do not include one standalone implementation of every ablation configuration**; the ablation results therefore are not fully reproducible from these ten scripts alone.

When citing or using these resources, identify the repository commit, specific script, input scenario, and partition protocol. No dataset download link, publication DOI, or project-wide software license is asserted here without a verified corresponding resource.
