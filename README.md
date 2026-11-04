# MANET Blackhole Detection

Reproducibility resources for detecting AODV Blackhole attacks in mobile ad hoc networks (MANETs). The repository contains OMNeT++/INET simulation configurations, behavioral datasets, Python feature-selection and evaluation scripts, and experimental results.

## Repository contents

- [`simulation/`](simulation/) — OMNeT++/INET configurations and modified AODV implementation; see the [simulation guide](simulation/README.md).
- [`scripts/`](scripts/) — feature selection and evaluation programs.
- [`results/`](results/) — reported metrics and experimental outputs.
- [Raw datasets (`raw-data-v1`)](https://github.com/AbidGraini/MANET-Blackhole-Detection/releases/tag/raw-data-v1) — nine compressed datasets and checksums.
- [Node-level predictions (`predictions-v1`)](https://github.com/AbidGraini/MANET-Blackhole-Detection/releases/tag/predictions-v1) — prediction CSV files for the evaluation experiments.

## Experiments

| Experiment | Script |
|---|---|
| Stability-driven feature selection (7, 25, 50 nodes) | [`stability_driven_feature_selection.py`](scripts/stability_driven_feature_selection.py) |
| Intra-dataset: 7 nodes | [`intra_dataset_7node_matched_evaluation.py`](scripts/intra_dataset_7node_matched_evaluation.py) |
| Intra-dataset: 25 nodes | [`intra_dataset_25node_matched_evaluation.py`](scripts/intra_dataset_25node_matched_evaluation.py) |
| Intra-dataset: 50 nodes | [`intra_dataset_50node_matched_evaluation.py`](scripts/intra_dataset_50node_matched_evaluation.py) |
| Intra-dataset: 75 nodes | [`intra_dataset_75node_matched_evaluation.py`](scripts/intra_dataset_75node_matched_evaluation.py) |
| Intra-dataset: 100 nodes | [`intra_dataset_100node_matched_evaluation.py`](scripts/intra_dataset_100node_matched_evaluation.py) |
| Cross-scenario: 25 → 15 nodes | [`cross_scenario_25_to_15_evaluation.py`](scripts/cross_scenario_25_to_15_evaluation.py) |
| Cross-scenario: 100 → 75 nodes | [`cross_scenario_100_to_75_evaluation.py`](scripts/cross_scenario_100_to_75_evaluation.py) |
| Selective dropping: within-variant | [`selective_dropping_within_variant_7node_evaluation.py`](scripts/selective_dropping_within_variant_7node_evaluation.py) |
| Selective dropping: cross-behavior | [`selective_dropping_cross_behavior_7node_evaluation.py`](scripts/selective_dropping_cross_behavior_7node_evaluation.py) |

The proposed representation consists of six behavioral features: `numPacketsForwarded:count`, `packetSent:count`, `numRREPsSent:count`, `numRREQsSent:count`, `arpReplySent:count`, and `linkBroken:count`.

## Reproduce the experiments

**Requirements:** Python 3 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `shap`, and `joblib`. The simulation environment uses OMNeT++ 6.1 and INET 4.5.4.

```bash
python -m pip install numpy pandas scipy scikit-learn shap joblib
```

1. Download the [raw datasets](https://github.com/AbidGraini/MANET-Blackhole-Detection/releases/tag/raw-data-v1), verify their published checksums, and extract the CSV files into `data/` at the repository root. Alternatively, set `MANET_DATA_DIR` to the directory containing the extracted files.
2. Run commands from the repository root. Run feature selection first to generate or verify the X7/X25/X50 run-partition manifests:

   ```bash
   python scripts/stability_driven_feature_selection.py
   ```

3. Run the required evaluation script from the table above. X7/X25/X50 reuse the partitions exported by feature selection; X75/X100 create and then reuse their own partitions. Selective-dropping scripts create their run-grouped partitions; cross-scenario scripts train on the source scenario and evaluate on the separate target scenario. For example:

   ```bash
   python scripts/intra_dataset_75node_matched_evaluation.py
   python scripts/cross_scenario_100_to_75_evaluation.py
   ```

4. Compare the exported metrics and predictions with [`results/`](results/) and the [prediction release](https://github.com/AbidGraini/MANET-Blackhole-Detection/releases/tag/predictions-v1).

For execution from another directory, set `MANET_PROJECT_ROOT` to the repository root. Output locations can be configured using the environment variables defined near the beginning of each script. The cross-scenario scripts create compact intermediate datasets from their source and target raw CSVs.

## Evaluation protocol

- **Feature selection:** development scenarios of 7, 25, and 50 nodes; grouped nested validation and training-only feature selection. The final six features are fixed before evaluation on other network sizes.
- **Intra-dataset:** simulation-run-grouped, stratified 70/30 train/holdout split; full non-leakage, statistically filtered, and proposed six-feature representations; RF, DT, RBF-SVM, KNN, and MLP.
- **Cross-scenario:** training on the source network (25 → 15 or 100 → 75) and testing on the separate target network; source-only preprocessing and feature filtering, with run-level bootstrap intervals and KS analyses.
- **Selective packet dropping:** independent 100%, 50%, and 25% within-variant evaluations; cross-behavior training on 100% and 25% TRAIN70 runs and testing on 50% TEST30 runs, with shared normal controls counted once.
