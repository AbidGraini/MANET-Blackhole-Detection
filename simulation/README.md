# MANET Blackhole Simulation Sources

## Scope

OMNeT++ 6.1 and INET 4.5.4 simulation definitions for AODV Blackhole detection experiments, including the full packet-dropping baseline and the selective packet-dropping sensitivity analysis.

## Repository structure

- `networks/AODVNetwork.ned` — variable-size MANET topology.
- `inet_aodv/full_drop/` — AODV with forged RREP responses and full dropping of intercepted transit packets.
- `inet_aodv/selective_dropping/` — AODV with configurable packet-dropping probability.
- `configs/X7/`, `X15/`, `X25/`, `X50/`, `X75/`, `X100/` — network-size configuration profiles.
- `configs/BH7_selective_dropping/` — Drop100, Drop50 and Drop25 configurations for seven nodes.

## Running the simulations

Use one matching set of `Aodv.cc`, `Aodv.h`, and `Aodv.ned` at a time in the `inet/routing/aodv/` directory of the compatible INET source tree. Rebuild INET after changing the selected implementation. Run the full-drop dataset profiles with `full_drop`; use `selective_dropping` for the drop-probability experiments.

The simulation network is `AODVNetwork`, with its size controlled by the `numHosts` parameter. The seven-node selective-dropping configuration uses a dedicated random stream for the probabilistic forwarding decision (`**.aodv.rng-1 = 3`). Drop probabilities are 1.0, 0.5 and 0.25.

## Configuration provenance and validation

The six network-size INI files were consolidated from a multi-experiment configuration. Some network-size settings were reconstructed from the documented experimental parameters rather than recovered from a dedicated per-size INI. Verify these settings against the configurations and run records used to generate the datasets before claiming numerical reproducibility.

The NED and INI source files were checked for consistent network and module references. Source-code cleanup did not modify executable C++ or NED definitions. Compilation, execution and full numerical reproduction have not been verified as part of this static audit. Dataset extraction scripts, exact run partitions, model training code and machine-readable results are required for complete experimental reproduction.

## Licensing

INET AODV source files retain the original OpenSim Ltd. copyright and `LGPL-3.0-or-later` SPDX notices. Distribution must comply with the applicable upstream license.
