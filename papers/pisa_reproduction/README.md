# PISA Reproduction

This folder contains a stricter reproduction-oriented implementation of:

*Permutation-Invariant Set Autoencoders with Fixed-Size Embeddings for Multi-Agent Learning*  
Ryan Kortvelesy, Steven Morad, Amanda Prorok  
AAMAS 2023

## Scope

The implementation focuses on the two experiments highlighted in the paper:

- `4.1 Random Set Reconstruction`
- `4.2 Multi-Agent Sensor Fusion`

The goal is to make the repository easier to run and closer to the paper protocol than the original concept reproduction.

## Important caveats

- The PISA-style training/evaluation loop now uses predicted cardinality at inference time by default.
- `GRU`, `DSPN`, and `TSPN` are implemented as lightweight, paper-inspired baselines inside this repository. They are not direct vendor drops of the authors' official codebases.
- `4.2` remains a synthetic communication experiment, not a full MARL task reproduction.
- `reproduction_report.pdf` is kept as a legacy comparison artifact from the earlier concept-reproduction version.

## Files

- `dataset.py`: random set sampling and synthetic multi-agent scene generation
- `model.py`: PISA model and duplicate detector
- `baselines.py`: GRU / DSPN / TSPN inspired baselines
- `reproduce.py`: experiment logic, training loops, metrics, and plots
- `train.py`: CLI entrypoint
- `results/`: generated metrics and figures

## Install

From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

## Run

From `papers/pisa_reproduction`:

```bash
python3 train.py random
python3 train.py fusion
python3 train.py ablation
python3 train.py full
```

You can override the output directory:

```bash
python3 train.py random --output-dir ./results_v2
```

## Output layout

- `results/random_reconstruction/`: 4.1 metrics, curves, and interpolation summaries
- `results/sensor_fusion/`: 4.2 metrics and rollout figures
- `results/ablation/`: internal PISA ablations
- `results/legacy/`: figures and metrics from the earlier concept reproduction, if retained

## Reproduction intent

This repository now separates three levels of confidence:

- **Paper-aligned**: random reconstruction protocol, cardinality-aware decode/eval loop, ablation structure
- **Paper-inspired**: baseline implementations and interpolation summary tooling
- **Synthetic approximation**: multi-agent sensor fusion environment and duplicate-filter rollout
