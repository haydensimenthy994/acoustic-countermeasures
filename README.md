# acoustic-countermeasures

## Project objective
Build a baseline drone-vs-non-drone acoustic proxy classifier using public datasets, then evaluate adversarial attacks such as FGSM, PGD, and EOT-based PGD.

## Current status
This repository is currently in **Phase 1: dataset setup, preprocessing, and baseline proxy classifier development**.

### Implemented so far
- Project repository and local research structure
- Python virtual environment and dependency setup
- Config files for data, model, and training
- Audio preprocessing pipeline
- Log-mel spectrogram feature extraction
- Metadata manifest builder
- Train/validation/test split generation
- PyTorch dataset loader
- Placeholder proxy CNN model scaffold
- Seed and logging utilities
- Dataset/model smoke test scripts
- AudioSet metadata download pipeline
- UAV-DB access obtained

### In progress
- Inspecting UAV-DB metadata tables
- Mapping UAV-DB content into binary labels (`drone`, `no_drone`)
- Curating the local training dataset
- Implementing the full training loop
- Running the first clean baseline experiment

## Planned phases
1. Dataset setup and preprocessing
2. Baseline proxy classifier training
3. Clean evaluation
4. Adversarial attack implementation
5. Over-the-air simulation
6. Transferability / black-box evaluation

## Repository structure
- `configs/` — YAML configs for data, model, and training
- `data/` — raw, interim, processed, and metadata files
- `src/data/` — preprocessing, manifest building, splitting, dataset loading
- `src/features/` — audio feature extraction (log-mel spectrograms)
- `src/models/` — proxy model definitions and training logic
- `src/evaluation/` — evaluation metrics and result analysis
- `src/attacks/` — adversarial attack implementations
- `src/simulation/` — over-the-air / acoustic simulation components
- `src/utils/` — utilities such as logging and reproducibility
- `scripts/` — smoke tests and helper scripts
- `outputs/` — checkpoints, logs, figures, and experiment results

## Environment setup
Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt