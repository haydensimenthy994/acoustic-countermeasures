# acoustic-countermeasures

## Project objective
Build a baseline drone-vs-non-drone acoustic proxy classifier using public datasets, then evaluate adversarial attacks such as FGSM, PGD, and EOT-based PGD.

## Current status
Phases 1–6 complete. Two classifiers trained (ProxyAudioCNN scratch on
log-mel; CNN14ProxyClassifier fine-tuned from PANNs on raw waveform).
In-distribution white-box attacks (FGSM, PGD, EOT-PGD at five ε values),
jamming, and acoustic-spoofing baselines are run. Black-box transfer
(CNN14 → ProxyAudioCNN) is evaluated for FGSM, PGD, and EOT-PGD.
Cross-dataset evaluation on SWARM-AUDIO-DATASET covers clean accuracy,
white-box attacks (FGSM, PGD, EOT-PGD), baselines, and black-box transfer
(PGD and EOT-PGD). Thesis figures are generated from saved CSVs/JSONs.

Frozen result snapshots live in `outputs/results_vast_final/`.

### Reproducibility / correctness notes (May 2026 audit)
After a code review of the result files, the following were addressed:

- `evaluate_spoofing` previously computed the model's drone-class
  recall and mis-reported it as a spoofing ASR. The legacy number is
  now produced by `evaluate_drone_recall` (`outputs/results/drone_recall.json`)
  under its correct name, and a real acoustic-spoofing baseline
  (drone audio injected into no_drone clips at a range of SNRs) is
  written to `outputs/results/spoofing_results.csv`.
- `evaluate_jamming` now reports both a `conditional_asr` (using the
  same clean-correct denominator as the gradient attacks, suitable for
  Figure 5) and the legacy `asr` (population, `1 - accuracy`).
- The EOT-PGD OTA evaluation now applies the same additive Gaussian
  noise (SNR 20–40 dB) that the attack saw during training.
  Previously the OTA evaluation applied RIR + gain only, making the
  reported OTA ASR optimistic. The same fix is applied to the OTA
  branch of `scripts/run_blackbox_transfer.py`.
- The RIR bank is now generated with a fixed seed
  (`build_rir_bank(seed=42)`), so EOT-PGD runs are bit-reproducible.
- Black-box PGD/EOT transfer uses clean-start attacks
  (`random_start=False` in `scripts/run_blackbox_transfer.py` and the
  cross-dataset transfer runners) so perturbations are crafted from
  clean audio rather than random L∞ noise inside the ε-ball.
- Attack loops use `torch.autograd.grad` on the perturbation tensor
  instead of `model.zero_grad()` + `loss.backward()`, avoiding side
  effects when the same model is reused for mel/target evaluation.
- `set_seed(42)` is called at the top of every `scripts/run_*.py`
  attack and baseline script.
- `scripts/check_duration_leakage.py` audits whether drone vs
  no_drone clip durations differ enough to make padding a class
  shortcut.

## Planned phases
1. Dataset setup and preprocessing  [done]
2. Baseline proxy classifier training  [done]
3. Clean evaluation  [done]
4. Adversarial attack implementation  [done]
5. Over-the-air simulation  [done]
6. Transferability / black-box evaluation  [done]

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
- `scripts/` — experiment runners and helper scripts
- `outputs/` — checkpoints, logs, figures, and experiment results
- `outputs/results_vast_final/` — frozen thesis result CSVs/JSONs

## Key experiment scripts

In-distribution (400-sample test split, `data/metadata/split_metadata.csv`):

| Script | Output |
|--------|--------|
| `scripts/run_fgsm.py` | `outputs/results/fgsm_results_cnn14.csv` |
| `scripts/run_pgd.py` | `outputs/results/pgd_results_cnn14.csv` |
| `scripts/run_eot_pgd.py` | `outputs/results/eot_pgd_results_cnn14.csv` |
| `scripts/run_blackbox_transfer.py` | FGSM, PGD, and EOT transfer CSVs |
| `scripts/run_blackbox_pgd_transfer.py` | `blackbox_pgd_transfer.csv` only |
| `scripts/run_blackbox_eot_transfer.py` | `blackbox_eot_transfer.csv` only |

SWARM cross-dataset (`data/metadata/swarm_test_manifest.csv` from
`scripts/build_swarm_manifest.py`):

| Script | Output |
|--------|--------|
| `scripts/evaluate_cross_dataset.py` | `cross_dataset_swarm.csv` |
| `scripts/run_cross_dataset_attacks.py` | `cross_dataset_attacks.csv` |
| `scripts/run_cross_dataset_transfer.py` | `cross_dataset_transfer.csv` |
| `scripts/run_cross_dataset_baselines.py` | `cross_dataset_jamming.csv`, `cross_dataset_spoofing.csv` |
| `scripts/generate_cross_figures.py` | cross-dataset thesis figures |

Long EOT and cross-dataset sweeps support `--resume` and `--epsilons`
(comma-separated) so interrupted runs skip completed ε values.

Example:

```powershell
python scripts/run_eot_pgd.py --epsilons 0.001,0.005,0.01,0.02,0.05 --resume
python scripts/run_cross_dataset_attacks.py --attacks EOT-PGD --resume
python scripts/run_cross_dataset_transfer.py --attacks PGD,EOT-PGD --resume
```

## Environment setup
Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

EOT-PGD also requires `pyroomacoustics`:

```powershell
pip install pyroomacoustics
```