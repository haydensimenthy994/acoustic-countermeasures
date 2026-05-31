# Technical Code Review — `acoustic-countermeasures`

**Project:** Honours thesis — acoustic drone detection under adversarial countermeasures  
**Root:** `honours/acoustic-countermeasures/`  
**Review date:** May 2026 (updated after 5090 reproduction + SWARM cross-dataset branch)  
**Frozen results for thesis:** `outputs/results_vast_final/`, `outputs/figures_vast_final/`, `outputs/figures_cross/`

---

## Executive summary

This codebase implements a reproducible research pipeline: train two binary classifiers (log-mel ProxyAudioCNN and raw-waveform CNN14ProxyClassifier), evaluate white-box and black-box attacks, acoustic baselines, and optional external validation on SWARM-AUDIO-DATASET.

**Verdict:** Suitable for honours submission **if the thesis cites the May 2026 corrected snapshot** (`results_vast_final`). An earlier local backup (`results_backup_20260527_170040`) contains **stale EOT OTA metrics** (optimistic channel evaluation) and must not be cited.

**May 2026 audit (code + results):**

| Item | Status |
|------|--------|
| Spoofing baseline misnamed as recall | **Fixed** — real `evaluate_spoofing` + `evaluate_drone_recall` |
| Jamming ASR denominator | **Fixed** — `conditional_asr` for Figure 5 |
| EOT OTA missing noise at eval time | **Fixed** — batched `apply_eot_transforms_batched(with_noise=True)` |
| Seeded RIR bank | **Fixed** — `build_rir_bank(seed=42)` |
| Black-box summary hardcoded strings | **Fixed** — reads from CSV |
| SWARM cross-dataset experiments | **Added** — attacks, baselines, FGSM transfer, 5 figures |
| Automated tests / CI | **Still open** |
| `pyroomacoustics` in requirements.txt | **Still open** |
| Full one-command reproduction | **Partial** — `run_all_final.ps1` incomplete |

---

## 1. Repository structure

### 1.1 Annotated directory tree (current)

```text
honours/
└── acoustic-countermeasures/
    ├── README.md
    ├── requirements.txt              torchlibrosa pinned; pyroomacoustics NOT listed
    ├── .gitignore
    ├── docs/
    │   └── CODE_REVIEW.md            this document
    ├── configs/
    │   ├── data.yaml
    │   ├── model.yaml
    │   └── train.yaml
    ├── src/
    │   ├── data/
    │   │   ├── preprocess.py
    │   │   ├── build_manifest.py
    │   │   ├── split.py
    │   │   └── dataset.py
    │   ├── features/spectrograms.py
    │   ├── models/
    │   │   ├── pann_proxy.py
    │   │   ├── cnn14_proxy.py
    │   │   └── train.py
    │   ├── attacks/
    │   │   ├── fgsm.py
    │   │   ├── pgd.py
    │   │   ├── eot_pgd.py            RIR + gain + noise; batched EOT stack
    │   │   ├── baselines.py          jamming, spoofing, drone_recall
    │   │   └── _pgd_logging.py       UNUSED — never imported
    │   ├── viz/
    │   │   ├── style.py
    │   │   ├── fig1..fig6_*.py       in-distribution figures
    │   │   └── cross/                SWARM figures (fig2–6, no training curves)
    │   └── utils/
    │       ├── seed.py
    │       └── logger.py
    ├── scripts/                      19 first-party experiment scripts + 2 smoke tests
    │   ├── run_all_final.ps1         partial orchestrator (no figure / sample steps)
    │   ├── evaluate_baseline.py
    │   ├── run_{fgsm,pgd,eot_pgd,baselines,blackbox_transfer}.py
    │   ├── run_pgd_with_samples.py
    │   ├── generate_all_figures.py
    │   ├── build_swarm_manifest.py
    │   ├── evaluate_cross_dataset.py
    │   ├── run_cross_dataset_{attacks,baselines,transfer}.py
    │   ├── run_cross_pgd_with_samples.py
    │   ├── reaggregate_cross_dataset_attacks.py
    │   ├── generate_cross_figures.py
    │   ├── check_duration_leakage.py
    │   └── parse_train_log.py
    ├── data/                         gitignored (raw, processed, external/SWARM)
    ├── outputs/                      gitignored
    │   ├── checkpoints/
    │   ├── results/                  working copy (may match vast_final)
    │   ├── results_vast_final/       frozen 5090 snapshot — cite in thesis
    │   ├── figures/
    │   ├── figures_vast_final/
    │   └── figures_cross/
    └── audioset_tagging_cnn/         vendored PANNs (~30 files; ~2 imported)
```

**Removed since original review:** empty `src/evaluation/`, `src/simulation/`, `src/utils/io.py`, empty `notebooks/` (stubs never implemented or deleted).

**README drift:** README still lists `src/evaluation/` and `src/simulation/` under repository structure; those paths do not exist. Cross-dataset workflow is not fully documented.

### 1.2 Misplaced, redundant, or unexplained files

| Path | Issue |
|------|--------|
| `src/attacks/_pgd_logging.py` | Dead code. `scripts/run_pgd_with_samples.py` inlines the same logic. |
| `audioset_tagging_cnn/` | Full PANNs clone; only `pytorch/models.py` (Cnn14) is imported via `cnn14_proxy.py`. |
| `outputs/checkpoints_backup/` | Duplicate checkpoints (~700 MB) if present — archive only. |
| `data/interim/` vs `data/processed/` | May duplicate WAVs if both populated — keep one canonical tree. |
| `outputs/results_backup_*` | Pre-audit or pre-5090 snapshots — do not cite for EOT OTA. |

---

## 2. Technology stack

| Component | Role | Notes |
|-----------|------|--------|
| Python 3.10+ | Language | Implied by venv |
| PyTorch / torchaudio | Training, attacks, mel | Unpinned in requirements.txt |
| torchlibrosa 0.1.0 | PANNs dependency | **Pinned** (librosa API compatibility) |
| pyroomacoustics | RIR in `eot_pgd.py` | **Required but NOT in requirements.txt** |
| soundfile, pandas, numpy, scipy | I/O and metrics | |
| scikit-learn | Split + classification metrics | |
| matplotlib | All figures | |
| PANNs Cnn14 (vendored) | CNN14 backbone | |
| torchvision, librosa, tqdm, jupyter | Listed | **Unused** in first-party code |

### 2.1 Notable configuration choices

- 16 kHz mono, 5 s fixed duration (padding at end of clip).
- 64-bin log-mel, 25 ms frame / 10 ms hop.
- Class balance by undersampling → 2,728 rows in `split_metadata.csv` (410 test).
- CNN14 training `lr=1e-4` hardcoded in `train.py` (overrides `model.yaml`).
- Early stopping `patience=5` hardcoded in `train.py`.
- `weights_only=False` on all `torch.load` (PANNs checkpoint format).

---

## 3. Architecture overview

### 3.1 Data flow

**In-distribution (UAV-DB + AudioSet, trained splits):**

1. Preprocess → manifest → stratified split  
2. `DroneAudioDataset` (`use_raw_waveform` flag)  
3. Train ProxyAudioCNN (`best_model.pt`) and CNN14ProxyClassifier (`best_model_cnn14.pt`)  
4. Attacks and baselines on **410 test clips**  
5. Results → `outputs/results/*.csv|json|npz`  
6. Figures → `outputs/figures/` (fig1–6)

**Cross-dataset (SWARM, eval-only, no retraining):**

1. `build_swarm_manifest.py` → `data/metadata/swarm_test_manifest.csv`  
2. Same CNN14 checkpoint, ~3,556 clips  
3. Clean eval (CNN14 + Proxy), FGSM/PGD attacks (PGD **20** steps), jamming/spoofing, **FGSM-only** black-box transfer  
4. Results → `cross_dataset_*`  
5. Figures → `outputs/figures_cross/` (fig2_cross–fig6_cross)

### 3.2 Entry points (thesis reproduction minimum)

**In-distribution**

```text
python -m src.data.preprocess ...
python -m src.data.build_manifest
python -m src.data.split
python -m src.models.train --model-type cnn14
python -m src.models.train --model-type proxy_cnn
python scripts/evaluate_baseline.py
python scripts/run_fgsm.py
python scripts/run_pgd.py
python scripts/run_eot_pgd.py
python scripts/run_baselines.py
python scripts/run_blackbox_transfer.py
python scripts/run_pgd_with_samples.py
python scripts/parse_train_log.py
python scripts/generate_all_figures.py
```

**Cross-dataset (SWARM data must exist under `data/external/`)**

```text
python scripts/build_swarm_manifest.py
python scripts/evaluate_cross_dataset.py
python scripts/run_cross_dataset_attacks.py
python scripts/run_cross_dataset_baselines.py
python scripts/run_cross_dataset_transfer.py
python scripts/reaggregate_cross_dataset_attacks.py   # if JSON lacks by_source
python scripts/run_cross_pgd_with_samples.py
python scripts/generate_cross_figures.py
```

`scripts/run_all_final.ps1` runs steps 1–6 in-dist and 7–10 cross **except** sample-export and figure generation.

### 3.3 Design patterns

- Linear ETL pipeline with CSV boundaries between stages.
- Strategy: one `DroneAudioDataset`, two feature modes (raw vs mel).
- Template method: attack kernels share `(model, features, labels, ε) → (adv, δ)`.
- Figures read serialised results only (no re-inference in viz).
- In-dist / cross script pairs (duplication by design for honours scope).

---

## 4. Module highlights

### 4.1 Data (`src/data/`)

- **`dataset.py`:** Loads WAV per `__getitem__`; pads/truncates to 5 s; returns `filepath` for cross joins.
- **`split.py`:** Undersample + 70/15/15, seed 42.
- **Risks:** No file-level cache; missing files crash workers; `infer_label` can silently return `None` in `build_manifest.py`.

### 4.2 Models (`src/models/`)

- **`pann_proxy.py`:** Small CNN on log-mel (~97% val in logs).
- **`cnn14_proxy.py`:** PANNs Cnn14 + 2-layer head; dynamic import avoids `models` name clash.
- **`train.py`:** Shared loop; hardcoded CNN14 lr and early-stop patience.

### 4.3 Attacks (`src/attacks/`)

| Module | Role |
|--------|------|
| `fgsm.py` | Single-step L∞ FGSM; conditional ASR |
| `pgd.py` | 40-step PGD (in-dist scripts); `random_start=False` for transfer |
| `eot_pgd.py` | EOT-PGD + `apply_eot_transforms_batched`; OTA majority vote with noise |
| `baselines.py` | Jamming SNR sweep; real spoofing; drone_recall sanity metric |

**EOT configuration (final run):** 20 outer steps, 5 EOT samples, 2 ε values (0.001, 0.005), 20-room RIR bank, seed 42.

### 4.4 Visualisation (`src/viz/`)

- **`style.py`:** Shared matplotlib rcParams and colour palette.
- **In-dist fig1–6:** Training curves, confusion, ASR vs ε, confidence histogram, ASR vs SNR, spectrograms.
- **Cross fig2–6:** No fig1 (no SWARM training). Fig5 includes spoofing curve on SWARM.

---

## 5. Cross-dataset scope (SWARM)

SWARM was integrated **late** (~5 days before submission). All SWARM work is **evaluation-only** on frozen checkpoints.

| Experiment | In-dist | SWARM |
|------------|---------|-------|
| Clean accuracy | Yes | Yes (CNN14 + Proxy) |
| FGSM / PGD white-box | Yes (PGD 40 steps) | Yes (PGD **20** steps) |
| EOT-PGD white-box | Yes (2 ε) | **No** |
| Jamming / spoofing | Yes | Yes |
| Black-box transfer | FGSM + PGD + EOT | **FGSM only** |
| Figures | fig1–6 | fig2_cross–fig6_cross |

**Interpretation:** High SWARM FGSM transfer ASR (~66% @ ε=0.001) partly reflects Proxy weakness on SWARM (~69% clean), not only transferability.

---

## 6. Algorithms (summary)

- **FGSM:** \(x' = x + \varepsilon \operatorname{sign}(\nabla_x L)\); conditional ASR on clean-correct samples.
- **PGD:** L∞ projected gradient descent; α = ε/10.
- **EOT-PGD:** Gradient averaged over random RIR, gain ∈ [0.7, 1.3], noise SNR ∈ [20, 40] dB.
- **Jamming:** Broadband Gaussian at target SNR; `conditional_asr` matches attack denominator.
- **Spoofing:** Mix scaled drone clip into clean-correct `no_drone` clips; measure induced `drone` predictions.
- **SNR metric:** \(10 \log_{10}(\|x\|^2 / \|\delta\|^2)\) per sample, averaged.

---

## 7. Code quality

### 7.1 Strengths

- Reproducibility: `set_seed(42)` on attack/baseline/cross runners; seeded RIR bank.
- Clean stage separation (data / models / attacks / viz / scripts).
- Shared `compute_perturbation_metrics()` across attacks.
- Conditional ASR used consistently for comparability.
- Cross aggregation: path normalisation, Wilson CIs, per-source breakdowns.
- Figure code decoupled from GPU inference.

### 7.2 Fragile areas

- ~48+ hardcoded paths (`outputs/...`, `data/metadata/...`).
- Duplicated `load_config()` and model-loading boilerplate across scripts.
- In-dist vs cross script pairs must be maintained together.
- `evaluate_baseline.py` omits `set_seed(42)`.
- No automated tests; smoke scripts only (`test_model.py`, `test_dataset.py`).

---

## 8. Issues and risks

### 8.1 Issue register (updated)

| ID | Severity | Status | Description |
|----|----------|--------|-------------|
| #1 | High | **FIXED** | Spoofing baseline was drone recall; now `evaluate_spoofing` + `evaluate_drone_recall`. |
| #2 | Medium | **FIXED** | OTA eval omitted noise; now uses full EOT stack. **OTA @ ε=0.001: 37%** (was ~75% in stale backup). |
| #3 | Medium | Open | RIR peak re-normalisation may affect effective L∞ budget. |
| #4 | Medium | **FIXED** | RIR bank now `build_rir_bank(seed=42)`. |
| #5 | Low | **FIXED** | Black-box summary reads CSVs, not hardcoded strings. |
| #6 | Low | Open | `train.py` CNN14 lr=1e-4 overrides config. |
| #7 | Low | Open | `evaluate_*` can return ASR=0 with total=0 silently. |
| #8 | Low | Open | `infer_label` returns None without loud failure. |
| #9 | Medium | Open | `pyroomacoustics` missing from `requirements.txt`. |
| #10 | Doc | Open | README structure section outdated; full repro order incomplete. |
| #11 | Science | Open | Cross PGD 20 vs in-dist PGD 40 steps — document in methods. |
| #12 | Scope | N/A | No EOT-PGD on SWARM — document as limitation. |

### 8.2 Security

- `torch.load(..., weights_only=False)` on checkpoints — acceptable for self-generated weights only.
- No credentials or network calls in experiment code.

### 8.3 Performance

- EOT-PGD batched (major improvement vs per-sample loops).
- Cross SWARM PGD / `run_cross_pgd_with_samples` still heavy (~3.5k clips).
- Dataset reads disk every epoch — fine at thesis scale.

---

## 9. Reproducibility

### 9.1 Strengths

- Fixed seed 42 for splits, training config, attack scripts, RIR bank.
- Results as CSV/JSON/NPZ — diffable and figure-regenerable.
- Frozen directories: `results_vast_final`, `figures_vast_final`, `figures_cross`.

### 9.2 Gaps

- Unpinned dependencies (except `torchlibrosa`).
- Install `pyroomacoustics` manually: `pip install pyroomacoustics`
- Raw UAV-DB not in repo; SWARM under `data/external/` (gitignored).
- No `requirements-lock.txt`, no pytest, no CI.
- `run_all_final.ps1` does not run figure or PGD-sample scripts.

### 9.3 Fresh environment (minimal)

```bash
pip install -r requirements.txt
pip install pyroomacoustics
# Place checkpoints in outputs/checkpoints/
# Place data per README / manifest paths
# Run script order in §3.2
```

---

## 10. Thesis headline results (`results_vast_final`)

**In-distribution test set (N = 410, clean accuracy 97.56%)**

| Method | Metric @ ε=0.001 | Notes |
|--------|------------------|-------|
| FGSM | 8.25% ASR | SNR ~45 dB |
| PGD | 99.0% ASR | 40 steps |
| EOT-PGD | 81.0% digital / **37.0% OTA** | 20 steps, 5 EOT |
| PGD → Proxy transfer | 43.8% transfer ASR | mel bridge |
| Jamming | ~5% conditional ASR @ high SNR | |
| Spoofing | 0–96% across SNR sweep | real false-alarm test |

**SWARM (conditional ASR, clean-correct denominator ~3491)**

| Metric | Value |
|--------|-------|
| CNN14 clean | 98.17% |
| Proxy clean | 68.87% |
| FGSM ASR @ ε=0.001 | 22.14% |
| PGD ASR @ ε=0.001 | 100% |
| FGSM transfer @ ε=0.001 | 66.40% |

---

## 11. Recommendations

### For thesis writing (no code required)

1. Cite **`outputs/results_vast_final`** only for EOT OTA and post-audit baselines.
2. State SWARM as **external, eval-only**, received late; no EOT on SWARM.
3. State **PGD 20 steps** on SWARM vs **40** in-dist.
4. Explain OTA definition: majority vote over 5 transforms with RIR + gain + noise 20–40 dB.
5. Explain high SWARM transfer ASR vs weak Proxy clean accuracy.

### Optional code hygiene (post-submission)

1. Add `pyroomacoustics` to `requirements.txt`; remove unused deps.
2. Update README tree; add full reproduction matrix (script → output → figure).
3. Delete `_pgd_logging.py`; add `set_seed` to `evaluate_baseline.py`.
4. Extend `run_all_final.ps1` with sample + figure steps.
5. Centralise paths in `configs/paths.yaml` or `src/utils/paths.py`.
6. Move RIR/noise helpers to a single module (optional `src/simulation/ota.py`).
7. Add minimal pytest: conditional ASR math, ε-ball clamp, one-batch FGSM sign.

---

## Appendix: Delta from original review (pre–May 2026 audit)

The first code review flagged spoofing metric error, optimistic EOT OTA, unseeded RIRs, and hardcoded black-box summaries. Those items are **fixed in code** and reflected in `results_vast_final`. A **SWARM branch** (clean eval, attacks, baselines, FGSM transfer, five cross figures) was added after the main experiments. EOT-PGD and PGD/EOT transfer were **not** run on SWARM. Total first-party Python: ~28 `src` files + ~19 experiment scripts + vendored PANNs.

---

*End of review.*
