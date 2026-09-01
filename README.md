# Membership Privacy Auditing in Transfer-Learned Medical Deep Learning

Systematic empirical analysis of how transfer-learning strategy affects membership inference leakage in biomedical image classifiers.

**Status:** Phase 1 — baseline training pipeline (DermaMNIST × ResNet18)

---

## Setup

```bash
python -m venv medmia
medmia\Scripts\activate       # Windows
pip install -r requirements.txt
```

---

## Run the first experiment

```bash
python scripts/train_target.py --config configs/derma_resnet18.yaml --regime scratch --seed 42
```

Regimes: `scratch` | `frozen` | `partial` | `full`

---

## Sanity checks (no data download needed)

```bash
pytest tests/test_models.py -v
```

---

## Repository structure

```
configs/          YAML hyperparameter files
src/data/         MedMNIST loaders and transforms
src/models/       ResNet-18 (Phase 1), DenseNet-121 (Phase 8)
src/training/     Trainer, evaluator, checkpoint utilities
src/attacks/      MIA attacks (Phase 4+)
scripts/          Entry-point scripts
experiments/      Per-run JSON configs, logs, and results
results/          Aggregated CSVs and summary tables
```

Seeds used: `42, 123, 2026`
