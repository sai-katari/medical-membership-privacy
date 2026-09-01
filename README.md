# Membership Privacy Auditing in Transfer-Learned Medical Deep Learning

How much does your fine-tuning strategy affect membership inference risk?
This repo investigates that question empirically using DermaMNIST and
standard ResNet-18 architectures across four training regimes.

## Background

Transfer learning is the default approach for biomedical image classification
when labeled data is scarce. But fine-tuning a pretrained model on sensitive
medical data raises a question that doesn't get asked often enough: does the
way you fine-tune affect how much the model memorizes individual training
examples?

This project tries to answer that with actual experiments rather than
assumptions.

## What I did

Trained ResNet-18 on DermaMNIST (224x224, MedMNIST+) under four conditions:

- **Scratch** - random init, everything trainable
- **Frozen** - ImageNet weights, only the classifier head updated (0.03% of params)
- **Partial FT** - ImageNet weights, layer4 + head unfrozen (75% of params)
- **Full FT** - ImageNet weights, everything unfrozen

Each regime was run with 3 random seeds (42, 123, 2026). After training,
I ran three membership inference attacks on every model: loss-based,
confidence-based, and entropy-based.

## Results

| Regime | Test AUROC | Gen Gap | Loss MIA AUC | Entropy TPR @ 1% FPR |
|--------|-----------|---------|--------------|----------------------|
| Frozen | 0.930 +/- 0.000 | 0.046 | 0.516 +/- 0.001 | 0.010 +/- 0.000 |
| Scratch | 0.931 +/- 0.004 | 0.062 | 0.525 +/- 0.011 | 0.010 +/- 0.001 |
| Partial FT | 0.967 +/- 0.001 | 0.148 | 0.679 +/- 0.036 | 0.015 +/- 0.003 |
| Full FT | 0.974 +/- 0.004 | 0.128 | 0.698 +/- 0.002 | 0.034 +/- 0.002 |

The generalization gap (train acc minus test acc) turns out to be a
strong predictor of leakage: Spearman r=0.965 (p<0.0001) against loss
MIA AUROC across all 12 runs.

Frozen probing barely leaks at all (AUC ~0.5 = basically random). Full
fine-tuning leaks the most. Partial FT has the largest generalization
gap of any regime, which explains why it leaks more than expected despite
using pretrained features.

One thing worth noting: entropy-based attacks are more sensitive than
loss or confidence attacks at low FPR, especially on full FT models.
Loss and confidence attacks give TPR=0 at 1% FPR for full FT, while
entropy gets 3.4%. That difference points to something about how full FT
models distribute their probability mass differently on training examples.

## Reproducing

```bash
pip install -r requirements.txt

# train one regime
python scripts/train_target.py --config configs/derma_resnet18.yaml --regime frozen --seed 42

# extract per-sample outputs
python scripts/extract_outputs.py --config configs/derma_resnet18.yaml --regime frozen --seed 42

# run attacks and analysis
python scripts/run_simple_attacks.py
python scripts/phase9_analysis.py
python scripts/generate_plots.py
```

Run all four regimes and all three seeds to replicate the full table.
Checkpoints are not in the repo but all per-run configs, metrics, and
attack results are under experiments/.

## Structure

    src/            source code (data loading, models, training, attacks)
    scripts/        end-to-end pipeline scripts
    configs/        hyperparameter files
    experiments/    per-run results and attack outputs
    results/        aggregated CSVs
    plots/          figures
    tests/          unit tests (model freezing, data shapes)

## What is next

Planning to extend to BloodMNIST and DenseNet-121 to check if the
pattern holds across datasets and architectures. LiRA (Carlini et al.)
is also on the list for a stronger attack baseline.

## Data

DermaMNIST from the MedMNIST+ benchmark (Yang et al., 2023).
Downloaded automatically via the medmnist package.
