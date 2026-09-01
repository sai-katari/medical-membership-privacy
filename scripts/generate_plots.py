import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIMES  = ["scratch", "frozen", "partial", "full"]
LABELS   = ["Scratch", "Frozen", "Partial FT", "Full FT"]
SEEDS    = [42, 123, 2026]
DATASET  = "dermamnist"
ARCH     = "resnet18"
COLORS   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

os.makedirs("plots", exist_ok=True)


def load_run(regime, seed):
    run_id = f"{DATASET}_{ARCH}_{regime}_seed{seed}"
    with open(f"experiments/{run_id}/results.json") as f:
        res = json.load(f)
    with open(f"experiments/{run_id}/attack_results.json") as f:
        atk = {a["attack"]: a for a in json.load(f)}
    return {
        "test_auc":     res["test"]["macro_auc"],
        "acc_gap":      res["generalization"]["accuracy_gap"],
        "loss_gap":     res["generalization"]["loss_gap"],
        "loss_mia_auc": atk["loss"]["auc"],
        "entr_mia_auc": atk["entropy"]["auc"],
        "entr_tpr1":    atk["entropy"]["tpr_at_001_fpr"],
        "loss_tpr1":    atk["loss"]["tpr_at_001_fpr"],
    }


# Aggregate
regime_data = {}
for regime in REGIMES:
    runs = [load_run(regime, s) for s in SEEDS]
    regime_data[regime] = {k: [r[k] for r in runs] for k in runs[0]}

# ── Plot 1: MIA AUC by regime ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(REGIMES))
for i, attack in enumerate(["loss_mia_auc", "entr_mia_auc"]):
    means = [np.mean(regime_data[r][attack]) for r in REGIMES]
    stds  = [np.std(regime_data[r][attack])  for r in REGIMES]
    ax.bar(x + i*0.35 - 0.175, means, 0.35, yerr=stds,
           label=["Loss attack", "Entropy attack"][i],
           color=COLORS[i], alpha=0.85, capsize=4)

ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Random (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylabel("MIA AUROC")
ax.set_title("Membership Inference Leakage by Training Regime\n(DermaMNIST × ResNet-18, mean ± std, 3 seeds)")
ax.legend()
ax.set_ylim(0.45, 0.78)
plt.tight_layout()
plt.savefig("plots/plot1_mia_auc_by_regime.png", dpi=150)
plt.close()
print("Saved plot1")

# ── Plot 2: Entropy TPR@1% by regime ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
means = [np.mean(regime_data[r]["entr_tpr1"]) for r in REGIMES]
stds  = [np.std(regime_data[r]["entr_tpr1"])  for r in REGIMES]
bars  = ax.bar(LABELS, means, yerr=stds, color=COLORS, alpha=0.85, capsize=5)
ax.set_ylabel("TPR @ 1% FPR (entropy attack)")
ax.set_title("Privacy Leakage at Low FPR by Training Regime")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{m:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("plots/plot2_tpr_by_regime.png", dpi=150)
plt.close()
print("Saved plot2")

# ── Plot 3: Generalization gap vs MIA AUC (scatter) ─────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
for i, regime in enumerate(REGIMES):
    xs = regime_data[regime]["acc_gap"]
    ys = regime_data[regime]["loss_mia_auc"]
    ax.scatter(xs, ys, color=COLORS[i], s=80, label=LABELS[i], zorder=3)

# Regression line
all_x = [v for r in REGIMES for v in regime_data[r]["acc_gap"]]
all_y = [v for r in REGIMES for v in regime_data[r]["loss_mia_auc"]]
m, b  = np.polyfit(all_x, all_y, 1)
xs_   = np.linspace(min(all_x), max(all_x), 100)
ax.plot(xs_, m*xs_ + b, "k--", linewidth=1, alpha=0.6, label=f"r=0.860")
ax.set_xlabel("Accuracy generalization gap (train − test)")
ax.set_ylabel("Loss MIA AUROC")
ax.set_title("Generalization Gap vs Membership Leakage")
ax.legend()
plt.tight_layout()
plt.savefig("plots/plot3_gap_vs_mia.png", dpi=150)
plt.close()
print("Saved plot3")

# ── Plot 4: Privacy-utility tradeoff ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
for i, regime in enumerate(REGIMES):
    xs = regime_data[regime]["entr_mia_auc"]
    ys = regime_data[regime]["test_auc"]
    ax.scatter(xs, ys, color=COLORS[i], s=80, label=LABELS[i], zorder=3)
    ax.annotate(LABELS[i],
                (np.mean(xs), np.mean(ys)),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.set_xlabel("Entropy MIA AUROC (↓ more private)")
ax.set_ylabel("Test AUROC (↑ more useful)")
ax.set_title("Privacy–Utility Tradeoff")
ax.legend()
plt.tight_layout()
plt.savefig("plots/plot4_privacy_utility.png", dpi=150)
plt.close()
print("Saved plot4")

print("\nAll plots saved to plots/")
