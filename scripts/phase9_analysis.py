import os, sys, json
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIMES = ["scratch", "frozen", "partial", "full"]
SEEDS   = [42, 123, 2026]
DATASET = "dermamnist"
ARCH    = "resnet18"


def load_run(regime, seed):
    run_id  = f"{DATASET}_{ARCH}_{regime}_seed{seed}"
    exp_dir = f"experiments/{run_id}"

    with open(f"{exp_dir}/results.json") as f:
        res = json.load(f)
    with open(f"{exp_dir}/attack_results.json") as f:
        atk = json.load(f)

    attacks = {a["attack"]: a for a in atk}

    return {
        "regime":       regime,
        "seed":         seed,
        "train_acc":    res["train"]["accuracy"],
        "test_acc":     res["test"]["accuracy"],
        "test_auc":     res["test"]["macro_auc"],
        "acc_gap":      res["generalization"]["accuracy_gap"],
        "loss_gap":     res["generalization"]["loss_gap"],
        "loss_mia_auc":  attacks["loss"]["auc"],
        "conf_mia_auc":  attacks["confidence"]["auc"],
        "entr_mia_auc":  attacks["entropy"]["auc"],
        "loss_tpr1":     attacks["loss"]["tpr_at_001_fpr"],
        "entr_tpr01":    attacks["entropy"]["tpr_at_0001_fpr"],
        "entr_tpr1":     attacks["entropy"]["tpr_at_001_fpr"],
    }


def mean_std(vals):
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def main():
    rows = []
    for regime in REGIMES:
        for seed in SEEDS:
            rows.append(load_run(regime, seed))

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'Regime':<10} {'Test AUC':>10} {'Gen Gap':>9} {'Loss MIA':>10} "
          f"{'Entr MIA':>10} {'Entr TPR@1%':>12}")
    print("="*80)

    for regime in REGIMES:
        r = [x for x in rows if x["regime"] == regime]
        print(f"{regime:<10} "
              f"{mean_std([x['test_auc']     for x in r]):>10}  "
              f"{mean_std([x['acc_gap']      for x in r]):>9}  "
              f"{mean_std([x['loss_mia_auc'] for x in r]):>10}  "
              f"{mean_std([x['entr_mia_auc'] for x in r]):>10}  "
              f"{mean_std([x['entr_tpr1']    for x in r]):>12}")

    # ── Spearman correlations ──────────────────────────────────────────────
    acc_gaps   = [x["acc_gap"]      for x in rows]
    loss_gaps  = [x["loss_gap"]     for x in rows]
    mia_aucs   = [x["loss_mia_auc"] for x in rows]
    entr_aucs  = [x["entr_mia_auc"] for x in rows]
    entr_tpr1s = [x["entr_tpr1"]   for x in rows]

    r1, p1 = spearmanr(acc_gaps,  mia_aucs)
    r2, p2 = spearmanr(loss_gaps, mia_aucs)
    r3, p3 = spearmanr(acc_gaps,  entr_tpr1s)

    print("\n── Spearman correlations (n=12) ──────────────────────────────────")
    print(f"  Accuracy gap  vs  Loss MIA AUC:      r={r1:.3f}  p={p1:.4f}")
    print(f"  Loss gap      vs  Loss MIA AUC:      r={r2:.3f}  p={p2:.4f}")
    print(f"  Accuracy gap  vs  Entropy TPR@1%:    r={r3:.3f}  p={p3:.4f}")

    # ── Per-regime averages for the paper table ────────────────────────────
    print("\n── Paper table (mean across 3 seeds) ────────────────────────────")
    print(f"{'Regime':<10} {'TrainAcc':>9} {'TestAUC':>8} {'GenGap':>8} "
          f"{'LossMIA':>8} {'EntrMIA':>8} {'EntrTPR1%':>10}")
    for regime in REGIMES:
        r = [x for x in rows if x["regime"] == regime]
        print(f"{regime:<10} "
              f"{np.mean([x['train_acc']     for x in r]):.3f}     "
              f"{np.mean([x['test_auc']      for x in r]):.3f}    "
              f"{np.mean([x['acc_gap']       for x in r]):.3f}   "
              f"{np.mean([x['loss_mia_auc']  for x in r]):.3f}   "
              f"{np.mean([x['entr_mia_auc']  for x in r]):.3f}   "
              f"{np.mean([x['entr_tpr1']     for x in r]):.4f}")

    # ── Save master CSV ────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    import csv
    with open("results/main_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved -> results/main_results.csv")


if __name__ == "__main__":
    main()
