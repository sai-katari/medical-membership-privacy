import os, sys, json
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIMES = ["scratch", "frozen", "partial", "full"]
SEEDS   = [42, 123, 2026]
DATASET = "dermamnist"
ARCH    = "resnet18"


def load_run(regime, seed):
    run_id = f"{DATASET}_{ARCH}_{regime}_seed{seed}"
    with open(f"experiments/{run_id}/results.json") as f:
        res = json.load(f)
    with open(f"experiments/{run_id}/attack_results.json") as f:
        atk = {a["attack"]: a for a in json.load(f)}
    return {
        "regime":       regime,
        "seed":         seed,
        "train_acc":    res["train"]["accuracy"],
        "test_acc":     res["test"]["accuracy"],
        "test_auc":     res["test"]["macro_auc"],
        "acc_gap":      res["generalization"]["accuracy_gap"],
        "loss_gap":     res["generalization"]["loss_gap"],
        "loss_mia_auc": atk["loss"]["auc"],
        "entr_mia_auc": atk["entropy"]["auc"],
        "entr_tpr1":    atk["entropy"]["tpr_at_001_fpr"],
        "loss_tpr1":    atk["loss"]["tpr_at_001_fpr"],
    }


def main():
    rows = [load_run(r, s) for r in REGIMES for s in SEEDS]

    print("\n" + "="*82)
    print(f"{'Regime':<10} {'Test AUC':>12} {'Gen Gap':>10} "
          f"{'Loss MIA':>12} {'Entr MIA':>12} {'Entr TPR@1%':>12}")
    print("="*82)

    for regime in REGIMES:
        r = [x for x in rows if x["regime"] == regime]
        def ms(key):
            v = [x[key] for x in r]
            return f"{np.mean(v):.3f}+/-{np.std(v, ddof=1):.3f}"
        print(f"{regime:<10} {ms('test_auc'):>12}  {ms('acc_gap'):>10}  "
              f"{ms('loss_mia_auc'):>12}  {ms('entr_mia_auc'):>12}  "
              f"{ms('entr_tpr1'):>12}")

    acc_gaps  = [x["acc_gap"]      for x in rows]
    loss_gaps = [x["loss_gap"]     for x in rows]
    mia_aucs  = [x["loss_mia_auc"] for x in rows]
    entr_tpr1 = [x["entr_tpr1"]   for x in rows]

    r1, p1 = spearmanr(acc_gaps,  mia_aucs)
    r2, p2 = spearmanr(loss_gaps, mia_aucs)
    r3, p3 = spearmanr(acc_gaps,  entr_tpr1)

    print("\nSpearman correlations (n=12)")
    print(f"  Accuracy gap  vs  Loss MIA AUC:    rho={r1:.3f}  p={p1:.2e}")
    print(f"  Loss gap      vs  Loss MIA AUC:    rho={r2:.3f}  p={p2:.2e}")
    print(f"  Accuracy gap  vs  Entropy TPR@1%:  rho={r3:.3f}  p={p3:.2e}")

    os.makedirs("results", exist_ok=True)
    import csv
    with open("results/main_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved -> results/main_results.csv")


if __name__ == "__main__":
    main()
