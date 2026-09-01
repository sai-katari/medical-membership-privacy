import os, sys, json, argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIMES = ["scratch", "frozen", "partial", "full"]
SEEDS   = [42, 123, 2026]


def tpr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    return float(tpr[valid].max()) if valid.any() else 0.0


def attack_accuracy(y_true, scores, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    idx = rng.permutation(len(y_true))
    y_true  = np.array(y_true)[idx]
    scores  = np.array(scores)[idx]
    half    = len(y_true) // 2
    cal_y, cal_s = y_true[:half], scores[:half]
    ev_y,  ev_s  = y_true[half:], scores[half:]
    fpr, tpr, thresholds = roc_curve(cal_y, cal_s)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    preds = (ev_s >= best_thresh).astype(int)
    return float((preds == ev_y).mean())


def run_attack(y_true, scores, name):
    auc  = roc_auc_score(y_true, scores)
    tpr1 = tpr_at_fpr(y_true, scores, 0.001)
    tpr2 = tpr_at_fpr(y_true, scores, 0.01)
    acc  = attack_accuracy(y_true, scores)
    return {"attack": name, "auc": round(auc,4),
            "tpr_at_0001_fpr": round(tpr1,4),
            "tpr_at_001_fpr":  round(tpr2,4),
            "accuracy":        round(acc,4)}


def run_all_attacks(run_id):
    path = f"experiments/{run_id}/sample_outputs.json"
    with open(path) as f:
        records = json.load(f)

    y_true      = [r["membership"] for r in records]
    loss_scores = [-r["loss"]      for r in records]
    conf_scores = [r["max_conf"]   for r in records]
    entr_scores = [-r["entropy"]   for r in records]

    results = [
        run_attack(y_true, loss_scores,  "loss"),
        run_attack(y_true, conf_scores,  "confidence"),
        run_attack(y_true, entr_scores,  "entropy"),
    ]

    out_path = f"experiments/{run_id}/attack_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    all_rows = []

    for regime in REGIMES:
        for seed in SEEDS:
            run_id  = f"dermamnist_resnet18_{regime}_seed{seed}"
            results = run_all_attacks(run_id)
            for r in results:
                all_rows.append({"regime": regime, "seed": seed, **r})
            print(f"{run_id}: loss_auc={results[0]['auc']:.4f}  "
                  f"tpr@0.1%={results[0]['tpr_at_0001_fpr']:.4f}  "
                  f"tpr@1%={results[0]['tpr_at_001_fpr']:.4f}")

    print("\n" + "="*70)
    print(f"{'Regime':<10} {'Attack':<12} {'AUC':>10} {'TPR@0.1%':>12} {'TPR@1%':>10}")
    print("="*70)

    for regime in REGIMES:
        for attack in ["loss", "confidence", "entropy"]:
            rows  = [r for r in all_rows
                     if r["regime"] == regime and r["attack"] == attack]
            aucs  = [r["auc"]             for r in rows]
            tpr1s = [r["tpr_at_0001_fpr"] for r in rows]
            tpr2s = [r["tpr_at_001_fpr"]  for r in rows]
            print(f"{regime:<10} {attack:<12} "
                  f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
                  f"{np.mean(tpr1s):.4f}±{np.std(tpr1s):.4f}  "
                  f"{np.mean(tpr2s):.4f}±{np.std(tpr2s):.4f}")

    os.makedirs("results", exist_ok=True)
    import csv
    with open("results/attack_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nSaved -> results/attack_summary.csv")


if __name__ == "__main__":
    main()
