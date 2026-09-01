import os, sys, json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIMES = ["scratch", "frozen", "partial", "full"]
SEEDS   = [42, 123, 2026]


def tpr_at_fpr(y_true, scores, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    return float(tpr[valid].max()) if valid.any() else 0.0


def attack_accuracy(y_true, scores, rng_seed=0):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng    = np.random.default_rng(rng_seed)

    member_idx    = np.where(y_true == 1)[0]
    nonmember_idx = np.where(y_true == 0)[0]

    n             = min(len(member_idx), len(nonmember_idx))
    member_idx    = rng.choice(member_idx,    size=n, replace=False)
    nonmember_idx = rng.choice(nonmember_idx, size=n, replace=False)

    idx   = rng.permutation(np.concatenate([member_idx, nonmember_idx]))
    y_bal = y_true[idx]
    s_bal = scores[idx]

    member_bal    = np.where(y_bal == 1)[0]
    nonmember_bal = np.where(y_bal == 0)[0]
    rng.shuffle(member_bal)
    rng.shuffle(nonmember_bal)

    m_half = len(member_bal)    // 2
    n_half = len(nonmember_bal) // 2

    cal_idx  = np.concatenate([member_bal[:m_half],  nonmember_bal[:n_half]])
    eval_idx = np.concatenate([member_bal[m_half:],  nonmember_bal[n_half:]])

    cal_y, cal_s = y_bal[cal_idx],  s_bal[cal_idx]
    ev_y,  ev_s  = y_bal[eval_idx], s_bal[eval_idx]

    fpr, tpr, thresholds = roc_curve(cal_y, cal_s)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    preds = (ev_s >= best_thresh).astype(int)

    return float(balanced_accuracy_score(ev_y, preds))


def run_attack(y_true, scores, name):
    auc  = roc_auc_score(y_true, scores)
    tpr1 = tpr_at_fpr(y_true, scores, 0.001)
    tpr2 = tpr_at_fpr(y_true, scores, 0.01)
    acc  = attack_accuracy(y_true, scores)
    return {"attack": name, "auc": round(auc, 4),
            "tpr_at_0001_fpr":   round(tpr1, 4),
            "tpr_at_001_fpr":    round(tpr2, 4),
            "balanced_accuracy": round(acc,  4)}


def run_all_attacks(run_id):
    with open(f"experiments/{run_id}/sample_outputs.json") as f:
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

    with open(f"experiments/{run_id}/attack_results.json", "w") as f:
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

    print("\n" + "="*72)
    print(f"{'Regime':<10} {'Attack':<12} {'AUC':>10} {'TPR@0.1%':>12} {'TPR@1%':>10}")
    print("="*72)

    for regime in REGIMES:
        for attack in ["loss", "confidence", "entropy"]:
            rows  = [r for r in all_rows
                     if r["regime"] == regime and r["attack"] == attack]
            aucs  = [r["auc"]             for r in rows]
            tpr1s = [r["tpr_at_0001_fpr"] for r in rows]
            tpr2s = [r["tpr_at_001_fpr"]  for r in rows]
            print(f"{regime:<10} {attack:<12} "
                  f"{np.mean(aucs):.3f}+/-{np.std(aucs,  ddof=1):.3f}  "
                  f"{np.mean(tpr1s):.4f}+/-{np.std(tpr1s, ddof=1):.4f}  "
                  f"{np.mean(tpr2s):.4f}+/-{np.std(tpr2s, ddof=1):.4f}")

    os.makedirs("results", exist_ok=True)
    import csv
    with open("results/attack_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nSaved -> results/attack_summary.csv")


if __name__ == "__main__":
    main()
