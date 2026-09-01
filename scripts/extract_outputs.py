import os, sys, json, argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.data.transforms import get_transforms
from src.data.medmnist_loader import get_dataloaders
from src.models.model_factory import build_model
from src.training.checkpoint import load_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--regime", required=True)
    p.add_argument("--seed", type=int, required=True)
    return p.parse_args()

def extract_split(model, loader, device, membership_label):
    model.eval()
    records = []
    idx = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.squeeze(1).long()
            logits  = model(inputs.to(device))
            probs   = torch.softmax(logits, dim=1).cpu()
            preds   = probs.argmax(dim=1)

            for i in range(len(targets)):
                t      = targets[i].item()
                p_vec  = probs[i].numpy()
                pred   = preds[i].item()
                p_true = float(p_vec[t])
                loss   = float(-np.log(p_true + 1e-12))
                entropy = float(-(p_vec * np.log(p_vec + 1e-12)).sum())

                records.append({
                    "sample_idx": idx,
                    "true_class": t,
                    "pred_class": pred,
                    "prob_vector": p_vec.tolist(),
                    "p_true":     p_true,
                    "max_conf":   float(p_vec.max()),
                    "loss":       loss,
                    "entropy":    entropy,
                    "correct":    int(pred == t),
                    "membership": membership_label,
                })
                idx += 1
    return records

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_id  = f"{config['dataset']}_{config['architecture']}_{args.regime}_seed{args.seed}"
    out_dir = f"experiments/{run_id}"
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    transforms = get_transforms(config["image_size"])
    loaders, n_classes = get_dataloaders(
        config["dataset"], transforms,
        batch_size=256, num_workers=4,
        image_size=config["image_size"]
    )
    config["num_classes"] = n_classes

    model = build_model(config["architecture"], args.regime, n_classes).to(device)
    model, _ = load_checkpoint(model, run_id, device)

    print(f"Extracting members (train)...")
    members    = extract_split(model, loaders["train_eval"], device, membership_label=1)
    print(f"Extracting non-members (test)...")
    nonmembers = extract_split(model, loaders["test"],       device, membership_label=0)

    out_path = f"{out_dir}/sample_outputs.json"
    with open(out_path, "w") as f:
        json.dump(members + nonmembers, f)

    print(f"Saved {len(members)} members + {len(nonmembers)} non-members -> {out_path}")

if __name__ == "__main__":
    main()
