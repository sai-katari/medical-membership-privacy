#!/usr/bin/env python3
"""Train a single target model configuration.

Usage
-----
python scripts/train_target.py --config configs/derma_resnet18.yaml --regime scratch --seed 42

Regimes: scratch | frozen | partial | full
"""

import argparse
import json
import os
import platform
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.utils.logging import setup_logger
from src.data.transforms import get_transforms
from src.data.medmnist_loader import get_dataloaders, verify_dataset
from src.models.model_factory import build_model, get_optimizer_groups
from src.training.trainer import train
from src.training.evaluate import evaluate
from src.training.checkpoint import load_checkpoint

SEEDS = [42, 123, 2026]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  required=True, help="Path to YAML config")
    p.add_argument("--regime",  required=True, choices=["scratch", "frozen", "partial", "full"])
    p.add_argument("--seed",    type=int, required=True)
    p.add_argument("--verify-layers", action="store_true",
                   help="Print requires_grad for every parameter before training")
    return p.parse_args()


def build_run_id(dataset: str, arch: str, regime: str, seed: int) -> str:
    return f"{dataset}_{arch}_{regime}_seed{seed}"


def save_experiment_config(run_id: str, config: dict, regime: str, seed: int) -> None:
    record = {
        **config,
        "training_regime": regime,
        "seed":            seed,
        "run_id":          run_id,
        "pretrained":      regime != "scratch",
        "torch_version":   torch.__version__,
        "cuda_version":    torch.version.cuda,
        "platform":        platform.platform(),
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(f"experiments/{run_id}", exist_ok=True)
    with open(f"experiments/{run_id}/config.json", "w") as f:
        json.dump(record, f, indent=2)


def scalar_metrics(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("targets", "probs", "preds")}


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config: dict = yaml.safe_load(f)

    run_id = build_run_id(config["dataset"], config["architecture"], args.regime, args.seed)
    logger = setup_logger(run_id)
    logger.info(f"Run: {run_id}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    verify_dataset(config["dataset"], config["image_size"])

    transforms = get_transforms(config["image_size"])
    loaders, n_classes = get_dataloaders(
        config["dataset"],
        transforms,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"] and torch.cuda.is_available(),
        image_size=config["image_size"],
    )
    config["num_classes"] = n_classes
    logger.info(f"n_classes={n_classes}")

    logger.info(f"Building {config['architecture']} | regime={args.regime}")
    model = build_model(config["architecture"], args.regime, n_classes).to(device)

    if args.verify_layers:
        from src.models.resnet import verify_frozen_layers
        verify_frozen_layers(model)

    save_experiment_config(run_id, config, args.regime, args.seed)

    param_groups = get_optimizer_groups(config["architecture"], model, args.regime, config)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"]
    )

    logger.info("Training...")
    history, best_val_auc = train(
        model, loaders, optimizer, scheduler, config, run_id, device, logger
    )

    out_dir = f"experiments/{run_id}"
    with open(f"{out_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Loading best checkpoint...")
    model, ckpt_metrics = load_checkpoint(model, run_id, device)

    results: dict = {}
    for split, loader_key in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
        logger.info(f"Evaluating {split}...")
        results[split] = scalar_metrics(evaluate(model, loaders[loader_key], device, n_classes))

    results["generalization"] = {
        "accuracy_gap": round(results["train"]["accuracy"] - results["test"]["accuracy"], 6),
        "loss_gap":     round(results["test"]["loss"]      - results["train"]["loss"],    6),
    }
    results["meta"] = {
        "run_id":                run_id,
        "best_val_auc":          round(best_val_auc, 6),
        "best_checkpoint_epoch": ckpt_metrics.get("epoch"),
    }

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"  Train | acc {results['train']['accuracy']:.4f} "
                f"f1 {results['train']['macro_f1']:.4f} "
                f"auc {results['train']['macro_auc']:.4f}")
    logger.info(f"  Val   | acc {results['val']['accuracy']:.4f} "
                f"f1 {results['val']['macro_f1']:.4f} "
                f"auc {results['val']['macro_auc']:.4f}")
    logger.info(f"  Test  | acc {results['test']['accuracy']:.4f} "
                f"f1 {results['test']['macro_f1']:.4f} "
                f"auc {results['test']['macro_auc']:.4f}")
    logger.info(f"  Gen gap | acc {results['generalization']['accuracy_gap']:+.4f} "
                f"loss {results['generalization']['loss_gap']:+.4f}")
    logger.info(f"  Saved -> experiments/{run_id}/")


if __name__ == "__main__":
    main()
