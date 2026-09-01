"""Data loading sanity checks.

Tests marked 'download' require internet and ~200 MB disk space.
Skip them with: pytest tests/test_data.py -m "not download"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from src.data.transforms import get_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.data.medmnist_loader import get_dataset_info, get_dataloaders


def test_imagenet_stats():
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD)  == 3


def test_transforms_keys():
    t = get_transforms(224)
    assert set(t.keys()) == {"train", "val", "test"}


def test_transforms_output_shape():
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    tensor = get_transforms(224)["train"](img)
    assert tensor.shape == (3, 224, 224)


def test_dermamnist_info():
    info = get_dataset_info("dermamnist")
    assert info["n_classes"]  == 7
    assert info["n_channels"] == 3
    assert info["task"]       == "multi-class"


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="not in registry"):
        get_dataloaders("fakemnist", get_transforms(), batch_size=4)


@pytest.mark.download
def test_dermamnist_loader_shapes():
    loaders, n_classes = get_dataloaders(
        "dermamnist", get_transforms(), batch_size=8, num_workers=0
    )
    assert n_classes == 7
    assert set(loaders.keys()) == {"train", "train_eval", "val", "test"}
    inputs, targets = next(iter(loaders["train"]))
    assert inputs.shape == (8, 3, 224, 224)
    assert targets.squeeze(1).shape == (8,)


@pytest.mark.download
def test_train_eval_deterministic():
    loaders, _ = get_dataloaders(
        "dermamnist", get_transforms(), batch_size=16, num_workers=0
    )
    batch_a, _ = next(iter(loaders["train_eval"]))
    batch_b, _ = next(iter(loaders["train_eval"]))
    assert torch.allclose(batch_a, batch_b), "train_eval must be deterministic"
