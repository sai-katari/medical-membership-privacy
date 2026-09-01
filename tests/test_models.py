"""Model building and regime freezing checks.

No data download or GPU required. Runs in a few seconds.
Run before any training: pytest tests/test_models.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.models.resnet import build_resnet18, get_parameter_groups
from src.models.model_factory import build_model

NUM_CLASSES = 7

_CFG = {
    "lr_scratch":          3e-4,
    "lr_frozen_head":      1e-3,
    "lr_partial_backbone": 1e-4,
    "lr_partial_head":     1e-3,
    "lr_full_backbone":    1e-4,
    "lr_full_head":        1e-3,
}


def test_scratch_all_trainable():
    model = build_resnet18("scratch", NUM_CLASSES)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen == [], f"Scratch should have no frozen params; found: {frozen}"


def test_frozen_only_head_trainable():
    model = build_resnet18("frozen", NUM_CLASSES)
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_partial_correct_layers():
    model = build_resnet18("partial", NUM_CLASSES)
    for name, param in model.named_parameters():
        should_train = name.startswith("layer4.") or name.startswith("fc.")
        if should_train:
            assert param.requires_grad, f"{name} should be trainable in partial"
        else:
            assert not param.requires_grad, f"{name} should be frozen in partial"


def test_full_all_trainable():
    model = build_resnet18("full", NUM_CLASSES)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen == [], f"Full FT should have no frozen params; found: {frozen}"


@pytest.mark.parametrize("regime", ["scratch", "frozen", "partial", "full"])
def test_forward_output_shape(regime):
    model = build_resnet18(regime, NUM_CLASSES)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def _trainable(regime):
    model = build_resnet18(regime, NUM_CLASSES)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_trainable_param_ordering():
    assert _trainable("frozen")  < _trainable("partial"), "frozen < partial"
    assert _trainable("partial") < _trainable("full"),    "partial < full"
    assert _trainable("full")    == _trainable("scratch"), "full == scratch (total)"


def test_scratch_single_param_group():
    model = build_resnet18("scratch", NUM_CLASSES)
    groups = get_parameter_groups(model, "scratch", _CFG)
    assert len(groups) == 1
    assert groups[0]["lr"] == _CFG["lr_scratch"]


def test_frozen_single_param_group():
    model = build_resnet18("frozen", NUM_CLASSES)
    groups = get_parameter_groups(model, "frozen", _CFG)
    assert len(groups) == 1
    assert groups[0]["lr"] == _CFG["lr_frozen_head"]


def test_partial_two_param_groups():
    model = build_resnet18("partial", NUM_CLASSES)
    groups = get_parameter_groups(model, "partial", _CFG)
    assert len(groups) == 2
    lrs = {g["lr"] for g in groups}
    assert _CFG["lr_partial_backbone"] in lrs
    assert _CFG["lr_partial_head"] in lrs


def test_full_two_param_groups():
    model = build_resnet18("full", NUM_CLASSES)
    groups = get_parameter_groups(model, "full", _CFG)
    assert len(groups) == 2
    lrs = {g["lr"] for g in groups}
    assert _CFG["lr_full_backbone"] in lrs
    assert _CFG["lr_full_head"] in lrs


def test_invalid_regime_raises():
    with pytest.raises(ValueError, match="regime must be one of"):
        build_resnet18("finetune", NUM_CLASSES)


def test_invalid_arch_raises():
    with pytest.raises(ValueError, match="not supported"):
        build_model("vit", "scratch", NUM_CLASSES)
