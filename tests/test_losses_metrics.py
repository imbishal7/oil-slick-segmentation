import math

import torch

from oilslick.losses import FocalLoss, SoftDiceLoss, build_loss
from oilslick.metrics import Confusion, per_image_counts, threshold_counts


def test_dice_is_zero_for_confident_correct_prediction():
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:5, 2:5] = 1
    logits = (target * 2 - 1) * 20
    assert SoftDiceLoss()(logits, target).item() < 1e-3


def test_focal_alpha_weights_positive_class():
    target = torch.ones(1, 1, 4, 4)
    logits = torch.zeros_like(target)
    low, high = FocalLoss(alpha=0.25)(logits, target), FocalLoss(alpha=0.75)(logits, target)
    assert torch.isclose(high, 3 * low)


def test_focal_with_gamma_zero_is_weighted_bce():
    target = (torch.rand(1, 1, 8, 8) > 0.5).float()
    logits = torch.randn_like(target)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert torch.isclose(FocalLoss(gamma=0, alpha=0.5)(logits, target), 0.5 * bce)


def test_build_loss_names():
    assert isinstance(build_loss("bce"), torch.nn.BCEWithLogitsLoss)
    assert build_loss("focal_dice")(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)) >= 0


def test_confusion_metrics():
    m = Confusion(tp=2, fp=1, fn=1, tn=6).metrics()
    assert m["iou"] == 0.5
    assert m["dice"] == 4 / 6
    assert m["precision"] == m["recall"] == 2 / 3
    assert m["miou"] == (0.5 + 6 / 8) / 2
    assert m["pixel_fpr"] == 1 / 7
    assert math.isnan(Confusion(tn=5).metrics()["iou"])


def test_counts_per_image_and_per_threshold():
    target = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool).reshape(1, 1, 2, 2)
    prob = torch.tensor([[0.9, 0.4], [0.6, 0.1]]).reshape(1, 1, 2, 2)
    assert per_image_counts(prob >= 0.5, target).tolist() == [[1, 1, 1, 1]]
    counts = threshold_counts(prob, target, torch.tensor([0.05, 0.95]))
    assert counts.tolist() == [[2, 2, 0, 0], [0, 0, 2, 2]]
