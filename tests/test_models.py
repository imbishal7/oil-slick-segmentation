import pytest
import torch

from oilslick.models import build_model


@pytest.mark.parametrize("attention", [False, True])
def test_output_shape_and_attention_maps(attention):
    model = build_model(attention, base_channels=8).eval()
    x = torch.randn(2, 2, 64, 64)
    logits, alphas = model(x, return_attention=True)

    assert logits.shape == (2, 1, 64, 64)
    if attention:
        # one gate per skip connection, coarse to fine, each at its skip's resolution
        assert [a.shape[-1] for a in alphas] == [8, 16, 32, 64]
        assert all(((a >= 0) & (a <= 1)).all() for a in alphas)
    else:
        assert alphas == []


def test_attention_adds_few_parameters():
    plain = sum(p.numel() for p in build_model(False).parameters())
    gated = sum(p.numel() for p in build_model(True).parameters())
    assert plain < gated < 1.1 * plain
