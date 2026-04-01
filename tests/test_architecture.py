"""Unit tests for nas.architecture.Architecture."""
import pytest
import torch
from nas.layers import LayerConfig
from nas.architecture import Architecture


@pytest.fixture
def model():
    layers = [
        LayerConfig("DepthwiseSepConv", out_channels=16, kernel_size=3, stride=1, padding="same"),
        LayerConfig("DepthwiseSepConv", out_channels=32, kernel_size=3, stride=1, padding="same"),
    ]
    return Architecture(layers, num_classes=10, in_channels=1)


def test_forward_shape(model):
    """Output shape should be [batch, num_classes]."""
    x = torch.randn(4, 1, 40, 40)
    out = model(x)
    assert out.shape == (4, 10)


def test_param_count_positive(model):
    """param_count() should return a positive integer."""
    count = model.param_count()
    assert count > 0
