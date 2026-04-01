"""Unit tests for nas.search_space.SearchSpace."""
import pytest
from nas.hardware_config import HardwareConfig
from nas.search_space import SearchSpace
from nas.layers import LayerConfig


@pytest.fixture
def hw():
    return HardwareConfig.from_yaml("config/hardware.yaml")


def test_prune_reduces_size(hw):
    """Using a prefer_depthwise hint should reduce the search space."""
    ss_full = SearchSpace(hw, [])
    ss_pruned = SearchSpace(hw, [{"hint": "prefer_depthwise_separable", "priority": 1}])
    assert ss_pruned.size < ss_full.size


def test_sample_returns_list(hw):
    """sample() should return a list of LayerConfig objects."""
    ss = SearchSpace(hw, [])
    candidate = ss.sample()
    assert isinstance(candidate, list)
    assert len(candidate) >= 2
    for item in candidate:
        assert isinstance(item, LayerConfig)
