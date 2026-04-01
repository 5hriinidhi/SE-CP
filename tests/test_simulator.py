"""Unit tests for nas.simulator.LatencySimulator."""
import pytest
from nas.hardware_config import HardwareConfig
from nas.simulator import LatencySimulator
from nas.layers import LayerConfig


@pytest.fixture
def hw():
    return HardwareConfig.from_yaml("config/hardware.yaml")


@pytest.fixture
def sim(hw):
    return LatencySimulator(hw)


def test_feasible_small_arch(sim):
    """A small 2-layer DSConv architecture should be feasible."""
    layers = [
        LayerConfig("DepthwiseSepConv", out_channels=16, kernel_size=3),
        LayerConfig("DepthwiseSepConv", out_channels=32, kernel_size=3),
    ]
    result = sim.estimate(layers)
    assert result["feasibility_check_passed"] is True
    assert result["estimated_model_size_kb"] > 0


def test_infeasible_huge_arch(sim):
    """10 Conv2D(512) layers should exceed hardware limits."""
    layers = [LayerConfig("Conv2D", out_channels=512, kernel_size=3)] * 10
    result = sim.estimate(layers)
    assert result["feasibility_check_passed"] is False


def test_violation_message(sim):
    """When infeasible, constraint_violations should be non-empty strings."""
    layers = [LayerConfig("Conv2D", out_channels=512, kernel_size=3)] * 10
    result = sim.estimate(layers)
    assert len(result["constraint_violations"]) > 0
    for v in result["constraint_violations"]:
        assert isinstance(v, str)
        assert len(v) > 0
