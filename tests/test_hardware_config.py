"""Unit tests for nas.hardware_config.HardwareConfig."""
import pytest
from nas.hardware_config import HardwareConfig


@pytest.fixture
def hw():
    return HardwareConfig.from_yaml("config/hardware.yaml")


def test_loads_yaml(hw):
    """HardwareConfig.from_yaml should load correctly and populate fields."""
    assert hw.chip_id is not None
    assert hw.flash_kb > 0
    assert hw.sram_kb > 0
    assert hw.mhz > 0


def test_validate_fails_on_bad_flash():
    """Validation should raise AssertionError when flash_kb is too small."""
    bad_hw = HardwareConfig(
        chip_id="test", flash_kb=4, sram_kb=520, mhz=240,
        has_fpu=False, supports_simd=False
    )
    with pytest.raises(AssertionError, match="Flash must be >= 16"):
        bad_hw.validate()


def test_properties(hw):
    """max_model_weights_kb should be 80% of flash_kb."""
    expected = hw.flash_kb * 0.80
    assert hw.max_model_weights_kb == expected
