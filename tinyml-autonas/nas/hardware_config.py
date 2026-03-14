import yaml
from dataclasses import dataclass

@dataclass
class HardwareConfig:
    chip_id: str
    flash_kb: int
    sram_kb: int
    mhz: int
    has_fpu: bool
    supports_simd: bool
    
    # Optional field per schema standard practice, though not explicitly requested to be optional in Prompt 02
    architecture_family: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "HardwareConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Extract fields, using defaults for anything optional if missing
        return cls(
            chip_id=data["chip_id"],
            flash_kb=data["flash_kb"],
            sram_kb=data["sram_kb"],
            mhz=data["mhz"],
            has_fpu=data.get("has_fpu", False),
            supports_simd=data.get("supports_simd", False),
            architecture_family=data.get("architecture_family", "")
        )

    def validate(self):
        assert self.flash_kb >= 16, f"Flash must be >= 16 KB, got {self.flash_kb}"
        assert self.sram_kb >= 8, f"SRAM must be >= 8 KB, got {self.sram_kb}"
        assert self.mhz >= 1, f"Clock speed must be >= 1 MHz, got {self.mhz}"

    @property
    def max_model_weights_kb(self) -> float:
        return self.flash_kb * 0.80

    @property
    def max_peak_ram_kb(self) -> float:
        return self.sram_kb * 0.70

    def __repr__(self) -> str:
        return f"HardwareConfig(chip={self.chip_id}, flash={self.flash_kb}KB, sram={self.sram_kb}KB, {self.mhz}MHz)"
