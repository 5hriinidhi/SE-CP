"""Pydantic v2 models matching api_contract.json schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ── Hardware ──────────────────────────────────────────────
class HardwareConfigInput(BaseModel):
    chip_id: str
    flash_kb: int
    sram_kb: int
    mhz: int
    architecture_family: Optional[str] = None
    supports_simd: bool = False
    has_fpu: bool = False


class HardwareConfigResponse(HardwareConfigInput):
    hardware_id: str
    created_at: datetime


# ── Search ────────────────────────────────────────────────
class SearchRunInput(BaseModel):
    hardware_id: str
    domain: str
    dataset_path: str
    num_classes: int = Field(ge=2)
    trial_budget: int = 50
    max_latency_ms: Optional[float] = None
    max_model_size_kb: Optional[float] = None
    target_accuracy: Optional[float] = None
    tuner_strategy: str = "TPE"
    llm_model: str = "gpt-4o"
    quantization: str = "int8"
    tags: List[str] = []


class SearchRunResponse(BaseModel):
    run_id: str
    status: str = "queued"
    hardware_id: str
    domain: str
    trial_budget: int
    trials_completed: int = 0
    best_accuracy: Optional[float] = None
    best_latency_ms: Optional[float] = None
    best_model_size_kb: Optional[float] = None
    best_candidate_id: Optional[str] = None
    llm_hints_used: List[str] = []
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


# ── LLM Hints ────────────────────────────────────────────
class LLMHintsInput(BaseModel):
    domain: str
    hardware_id: str
    task_description: Optional[str] = None
    llm_model: str = "gpt-4o"
    max_hints: int = 8


class HintItem(BaseModel):
    hint: str
    reason: str
    priority: int


class LLMHintsResponse(BaseModel):
    domain: str
    llm_model: str
    hints: List[HintItem] = []
    pruned_search_space_size: Optional[int] = None
    original_search_space_size: Optional[int] = None
    reduction_ratio: Optional[float] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None


# ── Simulator ─────────────────────────────────────────────
class LayerDefinition(BaseModel):
    type: str
    out_channels: Optional[int] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[str] = None
    activation: Optional[str] = None
    units: Optional[int] = None
    multiplier: Optional[float] = None
    input_shape: Optional[List[int]] = None


class ArchitectureDefinition(BaseModel):
    layers: List[LayerDefinition]


class SimulateInput(BaseModel):
    hardware_id: str
    architecture: ArchitectureDefinition


class PerLayerBreakdown(BaseModel):
    layer_index: int
    type: str
    latency_ms: float
    size_kb: float


class SimulationResultResponse(BaseModel):
    estimated_latency_ms: float
    estimated_model_size_kb: float
    estimated_peak_ram_kb: float
    feasibility_check_passed: bool
    constraint_violations: List[str] = []
    per_layer_breakdown: List[PerLayerBreakdown] = []
