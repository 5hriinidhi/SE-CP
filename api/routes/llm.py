"""LLM Advisor routes: POST /llm/hints."""
import time
from fastapi import APIRouter, HTTPException
from api.models import LLMHintsInput, LLMHintsResponse

router = APIRouter(prefix="/llm", tags=["LLM Advisor"])


@router.post("/hints", response_model=LLMHintsResponse)
def get_llm_hints(body: LLMHintsInput):
    from nas.llm_advisor import LLMAdvisor
    from nas.hardware_config import HardwareConfig
    from nas.search_space import SearchSpace

    start = time.time()
    try:
        hw = HardwareConfig.from_yaml("config/hardware.yaml")
        advisor = LLMAdvisor(model=body.llm_model)
        hints = advisor.get_hints(domain=body.domain, hw=hw)
    except Exception as e:
        # Return empty hints on LLM failure (graceful degradation)
        hints = []

    elapsed_ms = (time.time() - start) * 1000

    # Compute search space reduction if hints were received
    original_size = None
    pruned_size = None
    reduction = None
    try:
        hw = HardwareConfig.from_yaml("config/hardware.yaml")
        ss_full = SearchSpace(hw, [])
        original_size = len(ss_full.candidates)
        ss_pruned = SearchSpace(hw, hints)
        pruned_size = len(ss_pruned.candidates)
        reduction = round(1 - (pruned_size / original_size), 3) if original_size > 0 else 0.0
    except Exception:
        pass

    return LLMHintsResponse(
        domain=body.domain,
        llm_model=body.llm_model,
        hints=[{"hint": h["hint"], "reason": h.get("reason", ""), "priority": h.get("priority", 3)} for h in hints],
        pruned_search_space_size=pruned_size,
        original_search_space_size=original_size,
        reduction_ratio=reduction,
        tokens_used=None,
        latency_ms=round(elapsed_ms, 1),
    )
