"""Search routes: POST /search, GET /search/{run_id}."""
import secrets
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from api.models import SearchRunInput, SearchRunResponse

router = APIRouter(prefix="/search", tags=["Search"])

# In-memory storage
_search_store: dict[str, dict] = {}


def _run_search_background(run_id: str, params: dict):
    """Background task that runs the NAS search."""
    from nas.controller import NASController
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    record = _search_store[run_id]
    record["status"] = "running"
    record["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        controller = NASController()
        ds = TensorDataset(torch.randn(40, 1, 40, 40), torch.randint(0, params["num_classes"], (40,)))
        dl = DataLoader(ds, batch_size=8)

        controller.config["trial_budget"] = params.get("trial_budget", 5)
        controller.config["num_classes"] = params["num_classes"]
        controller.config["epochs"] = 2  # Fast for background

        result = controller.run_search(dl, dl)

        record.update({
            "status": "completed",
            "trials_completed": result["trials_completed"],
            "best_accuracy": result["best_accuracy"],
            "best_latency_ms": result["best_latency_ms"],
            "best_model_size_kb": result["best_model_size_kb"],
            "best_candidate_id": result.get("best_candidate_id"),
            "llm_hints_used": result.get("llm_hints_used", []),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        record.update({
            "status": "failed",
            "error_message": str(e),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=SearchRunResponse)
def create_search(body: SearchRunInput, bg: BackgroundTasks):
    run_id = f"run_{secrets.token_hex(4)}"
    now = datetime.now(timezone.utc)
    record = {
        "run_id": run_id,
        "status": "queued",
        "hardware_id": body.hardware_id,
        "domain": body.domain,
        "trial_budget": body.trial_budget,
        "trials_completed": 0,
        "best_accuracy": None,
        "best_latency_ms": None,
        "best_model_size_kb": None,
        "best_candidate_id": None,
        "llm_hints_used": [],
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "error_message": None,
    }
    _search_store[run_id] = record
    bg.add_task(_run_search_background, run_id, body.model_dump())
    return record


@router.get("/{run_id}", response_model=SearchRunResponse)
def get_search(run_id: str):
    if run_id not in _search_store:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _search_store[run_id]
