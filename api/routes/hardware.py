"""Hardware routes: POST /hardware, GET /hardware."""
import secrets
from datetime import datetime, timezone
from fastapi import APIRouter, status
from api.models import HardwareConfigInput, HardwareConfigResponse

router = APIRouter(prefix="/hardware", tags=["Hardware"])

# In-memory storage
_hardware_store: dict[str, dict] = {}


@router.post("", status_code=status.HTTP_201_CREATED, response_model=HardwareConfigResponse)
def register_hardware(body: HardwareConfigInput):
    hw_id = f"hw_{secrets.token_hex(4)}"
    record = {
        **body.model_dump(),
        "hardware_id": hw_id,
        "created_at": datetime.now(timezone.utc),
    }
    _hardware_store[hw_id] = record
    return record


@router.get("")
def list_hardware():
    items = list(_hardware_store.values())
    return {"items": items, "total": len(items)}
