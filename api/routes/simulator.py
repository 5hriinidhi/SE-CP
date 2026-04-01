"""Simulator routes: POST /simulator/estimate."""
from fastapi import APIRouter, HTTPException
from api.models import SimulateInput, SimulationResultResponse

router = APIRouter(prefix="/simulator", tags=["Simulator"])


@router.post("/estimate", response_model=SimulationResultResponse)
def estimate_architecture(body: SimulateInput):
    from nas.hardware_config import HardwareConfig
    from nas.simulator import LatencySimulator
    from nas.layers import LayerConfig

    try:
        hw = HardwareConfig.from_yaml("config/hardware.yaml")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Hardware config error: {e}")

    # Convert API LayerDefinitions to nas LayerConfigs
    layer_configs = []
    for layer in body.architecture.layers:
        lc = LayerConfig(
            layer_type=layer.type,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size or 3,
            stride=layer.stride or 1,
            units=layer.units,
        )
        layer_configs.append(lc)

    sim = LatencySimulator(hw)
    result = sim.estimate(layer_configs)
    return result
