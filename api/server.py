# FastAPI app + CORS middleware [Prompt 13]
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.hardware import router as hardware_router
from api.routes.search import router as search_router
from api.routes.llm import router as llm_router
from api.routes.simulator import router as simulator_router

app = FastAPI(
    title="TinyML AutoNAS API",
    version="1.0.0",
    description="LLM-Guided Neural Architecture Search for microcontrollers.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hardware_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(llm_router, prefix="/api/v1")
app.include_router(simulator_router, prefix="/api/v1")
