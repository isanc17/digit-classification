from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager

from fastapi.responses import FileResponse
from fastapi import FastAPI

from app.services.model_service import load_model_instance
from app.routes.predict import router as predict_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Arrancando servidor de inferencia...")
    load_model_instance() 
    yield
    print("Apagando servidor, liberando recursos...")

# Inicializacion
app = FastAPI(
    title="MNIST Inferencia API",
    description="Microservicio E2E de clasificación de dígitos",
    version="1.0.0",
    lifespan=lifespan
)

# Middlewares 
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Observabilidad 
@app.get("/health", tags=["Sistema"])
async def health_check():
    """Endpoint para que Kubernetes o un Balanceador verifiquen que la API está viva."""
    return {"status": "ok", "service": "mnist-api"}

# Routing
app.include_router(predict_router)

# UI para PoC
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")