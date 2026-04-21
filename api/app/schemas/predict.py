from pydantic import BaseModel, Field, ConfigDict
from typing import List

class PredictRequest(BaseModel):
    """Esquema de entrada para la petición de inferencia."""
    image_base64: str = Field(
        ..., 
        description="Cadena en formato Base64 de la imagen. Puede incluir el prefijo 'data:image/...;base64,'",
        example="iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAY..."
    )

class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_class": 7,
                "confidence": 0.985,
                "probabilities": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.985, 0.005, 0.0]
            }
        }
    )

    predicted_class: int = Field(..., description="Dígito clasificado (0-9)", example=7)
    confidence: float = Field(..., description="Certeza de la predicción (0.0 a 1.0)", example=0.985)
    probabilities: List[float] = Field(..., description="Distribución Softmax para las 10 clases")