import logging
from fastapi import APIRouter, HTTPException
from app.schemas.predict import PredictRequest, PredictResponse
from app.services import image_service, model_service

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api/v1", tags=["Inferencia"])

@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    logger.info("Nueva petición de inferencia web")
    
    try:
        input_tensor = image_service.process_base64_image(request.image_base64)
        result_dict = model_service.predict_digit(input_tensor)
        
        return PredictResponse(
            predicted_class=result_dict["predicted_class"],
            confidence=result_dict["confidence"],
            probabilities=result_dict["probabilities"]
        )

    except ValueError as ve:
        logger.warning(f"Error de validación de datos (HTTP 400): {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except Exception as e:
        # Logueo de estado de variables locales para observabilidad
        shape_info = input_tensor.shape if 'input_tensor' in locals() else 'no procesado'
        logger.error(
            f"Error interno | input_shape: {shape_info} | error: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Error interno procesando la predicción.")