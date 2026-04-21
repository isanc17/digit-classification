import logging
import tensorflow as tf
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# evitando posibles conflictos de versiones de keras
class FlexibleDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

custom_objects = {
    "Dense": FlexibleDense,
    "keras.layers.Dense": FlexibleDense
}

_model = None

def load_model_instance(model_path: str = None):
    global _model
    if model_path is None:
        model_path = settings.MODEL_PATH
    if _model is None:
        try:
            logger.info(f"Cargando pesos desde {model_path}...")
            _model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects
            )
            logger.info("Modelo cargado correctamente.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise RuntimeError(f"Fallo en inicialización de Keras: {e}")
    return _model

def predict_digit(input_tensor: np.ndarray) -> dict:
    """Ejecuta la predicción y devuelve un desglose completo de probabilidades."""
    if _model is None:
        logger.error("Intento de predicción sin modelo inicializado.")
        raise RuntimeError("El modelo no está en memoria.")
    
    # Inferencia optimizada
    preds = _model(input_tensor, training=False).numpy()
    
    probabilities = [float(p) for p in preds[0]]  # Convertir a float nativos de Python
    predicted_class = int(np.argmax(probabilities))
    confidence = float(max(probabilities))
    
    logger.info(f"Inferencia exitosa | Clase: {predicted_class} | Confianza: {confidence:.2%}")
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }