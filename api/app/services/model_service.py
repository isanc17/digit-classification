import mlflow
import mlflow.keras
import logging
import tensorflow as tf
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
_active_alias = None

def load_model_instance():
    global _model, _active_alias
    if _model is not None:
        return _model

    use_registry = settings.USE_MLFLOW_REGISTRY
    logger.info(f"Registry {use_registry}")

    if use_registry:
        _active_alias = settings.MODEL_ALIAS
        try:
            uri = f"models:/MNIST_Classifier_E2E@{_active_alias}"
            logger.info(f"Cargando desde MLflow Registry: {uri}")
            _model = mlflow.keras.load_model(uri)
            logger.info(f"Modelo @{_active_alias} cargado desde Registry.")
            return _model
        except Exception as e:
            logger.warning(f"Registry falló, usando fallback local: {e}")

    # Fallback o modo local
    _active_alias = "local"
    logger.info(f"Cargando modelo local desde {settings.MODEL_PATH}...")
    _model = tf.keras.models.load_model(
        settings.MODEL_PATH,
        compile=False,
        custom_objects=custom_objects
    )
    logger.info("Modelo local cargado correctamente.")
    return _model

def predict_digit(input_tensor: np.ndarray) -> dict:
    if _model is None:
        raise RuntimeError("El modelo no está en memoria.")
    
    preds = _model(input_tensor, training=False).numpy()
    probabilities = [float(p) for p in preds[0]]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(max(probabilities))
    
    logger.info(f"Inferencia | Clase: {predicted_class} | Confianza: {confidence:.2%} | Modelo: {_active_alias}")
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "model_alias": _active_alias
    }