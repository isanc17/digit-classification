import base64
import io
import logging
import numpy as np
from PIL import Image

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def process_base64_image(b64_str: str) -> np.ndarray:
    """Decodifica y preprocesa la imagen para la inferencia."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]

    try:
        image_bytes = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
    except Exception as e:
        logger.error(f"Error decodificando Base64: {e}")
        raise ValueError(f"Cadena Base64 inválida o corrupta.")

    image = image.resize((28, 28))
    image_array = np.array(image)

    # Observabilidad y mitigación del Domain Gap
    original_mean = np.mean(image_array)
    if original_mean > 127:
        image_array = 255 - image_array
        logger.info(f"Inversión de colores aplicada | Mean original: {original_mean:.1f}")
    else:
        logger.info(f"Sin inversión de colores | Mean original: {original_mean:.1f}")

    # Normalización
    img_normalized = image_array.astype("float32") / 255.0

    # Reshape explícito y limpio
    input_tensor = img_normalized.reshape(1, 28, 28, 1)

    return input_tensor