import requests
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Cargar una imagen real de MNIST
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Tomar la primera imagen
img_array = x_test[7]
real_label = y_test[7]

img = Image.fromarray(img_array.astype('uint8'), mode='L')
buffer = io.BytesIO()
img.save(buffer, format='PNG')
img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"image_base64": img_base64}
)

print(f"Etiqueta real:  {real_label}")

data = response.json()

print(f"Predicción API: {data['predicted_class']}")
print(f"Confianza:      {data['confidence']:.2%}")
