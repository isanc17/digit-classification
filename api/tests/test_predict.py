import requests
import base64
from PIL import Image, ImageDraw
import io
import numpy as np

# Crear imagen de prueba con el dígito "3"
img = Image.new('L', (28, 28), color=0)
draw = ImageDraw.Draw(img)
draw.text((8, 8), "3", fill=255)

# Convertir a base64
buffer = io.BytesIO()
img.save(buffer, format='PNG')
img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Hacer la petición
response = requests.post(
    "http://localhost:8000/predict",
    json={"image_base64": img_base64}
)

print(f"Status: {response.status_code}")
print(f"Respuesta: {response.json()}")
