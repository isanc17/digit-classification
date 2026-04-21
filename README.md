# MNIST Microservice API - MLOps & Deployment

Microservicio de inferencia de alto rendimiento para la clasificación de dígitos manuscritos (MNIST), diseñado con principios de escalabilidad, MLOps y resiliencia de infraestructura.

## Arquitectura Tecnológica
* **Core ML:** TensorFlow / Keras (Legacy Keras Bypass para compatibilidad universal).
* **Backend:** FastAPI + Uvicorn (Asíncrono, validación estricta de esquemas).
* **Tracking & MLOps:** MLflow (Gestión de hiperparámetros y versionado de modelos).
* **Infraestructura:** AWS Cloud9 (EC2) / Entorno Virtualizado.

## Decisiones de Arquitectura
1.  **Defensive Preprocessing:** El pipeline aplica redimensionamiento forzado (28x28) y conversión matemática condicional (inversión de colores mediante umbral de media de píxeles) para absorber el "Domain Gap" entre imágenes sintéticas de Frontend y datos manuscritos de entrenamiento.
2.  **Version Mismatch Mitigation:** Implementación de la clase `FlexibleDense` mediante inyección de `custom_objects` para interceptar y neutralizar metadatos tóxicos (`quantization_config`) derivados del salto entre Keras 2 y Keras 3 en la nube.
3.  **Optimización de Memoria:** Uso exclusivo de `tensorflow-cpu` en el entorno de producción (`requirements.txt`) y predicción directa mediante llamadas a tensores (`model(tensor)`) en lugar del pesado método `.predict()`, reduciendo la latencia de inferencia y el footprint de memoria.

## Despliegue Rápido
```bash
# 1. Instalar dependencias optimizadas
pip install -r requirements.txt

# 2. Levantar el microservicio
uvicorn main:app --host 0.0.0.0 --port 8000