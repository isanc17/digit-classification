FROM python:3.10-slim

# Variables de Entorno de Python en Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# directorio
WORKDIR /app

COPY requirements.txt .

# Instalación de dependencias del sistema y de Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Usuario no-root para evitar privilege escalation
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app

# Copiar  monorepo 
COPY . .

USER appuser

# Puerto del contenedor
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]