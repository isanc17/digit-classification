from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "MNIST Inferencia API"
    MODEL_PATH: str = "models/modelo_mnist.keras"
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    USE_MLFLOW_REGISTRY: bool = True
    MODEL_ALIAS: str = "champion"
    MLFLOW_TRACKING_URI: str = "http://localhost:8080"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()