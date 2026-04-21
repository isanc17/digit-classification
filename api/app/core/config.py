from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "MNIST Inferencia API"
    MODEL_PATH: str = "models/modelo_mnist.keras"
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    
    class Config:
        env_file = ".env"

settings = Settings()