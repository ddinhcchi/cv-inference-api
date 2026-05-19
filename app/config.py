from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    device: str = "cpu"
    model_weights: str = "yolov8n.pt"
    default_conf: float = 0.4
    max_image_side: int = 1920
    api_key: str = ""
    allowed_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
