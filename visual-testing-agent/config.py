# Pydantic settings + env vars
from pathlib import Path
from pydantic_settings import BaseSettings


# Resolve .env from the project root (one level above this file)
_ENV_FILE = str(Path(__file__).parent.parent / ".env")


class Settings(BaseSettings):
    gemini_api_key: str        # GEMINI_API_KEY in .env
    output_dir: str = "output"
    model: str = "gemini-3.5-flash"

    model_config = {
        "env_file": _ENV_FILE,
        "extra": "ignore",     # silently ignore unknown env vars (e.g. FIGMA_TOKEN)
    }


settings = Settings()
