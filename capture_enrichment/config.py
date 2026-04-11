from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    gemini_api_key: str
    s3_results_bucket: str = ""
    chunk_duration_sec: int = 30
    chunk_overlap_sec: int = 5
    video_downsample_fps: int = 5
    video_downsample_height: int = 480
    telemetry_resolution_sec: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
