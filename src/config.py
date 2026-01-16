"""Configuration settings for Mriia AI Tutor."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Key for Lapathon LLMs
    lapathon_api_key: str = ""
    
    # LLM API Configuration
    llm_base_url: str = "http://146.59.127.106:4000"
    
    # Model names
    lapa_model: str = "lapa"
    mamay_model: str = "mamay"
    lapa_function_calling_model: str = "lapa-function-calling"
    embedding_model: str = "text-embedding-qwen"
    
    # Phoenix Tracing
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "mriia-tutor"
    
    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    
    # Data paths
    data_dir: str = "Groke-Lapa/Lapathon2026_Mriia_public_files"
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    class Config:
        # Look for .env file in the project root (Groke-Lapa directory)
        env_file = str(Path(__file__).parent.parent / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def pages_parquet_path(self) -> Path:
        return Path(self.data_dir) / "text-embedding-qwen" / "pages_for_hackathon.parquet"
    
    @property
    def toc_parquet_path(self) -> Path:
        return Path(self.data_dir) / "text-embedding-qwen" / "toc_for_hackathon_with_subtopics.parquet"
    
    @property
    def questions_parquet_path(self) -> Path:
        return Path(self.data_dir) / "lms_questions_dev.parquet"
    
    @property
    def scores_parquet_path(self) -> Path:
        return Path(self.data_dir) / "benchmark_scores.parquet"
    
    @property
    def absences_parquet_path(self) -> Path:
        return Path(self.data_dir) / "benchmark_absences.parquet"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
