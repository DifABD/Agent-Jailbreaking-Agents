"""Configuration management for the Agent Jailbreak Research system."""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    huggingface_api_token: Optional[str] = Field(None, env="HUGGINGFACE_API_TOKEN")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./data/experiments.db", env="DATABASE_URL")
    
    # Application Settings
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_turns: int = Field(7, env="MAX_TURNS")
    
    # Default Models
    default_persuader_model: str = Field("gpt-4o", env="DEFAULT_PERSUADER_MODEL")
    default_persuadee_model: str = Field("gpt-4o", env="DEFAULT_PERSUADEE_MODEL")
    default_judge_model: str = Field("llama-guard-2-8b", env="DEFAULT_JUDGE_MODEL")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(1, env="API_WORKERS")
    
    # Dashboard Configuration
    dashboard_host: str = Field("0.0.0.0", env="DASHBOARD_HOST")
    dashboard_port: int = Field(8501, env="DASHBOARD_PORT")
    
    # Experiment Configuration
    experiment_timeout: int = Field(300, env="EXPERIMENT_TIMEOUT")
    max_concurrent_experiments: int = Field(5, env="MAX_CONCURRENT_EXPERIMENTS")
    validation_sample_rate: float = Field(0.15, env="VALIDATION_SAMPLE_RATE")
    
    # Model Configuration
    model_temperature: float = Field(0.7, env="MODEL_TEMPERATURE")
    model_max_tokens: int = Field(1000, env="MODEL_MAX_TOKENS")
    model_timeout: int = Field(30, env="MODEL_TIMEOUT")
    
    # Security
    secret_key: str = Field("dev-secret-key", env="SECRET_KEY")
    allowed_hosts: str = Field("localhost,127.0.0.1", env="ALLOWED_HOSTS")
    
    # Monitoring (Optional)
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings