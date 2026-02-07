"""GEAK Online Service Configuration"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    external_base_url: str | None = Field(
        default=None,
        description="External base URL for download links (e.g., https://geak.example.com)"
    )
    
    # SaFE Platform
    safe_api_base: str = Field(..., description="SaFE platform API base URL")
    safe_system_api_key: str | None = Field(
        default=None,
        description="System-level SaFE API key for background task status checks"
    )
    
    # GEAK Repository
    geak_repo_url: str = Field(
        default="https://github.com/AMD-AGI/GEAK.git",
        description="GEAK repository URL"
    )
    geak_branch: str = Field(
        default="geak_online",
        description="GEAK repository branch"
    )
    
    # Database
    database_path: str = Field(
        default="./data/geak.db",
        description="SQLite database path"
    )
    
    # Storage
    nfs_base_path: str = Field(
        default="/wekafs/geak",
        description="NFS base path for task storage"
    )
    
    # Default Runtime Configuration
    default_image: str = Field(
        default="harbor.amd.com/rocm/mini-swe-agent:latest",
        description="Default Docker image"
    )
    default_gpu_count: int = Field(default=1, description="Default GPU count")
    default_timeout: int = Field(default=3600, description="Default timeout in seconds")
    default_cpu: int = Field(default=4, description="Default CPU cores")
    default_memory: str = Field(default="16Gi", description="Default memory")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
