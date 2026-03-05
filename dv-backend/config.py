"""
DeepVariance Configuration Module
Centralized configuration management with environment variable validation
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class Config:
    """Application configuration with validation"""

    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Required configuration
        self.GROQ_API_KEY: str = self._get_required_env("GROQ_API_KEY")
        self.DATABASE_URL: str = self._get_required_env("DATABASE_URL")

        # API Configuration
        self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT: int = int(os.getenv("API_PORT", "8000"))

        # CORS Configuration
        cors_origins = os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
        self.CORS_ORIGINS: list[str] = [origin.strip()
                                        for origin in cors_origins.split(",")]

        # Storage Paths
        self.DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
        self.MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "./models"))
        self.RESULTS_DIR: Path = Path(os.getenv("RESULTS_DIR", "./results"))

        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)

        # Training Configuration
        self.DEFAULT_MAX_ITERATIONS: int = int(
            os.getenv("DEFAULT_MAX_ITERATIONS", "10"))
        self.DEFAULT_TARGET_ACCURACY: float = float(
            os.getenv("DEFAULT_TARGET_ACCURACY", "1.0"))
        self.DEFAULT_DEVICE: str = os.getenv("DEFAULT_DEVICE", "cpu")

        # Logging Configuration
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.SQL_ECHO: bool = os.getenv("SQL_ECHO", "false").lower() == "true"

        # Job resumer / submission settings
        # How often (seconds) the background resumer should poll for pending jobs
        self.RESUMER_POLL_INTERVAL: int = int(
            os.getenv("RESUMER_POLL_INTERVAL", "30"))
        # How many attempts to try submitting a job before marking as failed
        self.MAX_SUBMISSION_ATTEMPTS: int = int(
            os.getenv("MAX_SUBMISSION_ATTEMPTS", "5"))
        # ML pipeline service URL (used for AutoML submission)
        self.ML_PIPELINE_URL: str = os.getenv(
            "ML_PIPELINE_URL", "http://localhost:8001")

    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(
                f"❌ Required environment variable '{key}' is not set!\n"
                f"Please add it to your .env file or set it in your environment."
            )
        return value

    def log_configuration(self):
        """Log configuration details (safe for production - hides sensitive data)"""
        print("\n" + "=" * 60)
        print("🚀 DeepVariance Configuration")
        print("=" * 60)

        # API Keys (masked)
        print(f"✓ GROQ_API_KEY: {self._mask_api_key(self.GROQ_API_KEY)}")

        # Database (mask password)
        print(f"✓ DATABASE_URL: {self._mask_database_url(self.DATABASE_URL)}")

        # API Configuration
        print(f"✓ API Server: {self.API_HOST}:{self.API_PORT}")
        print(f"✓ CORS Origins: {', '.join(self.CORS_ORIGINS)}")

        # Storage
        print(f"✓ Data Directory: {self.DATA_DIR.absolute()}")
        print(f"✓ Models Directory: {self.MODELS_DIR.absolute()}")
        print(f"✓ Results Directory: {self.RESULTS_DIR.absolute()}")

        # Training Defaults
        print(f"✓ Max Iterations: {self.DEFAULT_MAX_ITERATIONS}")
        print(f"✓ Target Accuracy: {self.DEFAULT_TARGET_ACCURACY}")
        print(f"✓ Default Device: {self.DEFAULT_DEVICE}")

        # Logging
        print(f"✓ Log Level: {self.LOG_LEVEL}")
        print(f"✓ SQL Echo: {self.SQL_ECHO}")

        print("=" * 60 + "\n")

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """Mask API key for logging (show first 8 chars)"""
        if len(api_key) <= 8:
            return "***"
        return f"{api_key[:8]}...{api_key[-4:]}"

    @staticmethod
    def _mask_database_url(url: str) -> str:
        """Mask database password in URL"""
        # Format: postgresql://username:password@host:port/database
        if "://" in url and "@" in url:
            protocol, rest = url.split("://", 1)
            if "@" in rest:
                creds, host_part = rest.split("@", 1)
                if ":" in creds:
                    username, _ = creds.split(":", 1)
                    return f"{protocol}://{username}:***@{host_part}"
        return url


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def validate_config() -> bool:
    """
    Validate configuration at startup
    Returns True if valid, raises error if invalid
    """
    try:
        config = get_config()
        config.log_configuration()
        return True
    except ValueError as e:
        print(f"\n❌ Configuration Error:\n{e}\n", file=sys.stderr)
        print("Please check your .env file and ensure all required variables are set.\n", file=sys.stderr)
        raise
