"""Configuration management for AutoQA."""

import os
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field

from .models import ExplorationConfig


class UserConfig(BaseModel):
    """Configuration for a test user."""
    email: str
    password_env: Optional[str] = None
    storage_state_path: Optional[str] = None


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "claude"
    model: str = "claude-sonnet-4-20250514"
    api_key_env: str = "ANTHROPIC_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 4096
    failover_chain: list[str] = Field(
        default_factory=lambda: ["claude", "openrouter", "openai", "qwen"]
    )
    max_retries: int = 3


class RealTimeConfig(BaseModel):
    """Real-time testing configuration."""
    enabled: bool = True
    max_propagation_ms: int = 3000
    websocket_patterns: list[str] = Field(default_factory=list)


class AssertionConfig(BaseModel):
    """Assertion configuration."""
    dom_enabled: bool = True
    visual_enabled: bool = True
    visual_threshold: float = 0.01
    semantic_enabled: bool = True


class ExecutionConfig(BaseModel):
    """Test execution configuration."""
    parallel_workers: int = 4
    retry_count: int = 2
    screenshot_on_failure: bool = True


class ReportingConfig(BaseModel):
    """Reporting configuration."""
    allure_enabled: bool = True
    output_dir: str = "./reports"


class AppConfig(BaseModel):
    """Target application configuration."""
    base_url: str
    login_url: Optional[str] = None


class AutoQAConfig(BaseModel):
    """Main AutoQA configuration."""
    app: AppConfig
    users: dict[str, UserConfig] = Field(default_factory=dict)
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    real_time: RealTimeConfig = Field(default_factory=RealTimeConfig)
    assertions: AssertionConfig = Field(default_factory=AssertionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    test_generation: dict[str, Any] = Field(default_factory=lambda: {
        "coverage_target": "exhaustive",
        "include_negative_tests": True,
        "max_tests_per_flow": 20
    })


def load_config(config_path: Optional[str] = None) -> AutoQAConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches default locations.

    Returns:
        AutoQAConfig instance.
    """
    if config_path is None:
        # Search default locations
        search_paths = [
            Path("autoqa.config.yaml"),
            Path("autoqa.config.yml"),
            Path(".autoqa.yaml"),
            Path.home() / ".autoqa.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return AutoQAConfig(**data)

    # Return default config (requires app.base_url to be set)
    raise FileNotFoundError(
        "No configuration file found. Create autoqa.config.yaml with at least:\n"
        "app:\n  base_url: https://your-app.com"
    )


def create_default_config(base_url: str, output_path: str = "autoqa.config.yaml") -> AutoQAConfig:
    """Create a default configuration file.

    Args:
        base_url: The target application URL.
        output_path: Where to save the config file.

    Returns:
        The created AutoQAConfig.
    """
    config = AutoQAConfig(
        app=AppConfig(base_url=base_url),
        users={
            "user_a": UserConfig(
                email="tester.a@example.com",
                password_env="TEST_USER_A_PASSWORD"
            ),
            "user_b": UserConfig(
                email="tester.b@example.com",
                password_env="TEST_USER_B_PASSWORD"
            )
        }
    )

    # Save to file
    with open(output_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    return config
