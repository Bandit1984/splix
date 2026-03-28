"""Reward configuration loader for Splix environments."""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML not found. Install it with: pip install pyyaml"
    )


@dataclass
class RewardConfig:
    """Reward configuration for environment training."""
    score_weight: float = 0.01
    kill_weight: float = 1.0
    death_penalty: float = -2.0
    truncate_penalty: float = 0.0
    
    # Environment parameters
    arena_width: int = 80
    arena_height: int = 80
    opponent_count: int = 7
    max_steps: int = 800
    decision_interval_ms: int = 100
    obs_radius: int = 6
    game_mode: str = "default"
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "RewardConfig":
        """Load config from YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            RewardConfig instance populated from YAML
            
        Raises:
            FileNotFoundError: If yaml_path doesn't exist
            ValueError: If YAML is malformed
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        return cls(**data)
    
    @classmethod
    def default(cls) -> "RewardConfig":
        """Get default config (no YAML file needed)."""
        return cls()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (useful for logging/debugging)."""
        return asdict(self)
    
    def __repr__(self) -> str:
        """Pretty string representation."""
        lines = [f"{self.__class__.__name__}("]
        for key, value in asdict(self).items():
            lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)


class ConfigLoader:
    """Utility class for loading configs with fallback support."""
    
    @staticmethod
    def load_rewards(path: str | Path | None) -> RewardConfig:
        """Load reward config with fallback to default.
        
        Args:
            path: Path to YAML file, or None for default config
            
        Returns:
            RewardConfig instance
        """
        if path is None:
            return RewardConfig.default()
        
        try:
            return RewardConfig.from_yaml(path)
        except Exception as e:
            print(f"⚠️  Failed to load config from {path}: {e}")
            print("   Falling back to default config")
            return RewardConfig.default()
