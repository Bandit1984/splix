from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentEnvConfig:
    bridge_url: str = "ws://127.0.0.1:8080/ai-bridge"
    arena_width: int = 80
    arena_height: int = 80
    opponent_count: int = 7
    max_steps: int = 800
    decision_interval_ms: int = 100
    obs_radius: int = 6
    game_mode: str = "default"


@dataclass
class ExperimentRewardConfig:
    score_weight: float = 0.01
    kill_weight: float = 1.0
    death_penalty: float = -2.0
    truncate_penalty: float = 0.0


@dataclass
class ExperimentTrainingConfig:
    total_timesteps: int = 500_000
    n_envs: int = 4
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    gamma: float = 0.99
    vector_mode: str = "pooled"
    preflight_check: bool = True
    use_wandb: bool = False
    checkpoint_freq: int = 10_000
    eval_freq: int = 10_000


@dataclass
class ExperimentScenarioConfig:
    name: str = "default"
    episodes: int = 10
    opponent_count: int = 7
    max_steps: int = 800


@dataclass
class ExperimentEvaluationConfig:
    baselines: list[str] = field(default_factory=lambda: ["ppo", "random", "heuristic"])
    scenarios: list[ExperimentScenarioConfig] = field(
        default_factory=lambda: [ExperimentScenarioConfig()]
    )


@dataclass
class ExperimentConfig:
    experiment_name: str = "splix-ppo"
    seed: int = 42
    env: ExperimentEnvConfig = field(default_factory=ExperimentEnvConfig)
    rewards: ExperimentRewardConfig = field(default_factory=ExperimentRewardConfig)
    training: ExperimentTrainingConfig = field(default_factory=ExperimentTrainingConfig)
    evaluation: ExperimentEvaluationConfig = field(default_factory=ExperimentEvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        raw_evaluation = raw.get("evaluation", {})
        raw_scenarios = raw_evaluation.get("scenarios", [{}])
        scenarios = [ExperimentScenarioConfig(**scenario) for scenario in raw_scenarios]

        return cls(
            experiment_name=raw.get("experiment_name", "splix-ppo"),
            seed=int(raw.get("seed", 42)),
            env=ExperimentEnvConfig(**raw.get("env", {})),
            rewards=ExperimentRewardConfig(**raw.get("rewards", {})),
            training=ExperimentTrainingConfig(**raw.get("training", {})),
            evaluation=ExperimentEvaluationConfig(
                baselines=list(raw_evaluation.get("baselines", ["ppo", "random", "heuristic"])),
                scenarios=scenarios,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    def __init__(self, workspace_root: str | Path, experiments_dir: str = "aitraining/experiments/runs"):
        self.workspace_root = Path(workspace_root)
        self.runs_root = self.workspace_root / experiments_dir
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def get_git_metadata(self) -> dict[str, Any]:
        def _run(cmd: list[str]) -> str:
            result = subprocess.run(
                cmd,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip()

        commit = _run(["git", "rev-parse", "HEAD"]) or "unknown"
        short_commit = _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
        status = _run(["git", "status", "--porcelain"])
        return {
            "commit": commit,
            "short_commit": short_commit,
            "branch": branch,
            "dirty": bool(status),
        }

    def create_run_directory(self, experiment_name: str, git_meta: dict[str, Any]) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}-{git_meta.get('short_commit', 'unknown')}-{experiment_name}"
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "models").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        return run_dir

    def write_manifest(self, run_dir: Path, config: ExperimentConfig, git_meta: dict[str, Any]) -> None:
        manifest = {
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "run_dir": str(run_dir),
            "config": config.to_dict(),
            "git": git_meta,
        }
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config.to_dict(), f, sort_keys=False)

    def write_metrics(self, run_dir: Path, metrics: dict[str, Any], file_name: str = "metrics.json") -> None:
        with (run_dir / file_name).open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
