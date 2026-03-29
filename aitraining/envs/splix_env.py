import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import websocket
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor

from ..config import RewardConfig, ConfigLoader
from .observation_contract import (
    flatten_observation,
    observation_bounds,
    validate_observation_bounds,
)


class SplixEnvConfig:
    """Configuration for Splix Gymnasium environment.
    
    Can be initialized from a RewardConfig object or defaults.
    """
    
    def __init__(
        self,
        bridge_url: str = "ws://127.0.0.1:8080/ai-bridge",
        reward_config: RewardConfig | None = None,
        global_seed: int | None = None,
        strict_observation_contract: bool = False,
    ):
        self.bridge_url = bridge_url
        self.global_seed = global_seed
        self.strict_observation_contract = strict_observation_contract
        
        # Load reward config (merge with environment params)
        if reward_config is None:
            reward_config = RewardConfig.default()
        
        self.arena_width = reward_config.arena_width
        self.arena_height = reward_config.arena_height
        self.opponent_count = reward_config.opponent_count
        self.max_steps = reward_config.max_steps
        self.decision_interval_ms = reward_config.decision_interval_ms
        self.obs_radius = reward_config.obs_radius
        self.game_mode = reward_config.game_mode
        
        # Store reward weights for bridge communication
        self.reward_score_weight = reward_config.score_weight
        self.reward_kill_weight = reward_config.kill_weight
        self.reward_death_penalty = reward_config.death_penalty
        self.reward_truncate_penalty = reward_config.truncate_penalty


class SplixEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, config: SplixEnvConfig | None = None):
        super().__init__()
        self.config = config or SplixEnvConfig()
        self._obs_radius = self.config.obs_radius
        self._obs_tile_size = (self._obs_radius * 2 + 1) ** 2
        self._scalar_size = 8
        self._obs_low: np.ndarray | None = None
        self._obs_high: np.ndarray | None = None

        self.action_space = spaces.Discrete(5)
        obs_low, obs_high = observation_bounds(self._obs_tile_size)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self._obs_tile_size + self._scalar_size,),
            dtype=np.float32,
        )
        self._obs_low = obs_low
        self._obs_high = obs_high

        self._ws: websocket.WebSocket | None = None
        self._request_id = 1
        self._env_id: str | None = None

        self._connect()
        self._create_env()

    def _connect(self) -> None:
        self._ws = websocket.create_connection(self.config.bridge_url, timeout=15)
        hello = self._rpc("hello", {})
        if int(hello.get("protocol_version", 0)) < 1:
            raise RuntimeError("Unsupported bridge protocol version")

    def _create_env(self) -> None:
        payload = {
            "arena_width": self.config.arena_width,
            "arena_height": self.config.arena_height,
            "opponent_count": self.config.opponent_count,
            "max_steps": self.config.max_steps,
            "decision_interval_ms": self.config.decision_interval_ms,
            "obs_radius": self.config.obs_radius,
            "game_mode": self.config.game_mode,
            "reward_score_weight": self.config.reward_score_weight,
            "reward_kill_weight": self.config.reward_kill_weight,
            "reward_death_penalty": self.config.reward_death_penalty,
            "reward_truncate_penalty": self.config.reward_truncate_penalty,
            "global_seed": self.config.global_seed,
        }
        response = self._rpc("create_env", payload)
        self._env_id = response["env_id"]

    def _rpc(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("Bridge socket is not connected")

        request_id = self._request_id
        self._request_id += 1
        message = {
            "id": request_id,
            "method": method,
            "payload": payload,
        }
        self._ws.send(json.dumps(message))

        while True:
            raw = self._ws.recv()
            data = json.loads(raw)
            if data.get("id") != request_id:
                continue
            if not data.get("ok", False):
                raise RuntimeError(data.get("error", "Unknown bridge error"))
            return data.get("payload", {})

    def _flatten_observation(self, obs: dict[str, Any]) -> np.ndarray:
        flattened = flatten_observation(
            obs,
            opponent_count=self.config.opponent_count,
            max_steps=self.config.max_steps,
        )
        if self.config.strict_observation_contract:
            validate_observation_bounds(
                flattened,
                self._obs_low if self._obs_low is not None else self.observation_space.low,
                self._obs_high if self._obs_high is not None else self.observation_space.high,
            )
        return flattened

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self._env_id is None:
            raise RuntimeError("Missing env_id")
        payload: dict[str, Any] = {"env_id": self._env_id}
        if seed is not None:
            payload["global_seed"] = int(seed)
        if options:
            payload.update(options)
        response = self._rpc("reset", payload)
        observation = self._flatten_observation(response["observation"])
        return observation, response.get("info", {})

    def step(self, action: int):
        if self._env_id is None:
            raise RuntimeError("Missing env_id")
        response = self._rpc(
            "step",
            {
                "env_id": self._env_id,
                "action": int(action),
            },
        )
        observation = self._flatten_observation(response["observation"])
        reward = float(response["reward"])
        terminated = bool(response["done"])
        truncated = bool(response["truncated"])
        info = response.get("info", {})
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self._ws and self._env_id:
            try:
                self._rpc("close_env", {"env_id": self._env_id})
            except Exception:
                pass
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._env_id = None
        self._ws = None


def wait_for_bridge(url: str, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            ws = websocket.create_connection(url, timeout=2)
            ws.send(json.dumps({"id": 1, "method": "ping", "payload": {}}))
            _ = ws.recv()
            ws.close()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Bridge is not reachable at {url}")


def make_recorded_env(config: SplixEnvConfig) -> gym.Env[np.ndarray, int]:
    """Create an env wrapped with RecordEpisodeStatistics for automatic episode metrics."""
    env: gym.Env[np.ndarray, int] = SplixEnv(config)
    env = RecordEpisodeStatistics(env)
    return env


def make_monitored_env(config: SplixEnvConfig) -> gym.Env[np.ndarray, int]:
    """Create a high-level wrapped env with RecordEpisodeStatistics and Monitor."""
    env = make_recorded_env(config)
    env = Monitor(env)
    return env
