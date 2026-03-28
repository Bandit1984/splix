import json
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import websocket
from gymnasium import spaces


@dataclass
class SplixEnvConfig:
    bridge_url: str = "ws://127.0.0.1:8080/ai-bridge"
    arena_width: int = 80
    arena_height: int = 80
    opponent_count: int = 7
    max_steps: int = 800
    decision_interval_ms: int = 100
    obs_radius: int = 6
    game_mode: str = "default"


class SplixEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, config: SplixEnvConfig | None = None):
        super().__init__()
        self.config = config or SplixEnvConfig()
        self._obs_radius = self.config.obs_radius
        self._obs_tile_size = (self._obs_radius * 2 + 1) ** 2
        self._scalar_size = 8

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(self._obs_tile_size + self._scalar_size,),
            dtype=np.float32,
        )

        self._ws: websocket.WebSocket | None = None
        self._request_id = 1
        self._env_id: str | None = None

        self._connect()
        self._create_env()

    def _connect(self) -> None:
        self._ws = websocket.create_connection(self.config.bridge_url, timeout=15)
        hello = self._rpc("hello", {})
        if int(hello.get("protocol_version", 0)) != 1:
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
        local_tiles = np.asarray(obs["local_tiles"], dtype=np.float32)
        player = obs["player"]
        arena = obs["arena"]

        direction_map = {
            "right": 0.0,
            "down": 1.0,
            "left": 2.0,
            "up": 3.0,
            "paused": 4.0,
        }
        scalars = np.asarray(
            [
                float(player["x"]) / max(1.0, float(arena["width"])),
                float(player["y"]) / max(1.0, float(arena["height"])),
                direction_map.get(player["direction"], 4.0),
                float(player["score"]),
                float(player["kills"]),
                1.0 if player["dead"] else 0.0,
                1.0 if player["permanently_dead"] else 0.0,
                float(obs.get("step", 0)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([local_tiles, scalars], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self._env_id is None:
            raise RuntimeError("Missing env_id")
        payload: dict[str, Any] = {"env_id": self._env_id}
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
