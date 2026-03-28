"""WebSocket connection pooling for efficient multi-env training."""

import json
from typing import Any
import websocket

from ..utils import debug_print


class WebSocketPool:
    """Manages a single WebSocket connection for multiple environments.
    
    This allows multiple environments to share a single connection, reducing
    overhead and improving training throughput. Each env gets a unique env_id
    for RPC routing.
    
    Usage:
        pool = WebSocketPool("ws://127.0.0.1:8080/ai-bridge")
        env_id_1 = pool.create_env()
        env_id_2 = pool.create_env()
        obs1, _ = pool.reset(env_id_1)
        obs2, _ = pool.reset(env_id_2)
        # Now use env_id_1 and env_id_2 with step() etc
    """
    
    def __init__(self, bridge_url: str, timeout: float = 15.0):
        """Initialize pool with a bridge URL.
        
        Args:
            bridge_url: WebSocket URL to Deno bridge
            timeout: Connection timeout in seconds
        """
        self.bridge_url = bridge_url
        self.timeout = timeout
        self.ws: websocket.WebSocket | None = None
        self._request_id = 1
        self._connect()
    
    def _connect(self) -> None:
        """Establish WebSocket connection and verify protocol."""
        debug_print(f"Connecting to bridge at {self.bridge_url}")
        self.ws = websocket.create_connection(self.bridge_url, timeout=self.timeout)
        
        # Verify protocol version
        hello = self._rpc("hello", {})
        protocol_version = int(hello.get("protocol_version", 0))
        if protocol_version != 1:
            raise RuntimeError(
                f"Unsupported bridge protocol version: {protocol_version}"
            )
        debug_print(f"Bridge connected (protocol v{protocol_version})")
    
    def _rpc(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send RPC request and wait for response.
        
        Args:
            method: RPC method name
            payload: Request payload
            
        Returns:
            Response payload (error raises RuntimeError)
        """
        if self.ws is None:
            raise RuntimeError("Bridge socket is not connected")
        
        request_id = self._request_id
        self._request_id += 1
        
        message = {
            "id": request_id,
            "method": method,
            "payload": payload,
        }
        self.ws.send(json.dumps(message))
        
        # Wait for matching response
        while True:
            raw = self.ws.recv()
            data = json.loads(raw)
            if data.get("id") != request_id:
                continue
            if not data.get("ok", False):
                raise RuntimeError(data.get("error", "Unknown bridge error"))
            return data.get("payload", {})
    
    def create_env(
        self,
        arena_width: int = 80,
        arena_height: int = 80,
        opponent_count: int = 7,
        max_steps: int = 800,
        decision_interval_ms: int = 100,
        obs_radius: int = 6,
        game_mode: str = "default",
        reward_score_weight: float = 0.01,
        reward_kill_weight: float = 1.0,
        reward_death_penalty: float = -2.0,
        reward_truncate_penalty: float = 0.0,
    ) -> str:
        """Create a new environment session.
        
        Args:
            arena_width: Width of game arena
            arena_height: Height of game arena
            opponent_count: Number of opponent bots
            max_steps: Max steps per episode
            decision_interval_ms: Milliseconds between game ticks
            obs_radius: Observation window radius
            game_mode: Game mode name
            reward_score_weight: Reward scale for points
            reward_kill_weight: Reward scale for kills
            reward_death_penalty: Penalty for dying
            reward_truncate_penalty: Penalty for max_steps truncation
            
        Returns:
            env_id string to use in future calls
        """
        payload = {
            "arena_width": arena_width,
            "arena_height": arena_height,
            "opponent_count": opponent_count,
            "max_steps": max_steps,
            "decision_interval_ms": decision_interval_ms,
            "obs_radius": obs_radius,
            "game_mode": game_mode,
            "reward_score_weight": reward_score_weight,
            "reward_kill_weight": reward_kill_weight,
            "reward_death_penalty": reward_death_penalty,
            "reward_truncate_penalty": reward_truncate_penalty,
        }
        response = self._rpc("create_env", payload)
        env_id = response["env_id"]
        debug_print(f"Created env {env_id}")
        return env_id
    
    def reset(self, env_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment and get initial observation.
        
        Args:
            env_id: Environment ID from create_env()
            
        Returns:
            Tuple of (observation dict, info dict)
        """
        response = self._rpc("reset", {"env_id": env_id})
        return response.get("observation", {}), response.get("info", {})
    
    def step(
        self, env_id: str, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step environment with action.
        
        Args:
            env_id: Environment ID
            action: Action (0-4)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        response = self._rpc("step", {"env_id": env_id, "action": int(action)})
        return (
            response.get("observation", {}),
            float(response.get("reward", 0.0)),
            bool(response.get("done", False)),
            bool(response.get("truncated", False)),
            response.get("info", {}),
        )
    
    def close_env(self, env_id: str) -> None:
        """Close an environment session.
        
        Args:
            env_id: Environment ID
        """
        try:
            self._rpc("close_env", {"env_id": env_id})
            debug_print(f"Closed env {env_id}")
        except Exception as e:
            debug_print(f"Error closing env {env_id}: {e}")
    
    def close(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            try:
                self.ws.close()
                debug_print("Bridge connection closed")
            except Exception:
                pass
            self.ws = None
    
    def __del__(self) -> None:
        """Ensure connection is closed on cleanup."""
        self.close()
