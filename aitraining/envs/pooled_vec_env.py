from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .observation_contract import (
    flatten_observation,
    observation_bounds,
    validate_observation_bounds,
)
from .splix_env import SplixEnvConfig
from .websocket_pool import WebSocketPool


class PooledSplixVecEnv(VecEnv):
    """VecEnv that multiplexes many env_ids over a single websocket connection."""

    def __init__(self, n_envs: int, config: SplixEnvConfig):
        self.config = config
        self.pool = WebSocketPool(config.bridge_url)
        self.env_ids: list[str] = []
        self._last_actions: np.ndarray | None = None
        self._obs_tile_size = (config.obs_radius * 2 + 1) ** 2
        self._scalar_size = 8
        self._obs_low: np.ndarray | None = None
        self._obs_high: np.ndarray | None = None

        for _ in range(n_envs):
            env_index = len(self.env_ids)
            env_seed = (
                None
                if config.global_seed is None
                else int(config.global_seed) + env_index
            )
            env_id = self.pool.create_env(
                arena_width=config.arena_width,
                arena_height=config.arena_height,
                opponent_count=config.opponent_count,
                max_steps=config.max_steps,
                decision_interval_ms=config.decision_interval_ms,
                obs_radius=config.obs_radius,
                game_mode=config.game_mode,
                reward_score_weight=config.reward_score_weight,
                reward_kill_weight=config.reward_kill_weight,
                reward_death_penalty=config.reward_death_penalty,
                reward_truncate_penalty=config.reward_truncate_penalty,
                global_seed=env_seed,
            )
            self.env_ids.append(env_id)

        super().__init__(
            n_envs,
            self._build_observation_space(),
            self._build_action_space(),
        )

    def _build_observation_space(self):
        from gymnasium import spaces

        obs_low, obs_high = observation_bounds(self._obs_tile_size)
        return spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self._obs_tile_size + self._scalar_size,),
            dtype=np.float32,
        )

    def _build_action_space(self):
        from gymnasium import spaces

        return spaces.Discrete(5)

    def _flatten_observation(self, obs: dict[str, Any]) -> np.ndarray:
        flattened = flatten_observation(
            obs,
            opponent_count=self.config.opponent_count,
            max_steps=self.config.max_steps,
        )
        if self.config.strict_observation_contract:
            if self._obs_low is None or self._obs_high is None:
                self._obs_low = self.observation_space.low
                self._obs_high = self.observation_space.high
            validate_observation_bounds(flattened, self._obs_low, self._obs_high)
        return flattened

    def reset(self) -> np.ndarray:
        observations = []
        for env_id in self.env_ids:
            obs, _ = self.pool.reset(env_id)
            observations.append(self._flatten_observation(obs))
        return np.asarray(observations, dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._last_actions = np.asarray(actions)

    def step_wait(self):
        if self._last_actions is None:
            raise RuntimeError("step_async must be called before step_wait")

        action_map = {
            env_id: int(self._last_actions[i])
            for i, env_id in enumerate(self.env_ids)
        }
        batched = self.pool.step_many(action_map)

        observations = []
        rewards = []
        dones = []
        infos = []

        for env_id in self.env_ids:
            obs, reward, done, truncated, info = batched[env_id]
            terminal = bool(done or truncated)
            if terminal:
                final_obs = self._flatten_observation(obs)
                info = dict(info)
                info["terminal_observation"] = final_obs
                obs, _ = self.pool.reset(env_id)
            observations.append(self._flatten_observation(obs))
            rewards.append(float(reward))
            dones.append(terminal)
            infos.append(info)

        return (
            np.asarray(observations, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env_id in self.env_ids:
            self.pool.close_env(env_id)
        self.pool.close()

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self.config, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for _ in self._get_indices(indices):
            setattr(self.config, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("env_method is not supported for pooled bridge vec env.")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
