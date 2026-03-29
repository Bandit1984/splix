from __future__ import annotations

from typing import Any

from gymnasium.utils.env_checker import check_env
import numpy as np

from .observation_contract import validate_observation_bounds
from .pooled_vec_env import PooledSplixVecEnv
from .splix_env import SplixEnvConfig, make_monitored_env


def run_env_preflight(config: SplixEnvConfig) -> dict[str, Any]:
    """Run a Gymnasium contract pre-flight check and return a machine-readable report."""
    env = make_monitored_env(config)
    try:
        check_env(env.unwrapped, skip_render_check=True)
        obs, info = env.reset(seed=config.global_seed)
        validate_observation_bounds(obs, env.observation_space.low, env.observation_space.high)

        sampled_rewards: list[float] = []
        sampled_terminated = 0
        sampled_truncated = 0
        sampled_steps = 5
        for _ in range(sampled_steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            validate_observation_bounds(obs, env.observation_space.low, env.observation_space.high)
            sampled_rewards.append(float(reward))
            if terminated:
                sampled_terminated += 1
            if truncated:
                sampled_truncated += 1

        parity_report = {
            "observation_space_shape_match": True,
            "observation_space_low_match": True,
            "observation_space_high_match": True,
        }
        pooled_env = PooledSplixVecEnv(n_envs=1, config=config)
        try:
            parity_report["observation_space_shape_match"] = (
                tuple(env.observation_space.shape) == tuple(pooled_env.observation_space.shape)
            )
            parity_report["observation_space_low_match"] = bool(
                np.array_equal(env.observation_space.low, pooled_env.observation_space.low)
            )
            parity_report["observation_space_high_match"] = bool(
                np.array_equal(env.observation_space.high, pooled_env.observation_space.high)
            )
        finally:
            pooled_env.close()

        parity_report["ok"] = all(parity_report.values())

        report = {
            "ok": True,
            "observation_shape": list(obs.shape),
            "observation_dtype": str(obs.dtype),
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "initial_info_keys": sorted(list(info.keys())),
            "bounds_check": {
                "validated": True,
                "sampled_steps": sampled_steps,
                "reward_min": min(sampled_rewards),
                "reward_max": max(sampled_rewards),
                "terminated_count": sampled_terminated,
                "truncated_count": sampled_truncated,
                "strict_observation_contract": bool(config.strict_observation_contract),
            },
            "parity_check": parity_report,
        }
        report["ok"] = bool(report["ok"] and parity_report["ok"])
        return report
    finally:
        env.close()
