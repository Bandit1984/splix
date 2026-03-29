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
            "trajectory_compare_executed": False,
            "trajectory_match": False,
            "trajectory_steps": 0,
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

            # Soft parity smoke check: run both envs with identical fixed actions
            # and report whether trajectories match. We do not fail preflight on
            # mismatch because bridge timing/network effects can introduce drift.
            fixed_actions = [0, 1, 2]
            obs_single, _ = env.reset(seed=config.global_seed)
            obs_vec = pooled_env.reset()
            parity_report["trajectory_steps"] = len(fixed_actions)
            trajectory_match = bool(np.allclose(obs_single, obs_vec[0], atol=1e-6))

            for action in fixed_actions:
                obs_single, reward_single, done_single, trunc_single, _ = env.step(action)
                pooled_env.step_async(np.asarray([action], dtype=np.int64))
                obs_vec, reward_vec, done_vec, _ = pooled_env.step_wait()

                step_match = (
                    np.allclose(obs_single, obs_vec[0], atol=1e-6)
                    and np.isclose(float(reward_single), float(reward_vec[0]), atol=1e-6)
                    and bool(done_single or trunc_single) == bool(done_vec[0])
                )
                trajectory_match = bool(trajectory_match and step_match)

            parity_report["trajectory_compare_executed"] = True
            parity_report["trajectory_match"] = trajectory_match
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
