from __future__ import annotations

from typing import Any

from gymnasium.utils.env_checker import check_env

from .splix_env import SplixEnvConfig, make_monitored_env


def run_env_preflight(config: SplixEnvConfig) -> dict[str, Any]:
    """Run a Gymnasium contract pre-flight check and return a machine-readable report."""
    env = make_monitored_env(config)
    try:
        check_env(env.unwrapped, skip_render_check=True)
        obs, info = env.reset(seed=config.global_seed)
        report = {
            "ok": True,
            "observation_shape": list(obs.shape),
            "observation_dtype": str(obs.dtype),
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "initial_info_keys": sorted(list(info.keys())),
        }
        return report
    finally:
        env.close()
