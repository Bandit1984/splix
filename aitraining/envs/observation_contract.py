from __future__ import annotations

from typing import Any

import numpy as np


def scalar_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Canonical scalar bounds for Splix observations.

    Scalars are ordered as:
    x_norm, y_norm, direction_enum, score_norm, kills_norm,
    dead_flag, permanently_dead_flag, step_norm.
    """
    low = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    high = np.asarray([1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    return low, high


def observation_bounds(obs_tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical flattened observation bounds."""
    scalar_low, scalar_high = scalar_bounds()
    low = np.concatenate([
        np.full(obs_tile_size, -1.0, dtype=np.float32),
        scalar_low,
    ])
    high = np.concatenate([
        np.full(obs_tile_size, 2.0, dtype=np.float32),
        scalar_high,
    ])
    return low, high


def flatten_observation(
    obs: dict[str, Any],
    *,
    opponent_count: int,
    max_steps: int,
) -> np.ndarray:
    """Flatten raw bridge observation into canonical policy vector."""
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
            float(player["score"]) / max(1.0, float(arena["width"] * arena["height"])),
            float(player["kills"]) / max(1.0, float(opponent_count + 1)),
            1.0 if player["dead"] else 0.0,
            1.0 if player["permanently_dead"] else 0.0,
            float(obs.get("step", 0)) / max(1.0, float(max_steps)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([local_tiles, scalars], dtype=np.float32)


def validate_observation_bounds(
    flattened_obs: np.ndarray,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    *,
    atol: float = 1e-5,
) -> None:
    """Raise ValueError when an observation is out of declared bounds."""
    if flattened_obs.shape != obs_low.shape:
        raise ValueError(
            f"Observation shape mismatch: got {flattened_obs.shape}, expected {obs_low.shape}",
        )

    below = np.where(flattened_obs < (obs_low - atol))[0]
    above = np.where(flattened_obs > (obs_high + atol))[0]
    if below.size == 0 and above.size == 0:
        return

    violating = np.unique(np.concatenate([below, above]))
    preview_idx = violating[:10]
    raise ValueError(
        "Observation values out of bounds. "
        f"Violating indices (first {len(preview_idx)}): {preview_idx.tolist()} "
        f"Values: {flattened_obs[preview_idx].tolist()} "
        f"Low: {obs_low[preview_idx].tolist()} "
        f"High: {obs_high[preview_idx].tolist()}",
    )
