import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from aitraining.config import ConfigLoader
from aitraining.envs import run_env_preflight
from aitraining.envs.splix_env import SplixEnv, SplixEnvConfig, wait_for_bridge
from aitraining.experiments import ExperimentConfig, ExperimentRunner
from aitraining.utils import info_print, log_config, Timer


def heuristic_action(obs: np.ndarray, obs_radius: int) -> int:
    """Choose a direction using simple local tile scoring around current position."""
    side = obs_radius * 2 + 1
    tile_count = side * side
    tiles = obs[:tile_count].reshape(side, side)
    center = obs_radius

    # Actions: right=0, down=1, left=2, up=3, paused=4
    neighbors = {
        0: tiles[center, center + 1],
        1: tiles[center + 1, center],
        2: tiles[center, center - 1],
        3: tiles[center - 1, center],
    }
    tile_scores = {
        -1.0: -10_000.0,
        0.0: 1.5,
        1.0: 0.6,
        2.0: 0.2,
    }

    best_action = 4
    best_score = -float("inf")
    for action, tile_value in neighbors.items():
        score = tile_scores.get(float(tile_value), -1.0)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def select_action(policy: str, model: PPO | None, obs: np.ndarray, obs_radius: int) -> int:
    if policy == "random":
        return int(np.random.randint(0, 5))
    if policy == "heuristic":
        return heuristic_action(obs, obs_radius)
    if model is None:
        raise RuntimeError("PPO baseline selected but model is not loaded")
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def run_policy_scenario(
    policy: str,
    env: SplixEnv,
    episodes: int,
    obs_radius: int,
    model: PPO | None,
) -> dict[str, Any]:
    rewards = []
    lengths = []
    death_causes: dict[str, int] = {}

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0
        last_info: dict[str, Any] = {}
        while not done and not truncated:
            action = select_action(policy, model, obs, obs_radius)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            last_info = info
        rewards.append(ep_reward)
        lengths.append(ep_len)
        terminal = last_info.get("terminal_event")
        if isinstance(terminal, dict):
            cause = str(terminal.get("cause_bucket", "unknown"))
            death_causes[cause] = death_causes.get(cause, 0) + 1

    return {
        "episodes": episodes,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_length": float(np.std(lengths)) if lengths else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "death_causes": death_causes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO model on Splix bridge environment")
    parser.add_argument("--experiment-config", default="aitraining/config/experiment.yaml")
    parser.add_argument("--bridge-url", default=None)
    parser.add_argument("--reward-config", type=str, default=None,
                        help="Path to YAML reward config (must match training config)")
    parser.add_argument("--model-path", default="aitraining/models/ppo_splix_latest.zip")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Optional override for episodes per scenario")
    parser.add_argument("--preflight-check", action="store_true")
    args = parser.parse_args()

    exp_cfg = ExperimentConfig.from_yaml(args.experiment_config)
    if args.bridge_url is not None:
        exp_cfg.env.bridge_url = args.bridge_url

    bridge_url = exp_cfg.env.bridge_url
    model_path = args.model_path
    episodes_override = args.episodes

    wait_for_bridge(bridge_url)
    if not Path(model_path).exists():
        raise RuntimeError(f"Model not found: {model_path}")

    # Load reward config (must match training config)
    reward_config = ConfigLoader.load_rewards(args.reward_config)
    reward_config.arena_width = exp_cfg.env.arena_width
    reward_config.arena_height = exp_cfg.env.arena_height
    reward_config.opponent_count = exp_cfg.env.opponent_count
    reward_config.max_steps = exp_cfg.env.max_steps
    reward_config.decision_interval_ms = exp_cfg.env.decision_interval_ms
    reward_config.obs_radius = exp_cfg.env.obs_radius
    reward_config.game_mode = exp_cfg.env.game_mode
    reward_config.score_weight = exp_cfg.rewards.score_weight
    reward_config.kill_weight = exp_cfg.rewards.kill_weight
    reward_config.death_penalty = exp_cfg.rewards.death_penalty
    reward_config.truncate_penalty = exp_cfg.rewards.truncate_penalty

    runner = ExperimentRunner(Path.cwd())
    git_meta = runner.get_git_metadata()
    run_dir = runner.create_run_directory(f"{exp_cfg.experiment_name}-eval", git_meta)
    runner.write_manifest(run_dir, exp_cfg, git_meta)

    if args.preflight_check or exp_cfg.training.preflight_check:
        report = run_env_preflight(SplixEnvConfig(
            bridge_url=bridge_url,
            reward_config=reward_config,
            global_seed=exp_cfg.seed,
            strict_observation_contract=exp_cfg.training.strict_observation_contract,
        ))
        runner.write_metrics(run_dir, report, file_name="preflight_report.json")
        info_print("Gymnasium preflight check passed")

    log_config(reward_config.to_dict(), "Reward Configuration")

    ppo_model: PPO | None = None
    if "ppo" in exp_cfg.evaluation.baselines:
        with Timer("Loading model"):
            ppo_model = PPO.load(model_path)
        info_print(f"Model loaded from {model_path}")

    scenario_results = []
    info_print("Running evaluation matrix...")
    for scenario in exp_cfg.evaluation.scenarios:
        scenario_episodes = episodes_override or scenario.episodes
        scenario_reward_cfg = ConfigLoader.load_rewards(args.reward_config)
        scenario_reward_cfg.arena_width = exp_cfg.env.arena_width
        scenario_reward_cfg.arena_height = exp_cfg.env.arena_height
        scenario_reward_cfg.opponent_count = scenario.opponent_count
        scenario_reward_cfg.max_steps = scenario.max_steps
        scenario_reward_cfg.decision_interval_ms = exp_cfg.env.decision_interval_ms
        scenario_reward_cfg.obs_radius = exp_cfg.env.obs_radius
        scenario_reward_cfg.game_mode = exp_cfg.env.game_mode
        scenario_reward_cfg.score_weight = exp_cfg.rewards.score_weight
        scenario_reward_cfg.kill_weight = exp_cfg.rewards.kill_weight
        scenario_reward_cfg.death_penalty = exp_cfg.rewards.death_penalty
        scenario_reward_cfg.truncate_penalty = exp_cfg.rewards.truncate_penalty

        for baseline in exp_cfg.evaluation.baselines:
            env_seed = exp_cfg.seed
            if baseline == "random":
                np.random.seed(exp_cfg.seed)
            env = SplixEnv(SplixEnvConfig(
                bridge_url=bridge_url,
                reward_config=scenario_reward_cfg,
                global_seed=env_seed,
                strict_observation_contract=exp_cfg.training.strict_observation_contract,
            ))
            metrics = run_policy_scenario(
                baseline,
                env,
                scenario_episodes,
                scenario_reward_cfg.obs_radius,
                ppo_model,
            )
            env.close()

            row = {
                "scenario": scenario.name,
                "baseline": baseline,
                "episodes": scenario_episodes,
                **metrics,
            }
            scenario_results.append(row)
            print(
                f"  [{scenario.name}] {baseline}: avg_reward={metrics['avg_reward']:.3f}, "
                f"avg_length={metrics['avg_length']:.1f}"
            )

    info_print("Evaluation complete")
    runner.write_metrics(
        run_dir,
        {
            "model_path": str(Path(model_path).resolve()),
            "results": scenario_results,
        },
        file_name="eval_metrics.json",
    )

    csv_path = run_dir / "eval_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "baseline",
                "episodes",
                "avg_reward",
                "std_reward",
                "avg_length",
                "std_length",
                "max_reward",
                "min_reward",
                "death_causes",
            ],
        )
        writer.writeheader()
        for row in scenario_results:
            writer.writerow(row)

    info_print(f"Evaluation artifacts written to {run_dir}")


if __name__ == "__main__":
	main()
