import argparse
from pathlib import Path

from aitraining.config import ConfigLoader
from aitraining.envs import run_env_preflight
from aitraining.envs.splix_env import SplixEnvConfig, wait_for_bridge
from aitraining.experiments import ExperimentConfig, ExperimentRunner
from aitraining.utils import info_print, log_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gymnasium preflight checks for Splix bridge env")
    parser.add_argument("--experiment-config", default="aitraining/config/experiment.yaml")
    parser.add_argument("--bridge-url", default=None)
    parser.add_argument("--reward-config", type=str, default=None)
    args = parser.parse_args()

    exp_cfg = ExperimentConfig.from_yaml(args.experiment_config)
    if args.bridge_url is not None:
        exp_cfg.env.bridge_url = args.bridge_url

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

    wait_for_bridge(exp_cfg.env.bridge_url)
    log_config(reward_config.to_dict(), "Reward Configuration")

    report = run_env_preflight(
        SplixEnvConfig(
            bridge_url=exp_cfg.env.bridge_url,
            reward_config=reward_config,
            global_seed=exp_cfg.seed,
            strict_observation_contract=exp_cfg.training.strict_observation_contract,
        )
    )

    runner = ExperimentRunner(Path.cwd())
    git_meta = runner.get_git_metadata()
    run_dir = runner.create_run_directory(f"{exp_cfg.experiment_name}-preflight", git_meta)
    runner.write_manifest(run_dir, exp_cfg, git_meta)
    runner.write_metrics(run_dir, report, file_name="preflight_report.json")

    info_print("Preflight check passed")
    info_print(f"Preflight artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
