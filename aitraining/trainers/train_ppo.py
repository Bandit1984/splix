import argparse
import random
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from aitraining.config import ConfigLoader
from aitraining.envs import PooledSplixVecEnv, run_env_preflight
from aitraining.envs.splix_env import SplixEnvConfig, make_recorded_env, wait_for_bridge
from aitraining.experiments import ExperimentConfig, ExperimentRunner
from aitraining.utils import info_print, log_config

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except Exception:
    wandb = None
    WandbCallback = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on Splix bridge environment"
    )
    parser.add_argument("--experiment-config", default="aitraining/config/experiment.yaml")
    parser.add_argument("--bridge-url", default=None)
    parser.add_argument("--reward-config", type=str, default=None,
                        help="Path to YAML reward config (default: use built-in defaults)")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vector-mode", choices=["pooled", "classic", "subproc"], default=None)
    parser.add_argument("--preflight-check", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    exp_cfg = ExperimentConfig.from_yaml(args.experiment_config)
    if args.bridge_url is not None:
        exp_cfg.env.bridge_url = args.bridge_url
    if args.total_timesteps is not None:
        exp_cfg.training.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        exp_cfg.training.n_envs = args.n_envs
    if args.learning_rate is not None:
        exp_cfg.training.learning_rate = args.learning_rate
    if args.n_steps is not None:
        exp_cfg.training.n_steps = args.n_steps
    if args.batch_size is not None:
        exp_cfg.training.batch_size = args.batch_size
    if args.gamma is not None:
        exp_cfg.training.gamma = args.gamma
    if args.seed is not None:
        exp_cfg.seed = args.seed
    if args.vector_mode is not None:
        exp_cfg.training.vector_mode = args.vector_mode
    if args.preflight_check:
        exp_cfg.training.preflight_check = True
    if args.use_wandb:
        exp_cfg.training.use_wandb = True

    bridge_url = exp_cfg.env.bridge_url
    total_timesteps = exp_cfg.training.total_timesteps
    n_envs = exp_cfg.training.n_envs
    learning_rate = exp_cfg.training.learning_rate
    n_steps = exp_cfg.training.n_steps
    batch_size = exp_cfg.training.batch_size
    gamma = exp_cfg.training.gamma

    random.seed(exp_cfg.seed)
    np.random.seed(exp_cfg.seed)

    # Load reward config
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
    run_dir = runner.create_run_directory(exp_cfg.experiment_name, git_meta)
    runner.write_manifest(run_dir, exp_cfg, git_meta)

    model_dir = str(run_dir / "models") if args.model_dir is None else args.model_dir
    log_dir = str(run_dir / "logs") if args.log_dir is None else args.log_dir

    log_config(reward_config.to_dict(), "Reward Configuration")

    wait_for_bridge(bridge_url)
    info_print(f"Connected to bridge at {bridge_url}")

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path = Path(log_dir)
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    def env_factory():
        config = SplixEnvConfig(
            bridge_url=bridge_url,
            reward_config=reward_config,
            global_seed=exp_cfg.seed,
            strict_observation_contract=exp_cfg.training.strict_observation_contract,
        )
        return make_recorded_env(config)

    if exp_cfg.training.preflight_check:
        report = run_env_preflight(SplixEnvConfig(
            bridge_url=bridge_url,
            reward_config=reward_config,
            global_seed=exp_cfg.seed,
            strict_observation_contract=exp_cfg.training.strict_observation_contract,
        ))
        runner.write_metrics(run_dir, report, file_name="preflight_report.json")
        info_print("Gymnasium preflight check passed")

    info_print(f"Creating {n_envs} training environments (mode={exp_cfg.training.vector_mode})...")
    if exp_cfg.training.vector_mode == "pooled":
        env = PooledSplixVecEnv(
            n_envs=n_envs,
            config=SplixEnvConfig(
                bridge_url=bridge_url,
                reward_config=reward_config,
                global_seed=exp_cfg.seed,
                strict_observation_contract=exp_cfg.training.strict_observation_contract,
            ),
        )
        env = VecMonitor(env)
    elif exp_cfg.training.vector_mode == "subproc":
        env = make_vec_env(env_factory, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    else:
        env = make_vec_env(env_factory, n_envs=n_envs)

    eval_env = DummyVecEnv([env_factory])
    eval_env = VecMonitor(eval_env)

    info_print(f"Starting training for {total_timesteps} timesteps...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        seed=exp_cfg.seed,
        verbose=1,
        tensorboard_log=str(tensorboard_path),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, int(exp_cfg.training.checkpoint_freq // max(1, n_envs))),
            save_path=str(model_path),
            name_prefix="ppo_checkpoint",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_path),
            log_path=str(tensorboard_path),
            eval_freq=max(1, int(exp_cfg.training.eval_freq // max(1, n_envs))),
            deterministic=True,
            render=False,
        ),
    ]

    if exp_cfg.training.use_wandb and wandb is not None and WandbCallback is not None:
        wandb.init(project="splix-ai", name=run_dir.name, config=exp_cfg.to_dict(), sync_tensorboard=True)
        callbacks.append(WandbCallback(model_save_path=str(model_path), verbose=2))

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    model.save(model_path / "ppo_splix_latest")
    runner.write_metrics(
        run_dir,
        {
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "vector_mode": exp_cfg.training.vector_mode,
            "seed": exp_cfg.seed,
            "model_path": str(model_path / "ppo_splix_latest.zip"),
            "log_dir": str(tensorboard_path),
        },
    )
    info_print(f"Model saved to {model_path / 'ppo_splix_latest.zip'}")
    info_print(f"Run artifacts written to {run_dir}")
    eval_env.close()
    env.close()
    if exp_cfg.training.use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
	main()

