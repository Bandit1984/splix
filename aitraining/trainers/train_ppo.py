import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from aitraining.envs.splix_env import SplixEnv, SplixEnvConfig, wait_for_bridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Splix bridge environment")
    parser.add_argument("--bridge-url", default="ws://127.0.0.1:8080/ai-bridge")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--obs-radius", type=int, default=6)
    parser.add_argument("--model-dir", default="aitraining/models")
    parser.add_argument("--log-dir", default="aitraining/logs")
    args = parser.parse_args()

    bridge_url = args.bridge_url
    total_timesteps = args.total_timesteps
    n_envs = args.n_envs
    learning_rate = args.learning_rate
    n_steps = args.n_steps
    batch_size = args.batch_size
    gamma = args.gamma
    obs_radius = args.obs_radius
    model_dir = args.model_dir
    log_dir = args.log_dir

    wait_for_bridge(bridge_url)

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    tensorboard_path = Path(log_dir)
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    def env_factory() -> SplixEnv:
        config = SplixEnvConfig(
            bridge_url=bridge_url,
            obs_radius=obs_radius,
        )
        return SplixEnv(config)

    env = make_vec_env(env_factory, n_envs=n_envs)
    env = VecMonitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        verbose=1,
        tensorboard_log=str(tensorboard_path),
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path / "ppo_splix_latest")
    env.close()


if __name__ == "__main__":
	main()
