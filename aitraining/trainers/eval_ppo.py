import argparse
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO

from aitraining.envs.splix_env import SplixEnv, SplixEnvConfig, wait_for_bridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO model on Splix bridge environment")
    parser.add_argument("--bridge-url", default="ws://127.0.0.1:8080/ai-bridge")
    parser.add_argument("--model-path", default="aitraining/models/ppo_splix_latest.zip")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    bridge_url = args.bridge_url
    model_path = args.model_path
    episodes = args.episodes

    wait_for_bridge(bridge_url)
    if not Path(model_path).exists():
        raise RuntimeError(f"Model not found: {model_path}")

    env = SplixEnv(SplixEnvConfig(bridge_url=bridge_url))
    model = PPO.load(model_path)

    rewards = []
    lengths = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(int(action))
            ep_reward += reward
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)

    env.close()
    print("Evaluation complete")
    print(f"episodes={episodes}")
    print(f"avg_reward={np.mean(rewards):.3f}")
    print(f"avg_length={np.mean(lengths):.3f}")


if __name__ == "__main__":
	main()
