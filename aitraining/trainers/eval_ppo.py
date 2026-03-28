import argparse
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO

from aitraining.config import ConfigLoader
from aitraining.envs.splix_env import SplixEnv, SplixEnvConfig, wait_for_bridge
from aitraining.utils import info_print, log_config, Timer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO model on Splix bridge environment")
    parser.add_argument("--bridge-url", default="ws://127.0.0.1:8080/ai-bridge")
    parser.add_argument("--reward-config", type=str, default=None,
                        help="Path to YAML reward config (must match training config)")
    parser.add_argument("--model-path", default="aitraining/models/ppo_splix_latest.zip")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    bridge_url = args.bridge_url
    model_path = args.model_path
    episodes = args.episodes

    wait_for_bridge(bridge_url)
    if not Path(model_path).exists():
        raise RuntimeError(f"Model not found: {model_path}")

    # Load reward config (must match training config)
    reward_config = ConfigLoader.load_rewards(args.reward_config)
    log_config(reward_config.to_dict(), "Reward Configuration")

    with Timer("Loading model"):
        model = PPO.load(model_path)
    info_print(f"Model loaded from {model_path}")

    env = SplixEnv(SplixEnvConfig(
        bridge_url=bridge_url,
        reward_config=reward_config,
    ))

    rewards = []
    lengths = []
    
    info_print(f"Running {episodes} evaluation episodes...")
    for ep in range(episodes):
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
        if (ep + 1) % max(1, episodes // 4) == 0:
            print(f"  [{ep + 1}/{episodes}] Episode reward: {ep_reward:.1f}, length: {ep_len}")

    env.close()
    
    info_print("Evaluation complete")
    print(f"  Episodes: {episodes}")
    print(f"  Avg Reward: {np.mean(rewards):.3f} (±{np.std(rewards):.3f})")
    print(f"  Avg Length: {np.mean(lengths):.1f} (±{np.std(lengths):.1f})")
    print(f"  Max Reward: {np.max(rewards):.3f}")
    print(f"  Min Reward: {np.min(rewards):.3f}")


if __name__ == "__main__":
	main()
