# Splix Python AI Training

This folder contains a high-level Python RL stack for Splix using Gymnasium + Stable-Baselines3.

## Quickstart

1. Start the Deno dev server (includes /ai-bridge):
   deno task dev

2. Set up Python venv and dependencies:
   bash aitraining/scripts/setup_venv.sh

3. Train PPO (high-level defaults):
   source aitraining/.venv/bin/activate
   PYTHONPATH=. python -m aitraining.trainers.train_ppo

4. Evaluate latest model:
   PYTHONPATH=. python -m aitraining.trainers.eval_ppo --model-path aitraining/models/ppo_splix_latest.zip

## Bridge Protocol

The Python env talks to ws://127.0.0.1:8080/ai-bridge using JSON RPC-like messages:
- hello
- create_env
- reset
- observe
- step
- close_env
- ping
