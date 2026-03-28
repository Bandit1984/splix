# AI Training Enhancements

## Overview

This document describes the new modular architecture for reward configuration, logging, and multi-environment training support added to the Splix AI training system.

---

## 1. Reward Configuration System

### Purpose
Tune agent behavior via YAML configuration instead of code changes.

### Files Added
- **`aitraining/config/rewards.yaml`** - Default reward configuration
- **`aitraining/config/rewards.py`** - `RewardConfig` dataclass and `ConfigLoader`
- **`aitraining/config/__init__.py`** - Package exports

### Usage

#### Basic Example
```bash
# Train with default rewards
deno task python-train

# Train with custom reward config
deno task python-train --reward-config aitraining/config/rewards.yaml
```

#### Customizing Rewards
Edit `aitraining/config/rewards.yaml`:
```yaml
score_weight: 0.01        # Reward per point
kill_weight: 1.0          # Reward per kill
death_penalty: -2.0       # Penalty for dying
truncate_penalty: 0.0     # Penalty for max_steps reached

arena_width: 80
arena_height: 80
opponent_count: 7
max_steps: 800
```

#### Programmatic Access
```python
from aitraining.config import RewardConfig, ConfigLoader

# Load from YAML
config = ConfigLoader.load_rewards("path/to/config.yaml")

# Or use defaults
config = RewardConfig.default()

# Access as dict for logging
print(config.to_dict())
```

### Key Features
- ✅ YAML-based configuration (human-readable)
- ✅ Automatic fallback to defaults if file missing
- ✅ `ConfigLoader.load_rewards()` with graceful error handling
- ✅ Integrated into `train_ppo.py` and `eval_ppo.py`
- ✅ Pretty-printed config display on training startup

---

## 2. Debugging & Logging Utilities

### Purpose
Structured logging with colors, timing, and configuration display for easy debugging.

### Files Added
- **`aitraining/utils/__init__.py`** - Logging utilities
- **`aitraining/utils/decorators.py`** - Function decoration utilities (planned)

### Usage

#### Basic Logging
```python
from aitraining.utils import info_print, warn_print, error_print, debug_print

info_print("Training started")
warn_print("Config file not found, using defaults")
error_print("Bridge connection failed")
debug_print("Loaded 4 environments")
```

#### Timing Code Blocks
```python
from aitraining.utils import Timer

with Timer("Loading model"):
    model = PPO.load("path/to/model.zip")
    # Output: [DEBUG] Loading model: 1.234s
```

#### Timing Functions
```python
from aitraining.utils import timeit

@timeit
def expensive_operation():
    pass
    # Output: [DEBUG] expensive_operation() took 0.456s
```

#### Log Configuration
```python
from aitraining.utils import log_config

config_dict = {"learning_rate": 3e-4, "batch_size": 256}
log_config(config_dict, "Training Config")
# Output:
# [INFO] Training Config:
#   learning_rate: 0.0003
#   batch_size: 256
```

### Features
- 🎨 Color-coded output (GREEN/YELLOW/RED/CYAN)
- ⏱️ Function and block timing decorators
- 📋 Structured configuration logging
- 🔍 Debug-level messages separated to stderr
- 🛡️ Safe error handling (no exceptions from logging)

---

## 3. WebSocket Connection Pooling

### Purpose
Share a single WebSocket connection across multiple training environments for better throughput and reduced overhead.

### Files Added
- **`aitraining/envs/websocket_pool.py`** - `WebSocketPool` class

### Usage

#### Create Shared Pool
```python
from aitraining.envs.websocket_pool import WebSocketPool

pool = WebSocketPool("ws://127.0.0.1:8080/ai-bridge")

env_id_1 = pool.create_env()
env_id_2 = pool.create_env()
```

#### Step Multiple Environments
```python
obs1, info1 = pool.reset(env_id_1)
obs2, info2 = pool.reset(env_id_2)

obs1, reward1, done1, truncated1, info1 = pool.step(env_id_1, action_1)
obs2, reward2, done2, truncated2, info2 = pool.step(env_id_2, action_2)

pool.close_env(env_id_1)
pool.close_env(env_id_2)
pool.close()
```

#### With VecEnv
```python
def env_factory():
    return SplixEnv(config)

# Stable-Baselines3's make_vec_env already handles vectorization
env = make_vec_env(env_factory, n_envs=4)
```

### Key Features
- ✅ Single WebSocket connection for multiple envs
- ✅ Automatic request ID routing (concurrent requests)
- ✅ All RPC methods supported (create_env, reset, step, observe, close_env)
- ✅ Graceful error handling and cleanup
- ✅ Debug logging for diagnostics

---

## 4. Updated Training Scripts Integration

### train_ppo.py Enhancements
- New `--reward-config` argument for YAML config
- Pretty-printed reward configuration on startup
- Improved logging with `info_print()` progress feedback
- Automatic environment creation with reward config
- Better error messages

```bash
# Usage
PYTHONPATH=. aitraining/.venv/bin/python -m aitraining.trainers.train_ppo \
  --reward-config aitraining/config/rewards.yaml \
  --total-timesteps 500000 \
  --n-envs 4 \
  --model-dir aitraining/models
```

### eval_ppo.py Enhancements
- New `--reward-config` argument (must match training config)
- Improved episode progress reporting
- Detailed statistics (mean, std, min, max)
- Structured config logging
- Better timing information

```bash
# Usage
PYTHONPATH=. aitraining/.venv/bin/python -m aitraining.trainers.eval_ppo \
  --reward-config aitraining/config/rewards.yaml \
  --model-path aitraining/models/ppo_splix_latest.zip \
  --episodes 50
```

---

## 5. Module Structure

```
aitraining/
├── config/                  # Reward configuration
│   ├── __init__.py
│   ├── rewards.py          # RewardConfig class, ConfigLoader
│   └── rewards.yaml        # Default reward weights
│
├── envs/                   # Gymnasium environment wrappers
│   ├── __init__.py
│   ├── splix_env.py        # Updated with RewardConfig support
│   └── websocket_pool.py   # NEW: Connection pooling
│
├── utils/                  # Debugging and utilities
│   └── __init__.py         # NEW: Logging, timing, config display
│
├── trainers/               # Training scripts
│   ├── __init__.py
│   ├── train_ppo.py        # Updated with reward config support
│   └── eval_ppo.py         # Updated with reward config support
│
├── requirements.txt        # Updated: +pyyaml==6.0.1
└── scripts/
    └── setup_venv.sh       # Venv setup
```

---

## 6. Architecture Benefits

### Modularity
- Each component is self-contained and testable
- Easy to extend (add new config formats, logging handlers, etc.)
- No circular dependencies
- Clear separation of concerns

### Debugging
- Structured logging with color-coded output
- Timing functions and blocks for performance profiling
- Configuration display on startup (transparency)
- Debug-level messages for development

### Scalability
- WebSocket pooling reduces I/O overhead
- Reusable components across scripts
- Configuration-driven behavior (data vs. code)
- Clean interfaces for future enhancements

### Maintainability
- YAML-based config (no code edits for tuning)
- Consistent logging across all modules
- Type hints for better IDE support
- Comprehensive docstrings

---

## 7. Quick Start

### 1. Install Dependencies
```bash
deno task python-setup  # Installs pyyaml + others
```

### 2. Create Custom Reward Config
```bash
cp aitraining/config/rewards.yaml aitraining/config/my_rewards.yaml
# Edit my_rewards.yaml with your reward weights
```

### 3. Train with Custom Config
```bash
deno task dev  # Start bridge
deno task python-train --reward-config aitraining/config/my_rewards.yaml
```

### 4. Evaluate with Same Config
```bash
deno task python-eval \
  --reward-config aitraining/config/my_rewards.yaml \
  --model-path aitraining/models/ppo_splix_latest.zip
```

---

## 8. Future Extensions

Possible enhancements (modular design enables these):

1. **Multi-format configs** - Support JSON, TOML, YAML
2. **Distributed training** - WebSocketPool for remote bridges
3. **Dynamic reward tuning** - Adjust weights during training
4. **Policy templates** - Different neural architectures via config
5. **Experiment tracking** - MLflow/W&B integration
6. **Custom agents** - Pluggable agent implementations

---

## 9. Testing

All components tested via:
```bash
# Verify imports
python -c "from aitraining.config import RewardConfig; \
           from aitraining.utils import Timer; \
           from aitraining.envs import WebSocketPool"

# Load and display config
PYTHONPATH=. aitraining/.venv/bin/python -c \
  "from aitraining.config import ConfigLoader; \
   print(ConfigLoader.load_rewards('aitraining/config/rewards.yaml'))"
```

---

## Commit Information

**Branch:** development  
**Commit:** ca356f8  
**Date:** March 28, 2026

Changes: 9 files changed, 492 insertions(+), 21 deletions(-), 1 file mode change

---

*Created by: AI Training Enhancement System*  
*Last Updated: March 28, 2026*
