# `openarm_dataset` — RLDS / TFDS conversion for Octo fine-tuning

This folder wraps the npz files produced by `openarm bc-collect --images` into the
**RLDS (Open X-Embodiment)** schema that Octo's `scripts/finetune.py` consumes.

## What it produces

A TFDS dataset with one train split. Each episode is a sequence of steps shaped:

| Key | Shape | dtype | Notes |
|---|---|---|---|
| `observation.image` | `(256, 256, 3)` | `uint8` | top-down camera (tablecam) RGB |
| `observation.state` | `(23,)` | `float32` | joint pos/vel + EE pose |
| `action` | `(7,)` | `float32` | normalised joint-delta in `[-1, 1]` |
| `language_instruction` | string | | constant per episode, repeated each step |
| `is_first`, `is_last`, `is_terminal` | scalar | `bool` | RLDS episode flags |
| `discount`, `reward` | scalar | `float32` | discount=1.0; sparse reward at episode end |

The two tasks are merged with different prompts:
- `"reach the target"` from `demos/octo_reach.npz`
- `"insert the peg into the socket"` from `demos/octo_insert.npz`

## Usage

```bash
# one-off dependencies (heavy: tensorflow ~500 MB)
pip install tensorflow tensorflow-datasets

# collect the npz demos first (from the repo root)
openarm bc-collect --task reach  --episodes 500 --images --img-size 256 --out demos/octo_reach.npz
openarm bc-collect --task insert --episodes 500 --images --img-size 256 --out demos/octo_insert.npz

# materialise the RLDS dataset
cd scripts/openarm_dataset
tfds build

# the dataset is written to ~/tensorflow_datasets/openarm_dataset/1.0.0/
# Octo's finetune.py reads it by name.
```

## How Octo's `finetune.py` maps these keys

Octo's dataset config (`scripts/configs/finetune_config.py`) wants:

```python
image_obs_keys={"primary": "image"},   # -> observation.image
proprio_obs_key="state",                # -> observation.state
language_key="language_instruction",    # -> the prompt
action_dim=7,
action_horizon=16,                      # 16-step action chunks (short eps)
```

Done — at training time the policy sees image + state + language, predicts a chunk
of future actions. At eval time we feed the environment's RGB + state + a prompt
("reach the target" / "insert the peg into the socket") and execute the predicted
action chunks.
