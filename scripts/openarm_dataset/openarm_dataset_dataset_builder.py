"""TFDS DatasetBuilder for the OpenArm v2 scripted-demo dataset (RLDS format).

This wraps the npz files written by ``openarm bc-collect --images`` into the
RLDS (Open X-Embodiment) schema that Octo's fine-tune pipeline consumes:

    observation:
        image:   (256, 256, 3) uint8        -> image_obs_keys={"primary": "image"}
        state:   (23,)         float32      -> proprio_obs_key="state"
    action:      (7,)          float32      -> action_dim=7
    language_instruction: str               -> per-step, episode-wide value
    is_first / is_last / is_terminal: bool
    discount / reward:           float32

Build it with::

    pip install tensorflow tensorflow-datasets
    cd scripts/openarm_dataset
    tfds build

which materialises the dataset under ``~/tensorflow_datasets/openarm_dataset/``.
Octo's ``scripts/finetune.py`` then loads it via the standard RLDS dataloader.
"""
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# Each npz produced by ``openarm bc-collect --images`` is mapped to a
# language instruction. The fine-tune covers both tasks jointly so the policy
# can language-condition on the prompt at inference time.
TASK_DEMOS = {
    "reach the target":                "demos/octo_reach.npz",
    "insert the peg into the socket":  "demos/octo_insert.npz",
}

# Repo root, derived from this file's location (.../scripts/openarm_dataset/<builder>.py).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class OpenArmDataset(tfds.core.GeneratorBasedBuilder):
    """OpenArm v2 scripted-demo dataset for vision-language-action fine-tuning."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release: reach + insert from scripted experts at 256x256."}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Scripted demonstrations on the Enactic OpenArm v2 (7-DOF) in MuJoCo, "
                "covering two tasks (reach a target, insert a peg into a round socket). "
                "Collected with bc-collect from openarm_control/imitation/collect.py. "
                "Designed for fine-tuning Octo / generalist VLA policies."
            ),
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=tf.uint8,
                            encoding_format="png",
                            doc="Top-down camera (tablecam) RGB at 256x256.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(23,),
                            dtype=tf.float32,
                            doc="Proprioceptive state (joint pos/vel + end-effector pose).",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,),
                        dtype=tf.float32,
                        doc="Normalised joint-delta action in [-1, 1].",
                    ),
                    "discount": tfds.features.Scalar(dtype=tf.float32),
                    "reward":   tfds.features.Scalar(dtype=tf.float32),
                    "is_first": tfds.features.Scalar(dtype=tf.bool),
                    "is_last":  tfds.features.Scalar(dtype=tf.bool),
                    "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                    "language_instruction": tfds.features.Text(
                        doc="Task prompt (constant per episode, repeated on every step "
                            "as Octo's dataloader expects).",
                    ),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "task":      tfds.features.Text(),
                    "file_path": tfds.features.Text(),
                }),
            }),
            supervised_keys=None,
            homepage="https://github.com/Manas-arumalla/openarm-control",
        )

    def _split_generators(self, dl_manager):
        # One train split. Custom held-out eval is handled at policy-eval time, not here.
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        ep_id = 0
        for instruction, rel_path in TASK_DEMOS.items():
            npz_path = os.path.join(REPO_ROOT, rel_path)
            if not os.path.exists(npz_path):
                # Skip silently if a task's demos haven't been collected yet.
                continue
            d = np.load(npz_path)
            obs, act, imgs, ep_lens = d["obs"], d["act"], d["images"], d["ep_lens"]
            start = 0
            task_name = os.path.splitext(os.path.basename(rel_path))[0].replace("octo_", "")
            for L in ep_lens:
                end = start + int(L)
                steps = []
                for i in range(start, end):
                    steps.append({
                        "observation": {
                            "image": imgs[i],
                            "state": obs[i].astype(np.float32),
                        },
                        "action":   act[i].astype(np.float32),
                        "discount": np.float32(1.0),
                        # Sparse reward at episode end (the scripted expert always succeeds).
                        "reward":   np.float32(1.0 if i == end - 1 else 0.0),
                        "is_first": (i == start),
                        "is_last":  (i == end - 1),
                        "is_terminal": (i == end - 1),
                        "language_instruction": instruction,
                    })
                yield ep_id, {
                    "steps": steps,
                    "episode_metadata": {
                        "task":      task_name,
                        "file_path": rel_path,
                    },
                }
                ep_id += 1
                start = end
