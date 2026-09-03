"""Export a Footstep checkpoint without importing Isaac Gym."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DeploymentActor(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        state = checkpoint["model_state_dict"]
        dimensions = [checkpoint["num_actor_obs"]] + checkpoint["actor_hidden_dims"] + [checkpoint["num_actions"]]
        layers = []
        linear_index = 0
        for index in range(len(dimensions) - 1):
            layer = nn.Linear(dimensions[index], dimensions[index + 1])
            layer.weight.data.copy_(state[f"actor.mean_NN.{linear_index}.weight"])
            layer.bias.data.copy_(state[f"actor.mean_NN.{linear_index}.bias"])
            layers.append(layer)
            if index < len(dimensions) - 2:
                layers.append(nn.ELU())
                linear_index += 2
        self.network = nn.Sequential(*layers)
        self.register_buffer("obs_mean", state["actor.obs_rms.running_mean"].float())
        self.register_buffer("obs_var", state["actor.obs_rms.running_var"].float())

    def forward(self, observation):
        observation = (observation - self.obs_mean) / torch.sqrt(self.obs_var + 1.0e-5)
        return self.network(torch.clamp(observation, -5.0, 5.0))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(checkpoint_path, output_dir):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if (checkpoint["num_actor_obs"], checkpoint["num_actions"]) != (54, 12):
        raise ValueError("Expected the R1 foothold checkpoint dimensions 54 -> 12")
    actor = DeploymentActor(checkpoint).eval()
    scripted = torch.jit.trace(actor, torch.zeros(54))
    scripted.save(str(output_dir / "policy.pt"))

    state = checkpoint["model_state_dict"]
    linear_indices = (0, 2, 4, 6)
    arrays = {
        "obs_mean": state["actor.obs_rms.running_mean"].float().numpy(),
        "obs_var": state["actor.obs_rms.running_var"].float().numpy(),
        "epsilon": np.array(1.0e-5, dtype=np.float32),
    }
    for output_index, state_index in enumerate(linear_indices):
        arrays[f"weight_{output_index}"] = state[f"actor.mean_NN.{state_index}.weight"].numpy()
        arrays[f"bias_{output_index}"] = state[f"actor.mean_NN.{state_index}.bias"].numpy()
    np.savez(output_dir / "policy_weights.npz", **arrays)

    rng = np.random.default_rng(7)
    test_input = rng.normal(size=(32, 54)).astype(np.float32)
    with torch.no_grad():
        torch_output = actor(torch.from_numpy(test_input)).numpy()
    value = np.clip(
        (test_input - arrays["obs_mean"]) / np.sqrt(arrays["obs_var"] + 1.0e-5),
        -5.0,
        5.0,
    )
    for index in range(4):
        value = value @ arrays[f"weight_{index}"].T + arrays[f"bias_{index}"]
        if index < 3:
            value = np.where(value > 0.0, value, np.expm1(value))
    max_error = float(np.max(np.abs(torch_output - value)))
    validation = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_iteration": int(checkpoint["iter"]),
        "actor_observations": int(checkpoint["num_actor_obs"]),
        "actions": int(checkpoint["num_actions"]),
        "numpy_max_abs_error": max_error,
    }
    (output_dir / "export_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    if max_error > 1.0e-5:
        raise RuntimeError(f"NumPy export mismatch: {max_error}")
    print(json.dumps(validation, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
