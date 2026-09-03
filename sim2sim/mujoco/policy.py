"""Dependency-light inference for the exported Footstep actor."""

from pathlib import Path

import numpy as np


class NumpyMLPPolicy:
    """Four-layer ELU actor including the training-time observation RMS."""

    def __init__(self, weights, biases, obs_mean, obs_var, epsilon=1.0e-5):
        self.weights = [np.asarray(value, dtype=np.float32) for value in weights]
        self.biases = [np.asarray(value, dtype=np.float32) for value in biases]
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_var = np.asarray(obs_var, dtype=np.float32)
        self.epsilon = float(epsilon)
        if len(self.weights) != 4 or len(self.biases) != 4:
            raise ValueError("The R1 actor must contain four linear layers")
        if self.obs_mean.shape != (54,) or self.weights[0].shape[1] != 54:
            raise ValueError("The R1 foothold policy must consume 54 observations")
        if self.weights[-1].shape[0] != 12:
            raise ValueError("The R1 foothold policy must produce 12 actions")

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path) as bundle:
            return cls(
                [bundle[f"weight_{index}"] for index in range(4)],
                [bundle[f"bias_{index}"] for index in range(4)],
                bundle["obs_mean"],
                bundle["obs_var"],
                float(bundle["epsilon"]),
            )

    @staticmethod
    def _elu(value):
        return np.where(value > 0.0, value, np.expm1(value))

    def __call__(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape[-1] != 54:
            raise ValueError(f"Expected 54 observations, got {observation.shape}")
        value = (observation - self.obs_mean) / np.sqrt(
            self.obs_var + self.epsilon
        )
        value = np.clip(value, -5.0, 5.0)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = value @ weight.T + bias
            if index < len(self.weights) - 1:
                value = self._elu(value)
        return value.astype(np.float32, copy=False)
