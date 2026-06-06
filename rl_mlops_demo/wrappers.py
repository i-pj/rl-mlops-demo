"""Custom environment wrappers and factory for CarRacing-v3.

MLOps concepts demonstrated:
    - Environment preprocessing consistency
    - Multiprocessing-safe env creation for SubprocVecEnv (macOS spawn compatibility)
"""

from collections.abc import Callable

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.monitor import Monitor


class SkipFrame(gym.Wrapper):
    """Repeat the same action for `skip` frames, accumulating reward.

    This dramatically reduces the effective episode length and gives the policy
    a longer temporal horizon per training step.
    """

    def __init__(self, env: gym.Env, skip: int = 2):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class AddChannel(gym.ObservationWrapper):
    """Add a channel dimension to 2D grayscale images so VecFrameStack works."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        import numpy as np

        shape = env.observation_space.shape
        if len(shape) == 2:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(shape[0], shape[1], 1), dtype=np.uint8
            )

    def observation(self, obs):
        import numpy as np

        if len(obs.shape) == 2:
            return np.expand_dims(obs, axis=-1)
        return obs


def make_car_env(
    seed: int, frame_skip: int = 2, render_mode: str | None = None
) -> Callable[[], gym.Env]:
    """Return a callable that creates a single wrapped CarRacing env.

    IMPORTANT: This is a top-level named function, not a lambda/closure,
    because SubprocVecEnv on macOS uses `spawn` which requires picklable
    callables.
    """

    def _init() -> gym.Env:
        import torch

        # Prevent CPU thrashing in SubprocVecEnv workers by restricting them to 1 thread
        torch.set_num_threads(1)

        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = SkipFrame(env, skip=frame_skip)
        env = GrayscaleObservation(env, keep_dim=True)  # 96x96x1
        env = ResizeObservation(env, (64, 64))  # 64x64 (RL Zoo3 SOTA)
        env = AddChannel(env)  # 64x64x1
        env = Monitor(env)  # track episode rewards/lengths
        env.reset(seed=seed)
        return env

    return _init


def make_raw_car_env(seed: int, render_mode: str | None = None) -> Callable[[], gym.Env]:
    """Return the raw RGB contract used by the legacy external PPO checkpoint."""

    def _init() -> gym.Env:
        import torch

        torch.set_num_threads(1)
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init
