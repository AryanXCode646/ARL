"""Environment wrapper for partitioning training environment distributions by seed."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np


class TrainingDistributionWrapper(gym.Wrapper):
    """Restricts environment resets to a predefined, finite set of training seeds.

    Guarantees strict separation between training and evaluation distributions:
    the wrapped environment will only ever sample from the designated training seed pool.
    """

    def __init__(
        self,
        env: gym.Env,
        seeds: Sequence[int],
        shuffle: bool = True,
        rng_seed: int = 42,
    ) -> None:
        """Initialize training distribution wrapper.

        Args:
            env: Base Gymnasium environment.
            seeds: List or range of allowable training seeds.
            shuffle: Whether to sample randomly with replacement or sequentially.
            rng_seed: Master seed for seed sampler.
        """
        super().__init__(env)
        if not seeds:
            raise ValueError("Training seed distribution cannot be empty.")
        self.training_seeds: List[int] = [int(s) for s in seeds]
        self.shuffle = shuffle
        self._rng = np.random.default_rng(rng_seed)
        self._seed_index = 0
        self._last_seed: Optional[int] = None
        self._sampled_seeds_history: List[int] = []

    @property
    def last_seed(self) -> Optional[int]:
        """Return the most recently assigned environment seed."""
        return self._last_seed

    @property
    def sampled_seeds_history(self) -> List[int]:
        """History of all seeds sampled during resets."""
        return list(self._sampled_seeds_history)

    def sample_next_seed(self) -> int:
        """Draw the next training seed according to sampling strategy."""
        if self.shuffle:
            idx = int(self._rng.integers(0, len(self.training_seeds)))
            return self.training_seeds[idx]
        else:
            seed = self.training_seeds[self._seed_index % len(self.training_seeds)]
            self._seed_index += 1
            return seed

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment enforcing a seed from the training distribution.

        Always draws and enforces a seed from the designated training seed distribution,
        shielding the environment from arbitrary external seeds and guaranteeing zero test contamination.
        """
        if seed is not None and seed in self.training_seeds:
            selected_seed = seed
        else:
            selected_seed = self.sample_next_seed()

        self._last_seed = selected_seed
        self._sampled_seeds_history.append(selected_seed)

        obs, info = self.env.reset(seed=selected_seed, options=options)
        info["training_distribution_seed"] = selected_seed
        return obs, info
