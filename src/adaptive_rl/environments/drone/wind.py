"""3D Wind vector fields, atmospheric gradients, and stochastic Ornstein-Uhlenbeck gusts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class WindState3D:
    """Instantaneous state of 3D wind velocity vector."""

    steady: np.ndarray  # [wx, wy, wz] steady prevailing wind in m/s
    gust: np.ndarray  # [gx, gy, gz] stochastic turbulence component in m/s
    total: np.ndarray  # combined wind vector in m/s

    @property
    def speed(self) -> float:
        """Total instantaneous wind speed magnitude in m/s."""
        return float(np.linalg.norm(self.total))


class WindField3D:
    """Simulates 3D atmospheric wind fields with steady currents and stochastic gusts.

    Implements:
    1. Steady prevailing wind vector: w_steady = [wx, wy, wz].
    2. Altitude wind gradient: wind speed scales with altitude z.
    3. Ornstein-Uhlenbeck (OU) stochastic gust turbulence:
       dg = -theta * g * dt + sigma * sqrt(dt) * N(0, I)
    """

    def __init__(
        self,
        steady_wind: Tuple[float, float, float] = (1.5, 0.5, 0.0),
        gust_theta: float = 0.15,
        gust_sigma: float = 0.4,
        max_gust: float = 4.0,
        altitude_shear: float = 0.02,
        dt: float = 0.1,
    ) -> None:
        """Initialize the 3D wind field simulation.

        Args:
            steady_wind: Base prevailing wind vector [wx, wy, wz] in m/s.
            gust_theta: Mean-reversion rate of the Ornstein-Uhlenbeck gust process.
            gust_sigma: Diffusion volatility coefficient for wind turbulence.
            max_gust: Maximum allowable gust magnitude ceiling in m/s.
            altitude_shear: Linear altitude scaling coefficient (m/s increase per meter of altitude).
            dt: Time step delta in seconds.
        """
        if gust_theta < 0.0:
            raise ValueError(f"gust_theta cannot be negative, got {gust_theta}")
        if gust_sigma < 0.0:
            raise ValueError(f"gust_sigma cannot be negative, got {gust_sigma}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.steady_wind = np.array(steady_wind, dtype=np.float64)
        self.gust_theta = float(gust_theta)
        self.gust_sigma = float(gust_sigma)
        self.max_gust = float(max_gust)
        self.altitude_shear = float(altitude_shear)
        self.dt = float(dt)

        self._current_gust = np.zeros(3, dtype=np.float64)

    def reset(self, initial_gust: np.ndarray | None = None) -> None:
        """Reset the gust turbulence state to zero or specified initial vector."""
        if initial_gust is not None:
            self._current_gust = np.asarray(initial_gust, dtype=np.float64).copy()
        else:
            self._current_gust = np.zeros(3, dtype=np.float64)

    def step_gust(self, rng: np.random.Generator) -> np.ndarray:
        """Advance the stochastic Ornstein-Uhlenbeck gust turbulence by one time step dt.

        dg = -theta * g * dt + sigma * sqrt(dt) * noise

        Returns:
            np.ndarray: Updated gust vector [gx, gy, gz].
        """
        noise = rng.normal(loc=0.0, scale=1.0, size=3)
        dg = (
            -self.gust_theta * self._current_gust * self.dt
            + self.gust_sigma * np.sqrt(self.dt) * noise
        )
        self._current_gust += dg

        # Clamp gust magnitude to prevent unphysical explosive divergence
        gust_mag = float(np.linalg.norm(self._current_gust))
        if gust_mag > self.max_gust:
            self._current_gust = (self._current_gust / gust_mag) * self.max_gust

        return self._current_gust.copy()

    def get_wind(self, position: np.ndarray) -> WindState3D:
        """Calculate total instantaneous 3D wind velocity at a specific drone position.

        w_total = w_steady * (1 + shear * z) + w_gust

        Args:
            position: Drone 3D position [x, y, z] in meters.

        Returns:
            WindState3D: Instantaneous wind breakdown and total vector.
        """
        z = max(0.0, float(position[2]))
        altitude_factor = 1.0 + self.altitude_shear * z

        scaled_steady = self.steady_wind * altitude_factor
        total_wind = scaled_steady + self._current_gust

        return WindState3D(
            steady=scaled_steady.copy(),
            gust=self._current_gust.copy(),
            total=total_wind.copy(),
        )
