"""Battery discharge and energy constraint model for autonomous drone flight."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BatteryTelemetry:
    """Telemetry metrics for drone battery state and consumption."""

    remaining_energy: float
    capacity: float
    state_of_charge: float  # Normalized remaining charge in [0.0, 1.0]
    power_consumed: float  # Instantaneous power draw in energy units per second
    energy_step: float  # Energy consumed during this time step
    is_depleted: bool


class BatteryModel:
    """Simulates realistic electro-mechanical power consumption of a quadrotor drone.

    Power consumption formula:
        P_total = P_base + c_thrust * ||a||^2 + c_speed * ||v||^2
        Delta_E = P_total * dt
    """

    def __init__(
        self,
        capacity: float = 100.0,
        base_power: float = 0.05,
        thrust_coefficient: float = 0.15,
        speed_coefficient: float = 0.05,
    ) -> None:
        """Initialize the battery energy model.

        Args:
            capacity: Total battery energy capacity in nominal units.
            base_power: Baseline avionics, sensors, and flight controller power draw (per second).
            thrust_coefficient: Energy coefficient scaling with acceleration effort ||a||^2.
            speed_coefficient: Energy coefficient scaling with translational kinetic speed ||v||^2.
        """
        if capacity <= 0.0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if base_power < 0.0:
            raise ValueError(f"base_power cannot be negative, got {base_power}")
        if thrust_coefficient < 0.0:
            raise ValueError(f"thrust_coefficient cannot be negative, got {thrust_coefficient}")
        if speed_coefficient < 0.0:
            raise ValueError(f"speed_coefficient cannot be negative, got {speed_coefficient}")

        self.capacity = float(capacity)
        self.base_power = float(base_power)
        self.thrust_coefficient = float(thrust_coefficient)
        self.speed_coefficient = float(speed_coefficient)

        self._remaining_energy: float = self.capacity

    @property
    def remaining_energy(self) -> float:
        """Remaining energy in nominal units."""
        return self._remaining_energy

    @property
    def state_of_charge(self) -> float:
        """Normalized battery level in [0.0, 1.0]."""
        return max(0.0, min(1.0, self._remaining_energy / self.capacity))

    @property
    def is_depleted(self) -> bool:
        """True if battery has reached 0 energy capacity."""
        return self._remaining_energy <= 0.0

    def reset(self, initial_charge_ratio: float = 1.0) -> None:
        """Reset the battery to full capacity or specified charge fraction."""
        ratio = max(0.0, min(1.0, float(initial_charge_ratio)))
        self._remaining_energy = self.capacity * ratio

    def step(
        self,
        acceleration: np.ndarray,
        velocity: np.ndarray,
        dt: float = 0.1,
    ) -> BatteryTelemetry:
        """Advance energy consumption for one simulation step dt.

        Args:
            acceleration: Commanded acceleration vector [ax, ay, az].
            velocity: Current drone velocity vector [vx, vy, vz].
            dt: Time duration of the step in seconds.

        Returns:
            BatteryTelemetry: Detailed power consumption and remaining energy.
        """
        acc_norm_sq = float(np.sum(np.square(acceleration)))
        vel_norm_sq = float(np.sum(np.square(velocity)))

        # Compute power draw
        p_base = self.base_power
        p_thrust = self.thrust_coefficient * acc_norm_sq
        p_speed = self.speed_coefficient * vel_norm_sq
        p_total = p_base + p_thrust + p_speed

        energy_step = p_total * dt
        self._remaining_energy = max(0.0, self._remaining_energy - energy_step)

        return BatteryTelemetry(
            remaining_energy=self._remaining_energy,
            capacity=self.capacity,
            state_of_charge=self.state_of_charge,
            power_consumed=p_total,
            energy_step=energy_step,
            is_depleted=self.is_depleted,
        )
