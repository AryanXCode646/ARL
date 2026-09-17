"""3D Kinematics and continuous translation physics for autonomous drone navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DroneState3D:
    """Represents the continuous 3D state of a drone."""

    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in meters/second
    acceleration: np.ndarray  # [ax, ay, az] in meters/second^2

    def copy(self) -> DroneState3D:
        """Create a deep copy of the current kinematic state."""
        return DroneState3D(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
        )


class DroneKinematics3D:
    """3D point-mass quadrotor translational kinematic equations of motion.

    Simulates continuous 3D motion under commanded accelerations with aerodynamic drag
    damping and physical velocity limits:
        dv/dt = a - c_d * v
        dp/dt = v
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_velocity: float = 8.0,
        max_acceleration: float = 4.0,
        linear_damping: float = 0.05,
    ) -> None:
        """Initialize the 3D kinematic model.

        Args:
            dt: Integration time step duration in seconds.
            max_velocity: Maximum speed limit in meters per second.
            max_acceleration: Maximum acceleration limit per axis in m/s^2.
            linear_damping: Linear aerodynamic drag/damping coefficient.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if max_velocity <= 0.0:
            raise ValueError(f"max_velocity must be positive, got {max_velocity}")
        if max_acceleration <= 0.0:
            raise ValueError(f"max_acceleration must be positive, got {max_acceleration}")
        if linear_damping < 0.0:
            raise ValueError(f"linear_damping cannot be negative, got {linear_damping}")

        self.dt = float(dt)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.linear_damping = float(linear_damping)

        self.state = DroneState3D(
            position=np.zeros(3, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            acceleration=np.zeros(3, dtype=np.float64),
        )

    def reset(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> DroneState3D:
        """Reset the drone kinematic state.

        Args:
            position: Initial 3D position vector [x, y, z].
            velocity: Optional initial 3D velocity vector (defaults to zero).

        Returns:
            DroneState3D: Fresh kinematic state.
        """
        pos = np.asarray(position, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"Position must have shape (3,), got {pos.shape}")

        vel = (
            np.asarray(velocity, dtype=np.float64)
            if velocity is not None
            else np.zeros(3, dtype=np.float64)
        )
        if vel.shape != (3,):
            raise ValueError(f"Velocity must have shape (3,), got {vel.shape}")

        self.state = DroneState3D(
            position=pos.copy(),
            velocity=vel.copy(),
            acceleration=np.zeros(3, dtype=np.float64),
        )
        return self.state.copy()

    def step(self, action_acceleration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate 3D kinematics forward by one time step dt.

        Args:
            action_acceleration: Continuous 3D acceleration command [ax, ay, az].

        Returns:
            Tuple[np.ndarray, np.ndarray]: (new_position, new_velocity) as float64 arrays.
        """
        raw_acc = np.asarray(action_acceleration, dtype=np.float64)
        if raw_acc.shape != (3,):
            raise ValueError(f"Acceleration command must have shape (3,), got {raw_acc.shape}")

        # Clamp acceleration to configured bounds
        clamped_acc = np.clip(raw_acc, -self.max_acceleration, self.max_acceleration)
        self.state.acceleration = clamped_acc

        # Semi-implicit Euler integration with aerodynamic drag
        # a_eff = a - damping * v
        effective_acc = clamped_acc - self.linear_damping * self.state.velocity
        new_velocity = self.state.velocity + effective_acc * self.dt

        # Enforce maximum speed ceiling (spherical norm clamp)
        speed = float(np.linalg.norm(new_velocity))
        if speed > self.max_velocity:
            new_velocity = (new_velocity / speed) * self.max_velocity

        # Position update
        new_position = self.state.position + new_velocity * self.dt

        self.state.velocity = new_velocity
        self.state.position = new_position

        return self.state.position.copy(), self.state.velocity.copy()
