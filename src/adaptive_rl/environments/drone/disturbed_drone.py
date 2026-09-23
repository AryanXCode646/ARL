"""Gymnasium-compatible 3D drone navigation under environmental disturbances and constraints."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.drone.battery import BatteryModel
from adaptive_rl.environments.drone.dynamic_obstacles import (
    DynamicObstacleSphere3D,
    generate_dynamic_drone_obstacles,
)
from adaptive_rl.environments.drone.kinematics import DroneKinematics3D
from adaptive_rl.environments.drone.obstacles import (
    ObstacleSphere3D,
    compute_lidar_3d_readings,
    generate_drone_obstacles,
    generate_lidar_3d_ray_directions,
)
from adaptive_rl.environments.drone.wind import WindField3D


class DisturbanceRecoveryTracker:
    """Step-indexed disturbance/recovery event tracker (environment steps, never wall-clock).

    Formal recovery-time contract (see Issue #105):

    1. Disturbance event: the transient wind magnitude
       ``||gust + injected_disturbance||`` (m/s, steady wind excluded) reaches or
       exceeds ``threshold``. The threshold is derived from the environment's own
       stochastic scales (see ``DroneDisturbance3DEnv.disturbance_event_threshold``).
    2. Onset: the first step on which the magnitude crosses at-or-above the
       threshold while no event is active. Events never overlap: a new event can
       only begin after the previous one recovered.
    3. Recovery: the magnitude subsides strictly below the threshold AND the drone
       speed returns to within ``speed_tolerance`` (m/s) of the speed recorded at
       onset, sustained for ``hold_steps`` consecutive steps. The step on which the
       hold completes is the recovery step.
    4. Censoring: if the episode terminates (or is truncated) while an event is
       still open, that event is censored: it contributes to the event count but
       never to a recovery-time mean, and is never reported as ``0``.
    5. Units: every recovery time is an integer count of environment steps
       (``recovery_step - onset_step >= 1`` by construction; never wall-clock time).
    """

    def __init__(self) -> None:
        """Initialize an empty tracker with no recorded events."""
        self.reset()

    def reset(self) -> None:
        """Clear all recorded events and active-event state for a new episode."""
        self._events: List[Dict[str, Optional[int]]] = []
        self._active_onset: Optional[int] = None
        self._ref_speed: float = 0.0
        self._calm_streak: int = 0
        self._recovered_this_step: bool = False
        self._completed_this_step: Optional[int] = None

    def update(
        self,
        *,
        step: int,
        magnitude: float,
        speed: float,
        threshold: float,
        speed_tolerance: float,
        hold_steps: int,
    ) -> None:
        """Advance the tracker by one environment step.

        Args:
            step: Current 1-based environment step index within the episode.
            magnitude: Transient disturbance magnitude ``||gust + injected||`` in m/s.
            speed: Current drone speed in m/s.
            threshold: Onset/subsidence magnitude threshold in m/s. A non-positive
                threshold disables event detection entirely (no events recorded).
            speed_tolerance: Allowed deviation (m/s) from the onset reference speed.
            hold_steps: Required consecutive calm steps before recovery completes.
        """
        self._recovered_this_step = False
        self._completed_this_step = None
        mag = float(magnitude)

        if self._active_onset is None:
            if threshold > 0.0 and mag >= threshold:
                self._active_onset = int(step)
                self._ref_speed = float(speed)
                self._calm_streak = 0
                self._events.append({"onset": int(step), "recovery": None})
            return

        calm = (mag < threshold) and (abs(float(speed) - self._ref_speed) <= speed_tolerance)
        if calm:
            self._calm_streak += 1
        else:
            self._calm_streak = 0

        if self._calm_streak >= max(1, int(hold_steps)):
            recovery_time = int(step) - int(self._active_onset)
            self._events[-1]["recovery"] = recovery_time
            self._completed_this_step = recovery_time
            self._recovered_this_step = True
            self._active_onset = None
            self._calm_streak = 0

    def telemetry(self) -> Dict[str, Any]:
        """Return the per-step telemetry contract consumed by evaluation code."""
        completed = [e["recovery"] for e in self._events if e["recovery"] is not None]
        return {
            "disturbance_active": self._active_onset is not None,
            "disturbance_onset_step": self._active_onset,
            "recovered": self._recovered_this_step,
            "recovery_time": self._completed_this_step,
            "recovery_events": len(self._events),
            "recovery_completed": len(completed),
            "recovery_open": 1 if self._active_onset is not None else 0,
            "recovery_censored": 1 if self._active_onset is not None else 0,
            "recovery_times": [int(t) for t in completed],
            "episode_recovery_time": float(sum(completed) / len(completed)) if completed else None,
        }


class DroneDisturbance3DEnv(AdaptiveRLEnv[np.ndarray, np.ndarray]):
    """Autonomous 3D Drone Navigation under atmospheric disturbances and energy constraints.

    Simulates:
    1. Atmospheric wind fields: Steady crosswinds, altitude shear, and stochastic OU gusts.
    2. Dynamic obstacles: Moving 3D spherical obstacles with boundary bounce physics.
    3. Battery depletion: Power draw proportional to avionics, thrust effort, and airspeed.
    4. Hidden injected disturbance: An additional Ornstein-Uhlenbeck process
       (``disturbance_strength``) perturbing the dynamics through the same
       ``linear_damping`` force path as wind, but excluded from the wind
       observation channels, for controlled distribution-shift benchmarking.

    Recovery telemetry (Gymnasium ``info``, every step and reset):
    ``disturbance_magnitude``, ``disturbance_threshold``, ``disturbance_active``,
    ``disturbance_onset_step``, ``recovered``, ``recovery_time``,
    ``recovery_events``, ``recovery_completed``, ``recovery_open``,
    ``recovery_censored``, ``recovery_times``, ``episode_recovery_time``.
    See :class:`DisturbanceRecoveryTracker` for the formal contract.

    Actions:
        Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        Continuous 3D acceleration command [ax, ay, az].

    Observations:
        Box(-1.0, 1.0, shape=(33,), dtype=np.float32)
        [0:3]   : Normalized 3D position [x/X_max, y/Y_max, z/Z_max] in [0, 1]
        [3:6]   : Normalized 3D velocity [vx/v_max, vy/v_max, vz/v_max] in [-1, 1]
        [6:9]   : Normalized target goal [gx/X_max, gy/Y_max, gz/Z_max] in [0, 1]
        [9:12]  : Relative target vector [(gx - x)/X_max, (gy - y)/Y_max, (gz - z)/Z_max] in [-1, 1]
        [12]    : Normalized distance to goal ||goal - pos|| / max_diagonal in [0, 1]
        [13:29] : 16-ray 3D LiDAR normalized distance readings in [0, 1]
        [29]    : Normalized battery state of charge (remaining energy / capacity) in [0, 1]
        [30:33] : Normalized instantaneous 3D wind velocity vector in [-1, 1]
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        bounds: Tuple[float, float, float] = (50.0, 50.0, 25.0),
        start_pos: Optional[Tuple[float, float, float] | np.ndarray] = None,
        goal_pos: Optional[Tuple[float, float, float] | np.ndarray] = None,
        num_obstacles: int = 6,
        num_dynamic_obstacles: int = 3,
        obstacle_radius: float = 2.0,
        dynamic_obstacle_speed: float = 2.0,
        target_radius: float = 1.5,
        collision_radius: float = 0.8,
        lidar_range: float = 20.0,
        num_lidar_rays: int = 16,
        dt: float = 0.1,
        max_velocity: float = 8.0,
        max_acceleration: float = 4.0,
        linear_damping: float = 0.05,
        steady_wind: Tuple[float, float, float] = (1.5, 0.5, 0.0),
        gust_theta: float = 0.15,
        gust_sigma: float = 0.4,
        max_wind_speed: float = 10.0,
        battery_capacity: float = 100.0,
        battery_base_power: float = 0.05,
        battery_thrust_coeff: float = 0.15,
        battery_speed_coeff: float = 0.05,
        max_steps: int = 300,
        step_penalty: float = -0.05,
        goal_reward: float = 100.0,
        collision_reward: float = -100.0,
        battery_exhaustion_penalty: float = -50.0,
        progress_weight: float = 1.0,
        action_penalty_weight: float = 0.01,
        terminate_on_collision: bool = True,
        terminate_on_exhaustion: bool = True,
        render_mode: Optional[str] = None,
        wind_speed: Optional[float] = None,
        disturbance_strength: float = 0.0,
        disturbance_theta: float = 0.15,
        disturbance_event_factor: float = 1.5,
        recovery_speed_tolerance: float = 1.0,
        recovery_hold_steps: int = 5,
    ) -> None:
        """Initialize the Disturbed 3D Drone Navigation environment.

        New distribution-shift parameters (all optional, defaults preserve legacy
        behavior exactly):
            wind_speed: Scalar steady-wind magnitude in m/s. ``None`` (default)
                keeps ``steady_wind`` unchanged; otherwise the steady wind vector
                is rescaled to this magnitude preserving direction.
            disturbance_strength: Per-axis stationary standard deviation (m/s) of
                the hidden injected OU disturbance. ``0.0`` (default) disables it.
            disturbance_theta: Mean-reversion rate of the injected OU process.
            disturbance_event_factor: Dimensionless onset factor; a disturbance
                event starts when ``||gust + injected||`` reaches
                ``disturbance_event_factor`` times the transient RMS scale.
            recovery_speed_tolerance: Allowed speed deviation (m/s) from the
                onset reference speed for recovery.
            recovery_hold_steps: Consecutive calm steps required for recovery.
        """
        super().__init__()
        if any(b <= 0.0 for b in bounds):
            raise ValueError(f"Bounds must be strictly positive, got {bounds}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if battery_capacity <= 0.0:
            raise ValueError(f"battery_capacity must be positive, got {battery_capacity}")
        if wind_speed is not None and wind_speed < 0.0:
            raise ValueError(f"wind_speed cannot be negative, got {wind_speed}")
        if disturbance_strength < 0.0:
            raise ValueError(f"disturbance_strength cannot be negative, got {disturbance_strength}")
        if disturbance_theta < 0.0:
            raise ValueError(f"disturbance_theta cannot be negative, got {disturbance_theta}")
        if disturbance_event_factor <= 0.0:
            raise ValueError(
                f"disturbance_event_factor must be positive, got {disturbance_event_factor}"
            )
        if recovery_speed_tolerance < 0.0:
            raise ValueError(
                f"recovery_speed_tolerance cannot be negative, got {recovery_speed_tolerance}"
            )
        if recovery_hold_steps < 1:
            raise ValueError(f"recovery_hold_steps must be >= 1, got {recovery_hold_steps}")

        self.bounds: Tuple[float, float, float] = (
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
        )
        self.default_start = (
            np.asarray(start_pos, dtype=np.float64)
            if start_pos is not None
            else np.array([5.0, 5.0, 5.0], dtype=np.float64)
        )
        self.default_goal = (
            np.asarray(goal_pos, dtype=np.float64)
            if goal_pos is not None
            else np.array(
                [self.bounds[0] - 5.0, self.bounds[1] - 5.0, self.bounds[2] - 5.0], dtype=np.float64
            )
        )
        self.num_obstacles = num_obstacles
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.obstacle_radius = float(obstacle_radius)
        self.dynamic_obstacle_speed = float(dynamic_obstacle_speed)
        self.target_radius = float(target_radius)
        self.collision_radius = float(collision_radius)
        self.lidar_range = float(lidar_range)
        self.num_lidar_rays = num_lidar_rays
        self.dt = float(dt)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.linear_damping = float(linear_damping)
        self.max_wind_speed = float(max_wind_speed)
        self.max_steps = max_steps
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.collision_reward = float(collision_reward)
        self.battery_exhaustion_penalty = float(battery_exhaustion_penalty)
        self.progress_weight = float(progress_weight)
        self.action_penalty_weight = float(action_penalty_weight)
        self.terminate_on_collision = terminate_on_collision
        self.terminate_on_exhaustion = terminate_on_exhaustion
        self.render_mode = render_mode
        self.wind_speed: Optional[float] = None if wind_speed is None else float(wind_speed)
        self.disturbance_strength = float(disturbance_strength)
        self.disturbance_theta = float(disturbance_theta)
        self.disturbance_event_factor = float(disturbance_event_factor)
        self.recovery_speed_tolerance = float(recovery_speed_tolerance)
        self.recovery_hold_steps = int(recovery_hold_steps)

        self.max_diagonal = float(np.linalg.norm(self.bounds))

        # Farama Gymnasium Spaces
        # Action space: 3D acceleration command in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Observation space: 13 (base navigation) + 16 (lidar) + 1 (battery) + 3 (wind) = 33
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(17 + self.num_lidar_rays,),
            dtype=np.float32,
        )

        # Subsystems
        self.lidar_rays = generate_lidar_3d_ray_directions(num_rays=self.num_lidar_rays)
        self.kinematics = DroneKinematics3D(
            dt=self.dt,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            linear_damping=linear_damping,
        )
        self.wind_field = WindField3D(
            steady_wind=steady_wind,
            gust_theta=gust_theta,
            gust_sigma=gust_sigma,
            max_gust=4.0,
            dt=self.dt,
        )
        if self.wind_speed is not None:
            self._apply_wind_speed(self.wind_speed)
        self.battery = BatteryModel(
            capacity=battery_capacity,
            base_power=battery_base_power,
            thrust_coefficient=battery_thrust_coeff,
            speed_coefficient=battery_speed_coeff,
        )

        # Dynamic state
        self._position: np.ndarray = self.default_start.copy()
        self._velocity: np.ndarray = np.zeros(3, dtype=np.float64)
        self._goal: np.ndarray = self.default_goal.copy()
        self._static_obstacles: List[ObstacleSphere3D] = []
        self._dynamic_obstacles: List[DynamicObstacleSphere3D] = []
        self._current_step = 0
        self._prev_distance_to_goal: float = float(
            np.linalg.norm(self.default_goal - self.default_start)
        )
        self._injected_disturbance: np.ndarray = np.zeros(3, dtype=np.float64)
        self._last_disturbance_magnitude: float = 0.0
        self._recovery = DisturbanceRecoveryTracker()

    def _apply_wind_speed(self, value: float) -> None:
        """Rescale the steady wind vector to the requested magnitude (m/s).

        Preserves the currently configured steady-wind direction (falling back to
        the canonical default direction when it is the zero vector).
        """
        val = float(value)
        if val < 0.0:
            raise ValueError(f"wind_speed cannot be negative, got {value}")
        current = np.asarray(self.wind_field.steady_wind, dtype=np.float64)
        norm = float(np.linalg.norm(current))
        if norm > 0.0:
            direction = current / norm
        else:
            default = np.array([1.5, 0.5, 0.0], dtype=np.float64)
            direction = default / float(np.linalg.norm(default))
        self.wind_field.steady_wind = direction * val

    def _advance_disturbance(self) -> np.ndarray:
        """Advance the hidden injected OU disturbance by one time step dt.

        dx = -theta * x * dt + sigma * sqrt(dt) * N(0, I), with sigma chosen so
        the per-axis stationary standard deviation equals ``disturbance_strength``
        (m/s of equivalent wind). A no-op returning zeros when disabled.
        """
        if self.disturbance_strength > 0.0:
            if self.disturbance_theta > 0.0:
                sigma = self.disturbance_strength * math.sqrt(2.0 * self.disturbance_theta)
            else:
                sigma = self.disturbance_strength
            noise = self.np_random.normal(loc=0.0, scale=1.0, size=3)
            self._injected_disturbance += (
                -self.disturbance_theta * self._injected_disturbance * self.dt
                + sigma * math.sqrt(self.dt) * noise
            )
            # Clamp to prevent unphysical explosive divergence (mirrors max_gust).
            cap = 4.0 * self.disturbance_strength
            mag = float(np.linalg.norm(self._injected_disturbance))
            if mag > cap:
                self._injected_disturbance = (self._injected_disturbance / mag) * cap
        return self._injected_disturbance

    @property
    def disturbance_event_threshold(self) -> float:
        """Magnitude threshold (m/s) at which a disturbance event starts.

        Defined as ``disturbance_event_factor`` times the transient RMS scale
        ``sqrt(3) * sqrt(gust_stat^2 + disturbance_strength^2)``, where
        ``gust_stat = gust_sigma / sqrt(2 * gust_theta)`` is the per-axis
        stationary gust standard deviation. Returns ``0.0`` (detection disabled)
        when no stochastic disturbance is configured.
        """
        theta = float(self.wind_field.gust_theta)
        sigma = float(self.wind_field.gust_sigma)
        gust_stat = sigma / math.sqrt(2.0 * theta) if theta > 0.0 else sigma
        total_var = gust_stat * gust_stat + self.disturbance_strength * self.disturbance_strength
        if total_var <= 0.0:
            return 0.0
        return float(self.disturbance_event_factor * math.sqrt(3.0 * total_var))

    def get_effective_parameters(self) -> Dict[str, Any]:
        """Return the concrete parameter set governing current environment dynamics.

        Used by distribution-shift benchmarking to record exactly what was
        evaluated per scenario. Values reflect runtime state (e.g. after
        ``set_parameters``), not just constructor arguments.
        """
        steady = [float(v) for v in list(np.asarray(self.wind_field.steady_wind).tolist())]
        return {
            "bounds": [float(b) for b in self.bounds],
            "num_obstacles": int(self.num_obstacles),
            "num_dynamic_obstacles": int(self.num_dynamic_obstacles),
            "obstacle_radius": float(self.obstacle_radius),
            "dynamic_obstacle_speed": float(self.dynamic_obstacle_speed),
            "steady_wind": steady,
            "wind_speed": float(np.linalg.norm(self.wind_field.steady_wind)),
            "gust_theta": float(self.wind_field.gust_theta),
            "gust_sigma": float(self.wind_field.gust_sigma),
            "disturbance_strength": float(self.disturbance_strength),
            "disturbance_theta": float(self.disturbance_theta),
            "disturbance_event_factor": float(self.disturbance_event_factor),
            "disturbance_event_threshold": float(self.disturbance_event_threshold),
            "recovery_speed_tolerance": float(self.recovery_speed_tolerance),
            "recovery_hold_steps": int(self.recovery_hold_steps),
            "linear_damping": float(self.linear_damping),
            "max_acceleration": float(self.max_acceleration),
            "max_velocity": float(self.max_velocity),
            "max_steps": int(self.max_steps),
            "dt": float(self.dt),
            "battery_capacity": float(self.battery.capacity),
            "target_radius": float(self.target_radius),
            "collision_radius": float(self.collision_radius),
        }

    def _get_all_obstacles_as_spheres(self) -> List[ObstacleSphere3D]:
        """Combine static and dynamic obstacles into unified sphere list for distance querying."""
        combined: List[ObstacleSphere3D] = list(self._static_obstacles)
        for dyn_obs in self._dynamic_obstacles:
            combined.append(dyn_obs.to_static_sphere())
        return combined

    def _get_obs(self) -> np.ndarray:
        """Construct normalized 33-dimensional observation vector."""
        bx, by, bz = self.bounds
        v_max = self.kinematics.max_velocity
        w_max = max(1.0, self.max_wind_speed)

        norm_pos = [
            self._position[0] / bx,
            self._position[1] / by,
            self._position[2] / bz,
        ]
        norm_vel = [
            self._velocity[0] / v_max,
            self._velocity[1] / v_max,
            self._velocity[2] / v_max,
        ]
        norm_goal = [
            self._goal[0] / bx,
            self._goal[1] / by,
            self._goal[2] / bz,
        ]
        rel_goal = [
            (self._goal[0] - self._position[0]) / bx,
            (self._goal[1] - self._position[1]) / by,
            (self._goal[2] - self._position[2]) / bz,
        ]
        curr_dist = float(np.linalg.norm(self._goal - self._position))
        norm_dist = [min(1.0, curr_dist / self.max_diagonal)]

        all_obstacles = self._get_all_obstacles_as_spheres()
        lidar_readings = compute_lidar_3d_readings(
            origin=self._position,
            ray_directions=self.lidar_rays,
            obstacles=all_obstacles,
            bounds=self.bounds,
            max_range=self.lidar_range,
        )

        battery_soc = [self.battery.state_of_charge]

        # Instantaneous wind vector
        wind_state = self.wind_field.get_wind(self._position)
        norm_wind = [
            wind_state.total[0] / w_max,
            wind_state.total[1] / w_max,
            wind_state.total[2] / w_max,
        ]

        raw = np.concatenate(
            [
                norm_pos,
                norm_vel,
                norm_goal,
                rel_goal,
                norm_dist,
                lidar_readings,
                battery_soc,
                norm_wind,
            ]
        )
        return np.asarray(np.clip(raw, -1.0, 1.0), dtype=np.float32)

    def _check_collision(self, pos: np.ndarray) -> Tuple[bool, str]:
        """Check collisions against arena boundaries, static obstacles, and dynamic obstacles."""
        r = self.collision_radius
        bx, by, bz = self.bounds

        if pos[0] - r <= 0.0 or pos[0] + r >= bx:
            return True, "boundary_x"
        if pos[1] - r <= 0.0 or pos[1] + r >= by:
            return True, "boundary_y"
        if pos[2] - r <= 0.0 or pos[2] + r >= bz:
            return True, "boundary_z"

        for obs in self._static_obstacles:
            if obs.collides_with(pos, r):
                return True, "static_obstacle"

        for dyn_obs in self._dynamic_obstacles:
            if dyn_obs.collides_with(pos, r):
                return True, "dynamic_obstacle"

        return False, "none"

    def _get_info(self) -> Dict[str, Any]:
        """Construct detailed telemetry and status dictionary."""
        dist = float(np.linalg.norm(self._goal - self._position))
        speed = float(np.linalg.norm(self._velocity))
        wind_state = self.wind_field.get_wind(self._position)

        min_obs_dist = float("inf")
        for obs in self._get_all_obstacles_as_spheres():
            d = obs.distance_to(self._position) - self.collision_radius
            if d < min_obs_dist:
                min_obs_dist = d

        info: Dict[str, Any] = {
            "step": self._current_step,
            "max_steps": self.max_steps,
            "position": self._position.copy(),
            "velocity": self._velocity.copy(),
            "speed": speed,
            "goal": self._goal.copy(),
            "distance_to_goal": dist,
            "battery_soc": self.battery.state_of_charge,
            "remaining_energy": self.battery.remaining_energy,
            "battery_depleted": self.battery.is_depleted,
            "wind_vector": wind_state.total.copy(),
            "wind_speed": wind_state.speed,
            "num_static_obstacles": len(self._static_obstacles),
            "num_dynamic_obstacles": len(self._dynamic_obstacles),
            "min_obstacle_distance": min_obs_dist,
            "altitude": float(self._position[2]),
            "disturbance_magnitude": float(self._last_disturbance_magnitude),
            "disturbance_threshold": float(self.disturbance_event_threshold),
        }
        info.update(self._recovery.telemetry())
        return info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the disturbed drone navigation environment."""
        super().reset(seed=seed, options=options)
        self._current_step = 0

        # Optional parameter overrides via reset options
        num_static = self.num_obstacles
        num_dynamic = self.num_dynamic_obstacles
        if options:
            if "num_obstacles" in options:
                num_static = options["num_obstacles"]
            if "num_dynamic_obstacles" in options:
                num_dynamic = options["num_dynamic_obstacles"]

        self._position = self.default_start.copy()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._goal = self.default_goal.copy()

        self.kinematics.reset(self._position, self._velocity)
        self.wind_field.reset()
        self.battery.reset()
        self._injected_disturbance = np.zeros(3, dtype=np.float64)
        self._last_disturbance_magnitude = 0.0
        self._recovery.reset()

        # Generate static obstacle field
        self._static_obstacles = generate_drone_obstacles(
            bounds=self.bounds,
            start_pos=self._position,
            goal_pos=self._goal,
            num_obstacles=num_static,
            obstacle_radius=self.obstacle_radius,
            clearance_radius=self.collision_radius + 2.0,
            rng=self.np_random,
        )

        # Generate dynamic obstacles
        self._dynamic_obstacles = generate_dynamic_drone_obstacles(
            bounds=self.bounds,
            start_pos=self._position,
            goal_pos=self._goal,
            num_obstacles=num_dynamic,
            obstacle_radius=self.obstacle_radius,
            speed=self.dynamic_obstacle_speed,
            clearance_radius=self.collision_radius + 3.0,
            rng=self.np_random,
        )

        self._prev_distance_to_goal = float(np.linalg.norm(self._goal - self._position))

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one continuous 3D acceleration action step under environmental disturbances."""
        act_arr = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(act_arr):
            raise ValueError(f"Action {act_arr} outside action space bounds {self.action_space}")

        self._current_step += 1

        # 1. Advance stochastic wind gusts and the hidden injected disturbance
        self.wind_field.step_gust(self.np_random)
        self._advance_disturbance()
        wind_state = self.wind_field.get_wind(self._position)

        # 2. Command acceleration and apply wind + disturbance force
        raw_cmd_acc = act_arr * self.kinematics.max_acceleration
        # Wind drag produces a drift force proportional to wind speed: F_wind = c_d * w.
        # The injected disturbance acts through the same force path but is hidden
        # from the wind observation channels.
        effective_acc_cmd = raw_cmd_acc + self.linear_damping * (
            wind_state.total + self._injected_disturbance
        )

        new_pos, new_vel = self.kinematics.step(effective_acc_cmd)
        self._position = new_pos
        self._velocity = new_vel

        # 2b. Update disturbance/recovery event tracking (environment-step contract)
        transient = wind_state.gust + self._injected_disturbance
        self._last_disturbance_magnitude = float(np.linalg.norm(transient))
        self._recovery.update(
            step=self._current_step,
            magnitude=self._last_disturbance_magnitude,
            speed=float(np.linalg.norm(self._velocity)),
            threshold=self.disturbance_event_threshold,
            speed_tolerance=self.recovery_speed_tolerance,
            hold_steps=self.recovery_hold_steps,
        )

        # 3. Advance dynamic obstacles
        for dyn_obs in self._dynamic_obstacles:
            dyn_obs.step(self.dt)

        # 4. Advance battery consumption
        batt_telem = self.battery.step(raw_cmd_acc, self._velocity, dt=self.dt)

        # 5. Evaluate progress and spatial state
        curr_distance = float(np.linalg.norm(self._goal - self._position))
        dist_delta = self._prev_distance_to_goal - curr_distance
        self._prev_distance_to_goal = curr_distance

        # Collision & Goal checks
        is_collision, collision_type = self._check_collision(self._position)
        is_success = curr_distance <= self.target_radius
        is_exhausted = batt_telem.is_depleted

        terminated = False
        truncated = False
        info = self._get_info()
        info["collision"] = is_collision
        info["collision_type"] = collision_type
        info["success"] = is_success
        info["battery_exhausted"] = is_exhausted

        if is_collision:
            reward = self.collision_reward
            info["success"] = False
            if self.terminate_on_collision:
                terminated = True
        elif is_success:
            reward = self.goal_reward
            info["success"] = True
            terminated = True
        elif is_exhausted and self.terminate_on_exhaustion:
            reward = self.battery_exhaustion_penalty
            info["success"] = False
            terminated = True
        else:
            action_effort = float(np.sum(np.square(act_arr)))
            progress_reward = self.progress_weight * dist_delta
            effort_penalty = self.action_penalty_weight * action_effort
            # Small bonus for high remaining battery reserve
            energy_efficiency_bonus = 0.01 * batt_telem.state_of_charge
            reward = float(
                progress_reward + self.step_penalty - effort_penalty + energy_efficiency_bonus
            )

        if self._current_step >= self.max_steps and not terminated:
            truncated = True

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render textual 3D flight deck dashboard showing wind and battery status."""
        dist = float(np.linalg.norm(self._goal - self._position))
        speed = float(np.linalg.norm(self._velocity))
        pos_str = f"[{self._position[0]:5.1f}, {self._position[1]:5.1f}, {self._position[2]:5.1f}]"
        vel_str = f"[{self._velocity[0]:5.1f}, {self._velocity[1]:5.1f}, {self._velocity[2]:5.1f}]"
        goal_str = f"[{self._goal[0]:5.1f}, {self._goal[1]:5.1f}, {self._goal[2]:5.1f}]"

        wind_state = self.wind_field.get_wind(self._position)
        wind_str = f"[{wind_state.total[0]:4.1f}, {wind_state.total[1]:4.1f}, {wind_state.total[2]:4.1f}] m/s ({wind_state.speed:3.1f} m/s)"

        soc_pct = int(self.battery.state_of_charge * 100)
        bar_len = 10
        filled = int(self.battery.state_of_charge * bar_len)
        battery_bar = f"[{'=' * filled}{' ' * (bar_len - filled)}] {soc_pct:02d}%"

        lines = [
            "+----------------------------------------------------------------+",
            "|      AUTONOMOUS 3D DRONE FLIGHT DECK (DISTURBED / CONSTRAINED) |",
            "+----------------------------------------------------------------+",
            f"| Step: {self._current_step:03d}/{self.max_steps:03d} | Altitude (Z): {self._position[2]:5.1f}m | Speed: {speed:4.1f} m/s          |",
            f"| Position [X, Y, Z]: {pos_str:<28} |",
            f"| Velocity [Vx,Vy,Vz]: {vel_str:<28} |",
            f"| Waypoint [Gx,Gy,Gz]: {goal_str:<28} |",
            f"| Ambient Wind Field: {wind_str:<28} |",
            f"| Battery Reserve:    {battery_bar:<28} |",
            f"| Range to Target: {dist:5.1f}m | Obstacles: {len(self._static_obstacles):02d} static, {len(self._dynamic_obstacles):02d} moving      |",
            "+----------------------------------------------------------------+",
        ]
        dashboard = "\n".join(lines)

        if self.render_mode == "human":
            print(dashboard)
            return None
        return dashboard

    @property
    def battery_capacity(self) -> float:
        """Battery capacity property for curriculum inspection."""
        return self.battery.capacity

    @battery_capacity.setter
    def battery_capacity(self, value: float) -> None:
        """Set battery capacity dynamically and reset battery state."""
        self.battery.capacity = float(value)
        self.battery.reset()

    @property
    def wind_enabled(self) -> bool:
        """Check if environmental wind disturbances are active."""
        return (
            float(np.linalg.norm(self.wind_field.steady_wind)) > 0.0
            or self.wind_field.gust_sigma > 0.0
        )

    @wind_enabled.setter
    def wind_enabled(self, value: bool) -> None:
        """Enable or disable environmental wind vector fields."""
        if not value:
            self.wind_field.steady_wind = np.zeros(3, dtype=np.float64)
            self.wind_field.gust_sigma = 0.0
            self.wind_field.reset()

    @property
    def wind_base_speed(self) -> float:
        """Magnitude of steady prevailing wind."""
        return float(np.linalg.norm(self.wind_field.steady_wind))

    @wind_base_speed.setter
    def wind_base_speed(self, value: float) -> None:
        """Set steady wind velocity vector from base speed."""
        val = float(value)
        self.wind_field.steady_wind = np.array([val, val * 0.3, 0.0], dtype=np.float64)

    @property
    def wind_gusts(self) -> bool:
        """Check whether stochastic gust turbulence is active."""
        return self.wind_field.gust_sigma > 0.0

    @wind_gusts.setter
    def wind_gusts(self, value: bool) -> None:
        """Toggle stochastic gust volatility."""
        if value:
            self.wind_field.gust_sigma = 0.4
        else:
            self.wind_field.gust_sigma = 0.0
            self.wind_field.reset()

    def set_parameters(self, **kwargs: Any) -> None:
        """Dynamically update environment parameters (e.g. from curriculum stages)."""
        for key, value in kwargs.items():
            if key == "battery_capacity":
                self.battery_capacity = float(value)
            elif key == "wind_enabled":
                self.wind_enabled = bool(value)
            elif key == "wind_base_speed":
                self.wind_base_speed = float(value)
            elif key == "wind_gusts":
                self.wind_gusts = bool(value)
            elif key == "wind_speed":
                if value is not None:
                    self._apply_wind_speed(float(value))
                    self.wind_speed = float(value)
            elif key == "disturbance_strength":
                if float(value) < 0.0:
                    raise ValueError(f"disturbance_strength cannot be negative, got {value}")
                self.disturbance_strength = float(value)
            elif key == "disturbance_theta":
                if float(value) < 0.0:
                    raise ValueError(f"disturbance_theta cannot be negative, got {value}")
                self.disturbance_theta = float(value)
            elif key == "disturbance_event_factor":
                if float(value) <= 0.0:
                    raise ValueError(f"disturbance_event_factor must be positive, got {value}")
                self.disturbance_event_factor = float(value)
            elif key == "recovery_speed_tolerance":
                if float(value) < 0.0:
                    raise ValueError(f"recovery_speed_tolerance cannot be negative, got {value}")
                self.recovery_speed_tolerance = float(value)
            elif key == "recovery_hold_steps":
                if int(value) < 1:
                    raise ValueError(f"recovery_hold_steps must be >= 1, got {value}")
                self.recovery_hold_steps = int(value)
            elif hasattr(self, key):
                setattr(self, key, value)
