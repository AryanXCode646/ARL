"""Autonomous 3D drone navigation environments, kinematics, and obstacle dynamics."""

from adaptive_rl.environments.drone.battery import BatteryModel, BatteryTelemetry
from adaptive_rl.environments.drone.disturbed_drone import DroneDisturbance3DEnv
from adaptive_rl.environments.drone.drone3d import DroneNavigation3DEnv
from adaptive_rl.environments.drone.dynamic_obstacles import (
    DynamicObstacleSphere3D,
    generate_dynamic_drone_obstacles,
)
from adaptive_rl.environments.drone.kinematics import DroneKinematics3D, DroneState3D
from adaptive_rl.environments.drone.obstacles import (
    ObstacleSphere3D,
    compute_lidar_3d_readings,
    generate_drone_obstacles,
    generate_lidar_3d_ray_directions,
    ray_cast_box_boundaries_3d,
    ray_cast_sphere_3d,
)
from adaptive_rl.environments.drone.wind import WindField3D, WindState3D

__all__ = [
    "BatteryModel",
    "BatteryTelemetry",
    "DroneDisturbance3DEnv",
    "DroneKinematics3D",
    "DroneNavigation3DEnv",
    "DroneState3D",
    "DynamicObstacleSphere3D",
    "ObstacleSphere3D",
    "WindField3D",
    "WindState3D",
    "compute_lidar_3d_readings",
    "generate_drone_obstacles",
    "generate_dynamic_drone_obstacles",
    "generate_lidar_3d_ray_directions",
    "ray_cast_box_boundaries_3d",
    "ray_cast_sphere_3d",
]
